import logging
import math
import os
from collections import defaultdict

import tensorflow as tf

from .models.lstm import BiLSTMCRF


def read_and_decode(filename_queue, target_size):
    """
    Read and decode one example from a TFRecords file
    :param filename_queue: filename queue containing the TFRecords filenames
    :param target_size: number of classes for 'one-hot-vector' conversion
    :return: list of tensors representing one example
    """

    with tf.device('/cpu:0'):

        # New TFRecord file
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Contextual TFRecords features
        context_features = {
            "x_length": tf.FixedLenFeature([], dtype=tf.int64),
        }

        # Sequential TFRecords features
        sequence_features = {
            "x_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "y": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parsing contextual and sequential features
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        # Preparing tensor list, casting values to 32 bits when necessary
        tensor_list = [
            tf.cast(context_parsed["x_length"], tf.int32),
            tf.cast(sequence_parsed["x_tokens"], dtype=tf.int32),
            tf.one_hot(indices=tf.cast(sequence_parsed["y"], dtype=tf.int32), depth=target_size),
        ]

        return tensor_list


def _build_train_pipeline(tfrecords_file_path, buckets, num_classes=None, batch_size=None, nb_instances=None):
    """
    Build the train pipeline. Sequences are grouped into buckets for faster training.
    :param tfrecords_file_path: train TFRecords file path
    :param buckets: train buckets
    :param num_classes: number of labels (for one-hot encoding)
    :param batch_size: mini-batch size
    :return: queue runner list, queues, symbolic link to mini-batch
    """

    with tf.device('/cpu:0'):

        # Creating a list with tfrecords
        tfrecords_list = [tfrecords_file_path]

        # Will contains queue runners for thread creation
        queue_runner_list = list()

        # Filename queue, contains only on filename (train TFRecords file)
        filename_queue = tf.train.string_input_producer(tfrecords_list)

        # Decode one example
        tensor_list = read_and_decode(filename_queue, num_classes)

        # Random shuffle queue, allow for randomization of training instances (maximum size: 50% of nb. instances)
        shuffle_queue = tf.RandomShuffleQueue(nb_instances//2, nb_instances//4, dtypes=[tf.int32, tf.int32, tf.float32])

        # Enqueue and dequeue Ops + queue runner creation
        enqueue_op_shuffle_queue = shuffle_queue.enqueue(tensor_list)
        inputs = shuffle_queue.dequeue()
        queue_runner_list.append(tf.train.QueueRunner(shuffle_queue, [enqueue_op_shuffle_queue] * 1))

        # Bucketing according to bucket boundaries passed as arguments
        length, batch = tf.contrib.training.bucket_by_sequence_length(inputs[0], inputs, batch_size,
                                                                      sorted(buckets),
                                                                      num_threads=4,
                                                                      capacity=32,
                                                                      shapes=[[], [None], [None, None]],
                                                                      dynamic_pad=True)

        return queue_runner_list, [filename_queue, shuffle_queue], batch


# def _build_dev_pipeline(tfrecords_file_path, num_classes=None):
#
#     with tf.device('/cpu:0'):
#
#         tfrecords_list = [tfrecords_file_path]
#
#         filename_queue = tf.train.string_input_producer(tfrecords_list)
#
#         tensor_list = read_and_decode(filename_queue, num_classes)
#
#         padding_queue = tf.PaddingFIFOQueue(1000, dtypes=[tf.int32, tf.int32, tf.float32],
#                                             shapes=[[], [None], [None, None]])
#
#         enqueue_op = padding_queue.enqueue(tensor_list)
#         batch = padding_queue.dequeue_many(1)
#
#         queue_runner_list = list()
#         queue_runner_list.append(tf.train.QueueRunner(padding_queue, [enqueue_op] * 1))
#
#         return queue_runner_list, batch


def train_model(working_dir, embedding_object, data_object, train_config):

    config_tf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_tf.intra_op_parallelism_threads = train_config["cpu_cores"]
    config_tf.inter_op_parallelism_threads = train_config["cpu_cores"]

    logging.info("Building computation graph")
    logging.debug("-> Resetting TensorFlow graph")
    # Clearing TensorFlow computation graph
    tf.reset_default_graph()

    logging.debug("-> Creating coordinator")
    # Creating TensorFlow thread coordinator
    coord = tf.train.Coordinator()

    logging.debug("-> Computing bucket boundaries")
    train_bucket_boundaries = compute_bucket_boundaries(data_object.length_train_instances, train_config["batch_size"])
    # dev_bucket_boundaries = compute_bucket_boundaries(data_object.length_dev_instances, batch_size)

    train_nb_examples = data_object.nb_train_instances

    tfrecords_train_file_path = os.path.join(os.path.abspath(working_dir), "tfrecords", "train.tfrecords")
    # tfrecords_dev_file_path = os.path.join(os.path.abspath(working_dir), "tfrecords", "dev.tfrecords")

    logging.debug("-> Building train input pipeline")
    queue_runner_list_train, queue_list_train, batch_train = _build_train_pipeline(tfrecords_train_file_path,
                                                                                   train_bucket_boundaries,
                                                                                   num_classes=len(data_object.
                                                                                                   label_mapping),
                                                                                   batch_size=train_config["batch_size"])

    # queue_runner_list_dev, batch_dev = _build_dev_pipeline(tfrecords_dev_file_path, num_classes=num_classes)

    model_args = {
        "word_embedding_matrix_shape": embedding_object.embedding_matrix.shape,
        "pl_dropout": tf.placeholder(tf.float32),
        "pl_emb": tf.placeholder(tf.float32, [embedding_object.embedding_matrix.shape[0],
                                              embedding_object.embedding_matrix.shape[1]]),
        "lstm_hidden_size": 256,
        "output_size": 3
    }

    logging.debug("-> Instantiating NN model")
    with tf.name_scope('train'):
        model_train = BiLSTMCRF(batch_train, reuse=False, **model_args)

    with tf.device('/cpu:0'):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    logging.debug("-> Creating TensorFlow session and initializing graph")
    sess = tf.Session(config=config_tf)

    sess.run(init)
    sess.run(model_train.embedding_tokens_init, {model_args["pl_emb"]: embedding_object.embedding_matrix})

    logging.debug("-> Launching threads")
    threads_train = [item.create_threads(sess, coord=coord, start=True) for item in queue_runner_list_train]
    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    params = {
        model_args["pl_dropout"]: 0.5
    }

    display_every_n = math.ceil((train_nb_examples // train_config["batch_size"]) * 0.05) * train_config["batch_size"]

    logging.info("Learning !")

    iteration_number = 1
    counter = 0

    while iteration_number < train_config["max_iterations"]:

        if counter >= train_nb_examples:
            counter = 0

        _, loss = sess.run([model_train.optimize, model_train.loss], feed_dict=params)
        counter += 256
        cur_percentage = (float(counter) / train_nb_examples) * 100

        if counter % display_every_n == 0 or cur_percentage >= 100:

            logging.info("* epoch={} ({}%), loss={}, processed={}".format(
                iteration_number,
                round(cur_percentage, 2),
                loss,
                counter
            ))

        if counter >= train_nb_examples:
            iteration_number += 1

    logging.info("Stopping everything gracefully (or at least trying to)")
    coord.request_stop()
    for item in queue_list_train:
        item.close(cancel_pending_enqueues=True)

    for item in threads_train:
        coord.join(item)

    sess.close()


def compute_bucket_boundaries(sequence_lengths, batch_size):
    """
    Compute bucket boundaries based on the sequence lengths
    :param sequence_lengths: sequence length to consider
    :param batch_size: mini-batch size used for learning
    :return: buckets boundaries (list)
    """

    # Step 1 - Gather number of sequences per length
    seq_count_by_length = defaultdict(int)

    for length in sequence_lengths:
        seq_count_by_length[length] += 1

    # Step 2 - Compute bucket boundaries
    # Each buckets must contains at least four mini-batches
    output_buckets = list()
    current_count = 0

    for len_seq, nb_seq in sorted(seq_count_by_length.items(), reverse=True):

        current_count += nb_seq

        if current_count >= batch_size * 4:
            output_buckets.append(len_seq)
            current_count = 0

    if current_count < batch_size * 3 and current_count != 0:
        output_buckets.pop(-1)

    return sorted(output_buckets)
