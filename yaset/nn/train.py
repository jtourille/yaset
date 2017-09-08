import logging
import math
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

from .helpers import TrainLogger
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
            "x_id": tf.FixedLenFeature([], dtype=tf.string)
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
            context_parsed["x_id"],
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
        shuffle_queue = tf.RandomShuffleQueue(nb_instances//2, nb_instances//4, dtypes=[tf.string, tf.int32, tf.int32,
                                                                                        tf.float32])

        # Enqueue and dequeue Ops + queue runner creation
        enqueue_op_shuffle_queue = shuffle_queue.enqueue(tensor_list)
        inputs = shuffle_queue.dequeue()
        queue_runner_list.append(tf.train.QueueRunner(shuffle_queue, [enqueue_op_shuffle_queue] * 1))

        # Bucketing according to bucket boundaries passed as arguments
        length, batch = tf.contrib.training.bucket_by_sequence_length(inputs[1], inputs, batch_size,
                                                                      sorted(buckets),
                                                                      num_threads=4,
                                                                      capacity=32,
                                                                      shapes=[[], [], [None], [None, None]],
                                                                      dynamic_pad=True)

        return queue_runner_list, [filename_queue, shuffle_queue], batch


def _build_dev_pipeline(tfrecords_file_path, num_classes=None, batch_size=None, nb_instances=None):
    """
    Build the dev pipeline
    :param tfrecords_file_path: dev TFRecords file path
    :param num_classes: number of labels (for one-hot encoding)
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

        # Main queue
        padding_queue = tf.PaddingFIFOQueue(nb_instances, dtypes=[tf.string, tf.int32, tf.int32, tf.float32],
                                            shapes=[[], [], [None], [None, None]])

        # Enqueue and dequeue Ops + queue runner creation
        enqueue_op = padding_queue.enqueue(tensor_list)
        batch = padding_queue.dequeue_many(batch_size)
        queue_runner_list.append(tf.train.QueueRunner(padding_queue, [enqueue_op] * 1))

        return queue_runner_list, [filename_queue, padding_queue], batch


def train_model(working_dir, embedding_object, data_object, train_config):

    config_tf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_tf.intra_op_parallelism_threads = train_config["cpu_cores"]
    config_tf.inter_op_parallelism_threads = train_config["cpu_cores"]

    logging.info("Building computation graph")

    # Clearing TensorFlow computation graph
    logging.debug("-> Resetting TensorFlow graph")
    tf.reset_default_graph()

    # Creating TensorFlow thread coordinator
    logging.debug("-> Creating coordinator")
    coord = tf.train.Coordinator()

    # Computing bucket boundaries for bucketing
    logging.debug("-> Computing bucket boundaries for train instances")
    train_bucket_boundaries = compute_bucket_boundaries(data_object.length_train_instances, train_config["batch_size"])

    # Fetching 'train' and 'dev' instance counts
    train_nb_examples = data_object.nb_train_instances
    dev_nb_examples = data_object.nb_dev_instances

    # Computing TFRecords file paths
    tfrecords_train_file_path = os.path.join(os.path.abspath(working_dir), "tfrecords", "train.tfrecords")
    tfrecords_dev_file_path = os.path.join(os.path.abspath(working_dir), "tfrecords", "dev.tfrecords")

    # Building 'train' input pipeline sub-graph
    logging.debug("-> Building train input pipeline")
    queue_runner_list_train, queue_list_train,\
        batch_train = _build_train_pipeline(tfrecords_train_file_path,
                                            train_bucket_boundaries,
                                            num_classes=len(data_object.label_mapping),
                                            batch_size=train_config["batch_size"],
                                            nb_instances=train_nb_examples)

    # Building 'dev' input pipeline sub-graph
    logging.debug("-> Building dev input pipeline")
    queue_runner_list_dev, queue_list_dev,\
        batch_dev = _build_dev_pipeline(tfrecords_dev_file_path,
                                        num_classes=len(data_object.label_mapping),
                                        batch_size=train_config["batch_size"],
                                        nb_instances=dev_nb_examples)

    # Network parameters for **kwargs usage
    model_args = {
        "word_embedding_matrix_shape": embedding_object.embedding_matrix.shape,
        "pl_dropout": tf.placeholder(tf.float32),
        "pl_emb": tf.placeholder(tf.float32, [embedding_object.embedding_matrix.shape[0],
                                              embedding_object.embedding_matrix.shape[1]]),
        "lstm_hidden_size": 256,
        "output_size": 3
    }

    # Creating main computation sub-graph
    logging.debug("-> Instantiating NN model (train and dev)")
    with tf.name_scope('train'):
        model_train = BiLSTMCRF(batch_train, reuse=False, **model_args)

    # Creating dev computation sub-graph, setting reuse to 'true' for weight sharing
    with tf.name_scope('dev'):
        model_dev = BiLSTMCRF(batch_dev, reuse=True, **model_args)

    # Initialization Op
    with tf.device('/cpu:0'):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Creating TensorFlow Session object
    logging.debug("-> Creating TensorFlow session and initializing graph")
    sess = tf.Session(config=config_tf)

    # Initializing variables and embedding matrix
    sess.run(init)
    sess.run(model_train.embedding_tokens_init, {model_args["pl_emb"]: embedding_object.embedding_matrix})

    # Launching threads and starting TensorFlow queue runners
    logging.debug("-> Launching threads")
    threads_train = [item.create_threads(sess, coord=coord, start=True) for item in queue_runner_list_train]
    threads_dev = [item.create_threads(sess, coord=coord, start=True) for item in queue_runner_list_dev]

    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Computing the 5% threeshold for logging
    display_every_n_train = math.ceil((train_nb_examples //
                                       train_config["batch_size"]) * 0.05) * train_config["batch_size"]

    display_every_n_dev = math.ceil((dev_nb_examples //
                                     train_config["batch_size"]) * 0.05) * train_config["batch_size"]

    logging.info("Learning !")

    iteration_number = 1
    train_counter = 0

    train_logger = TrainLogger()

    # Looping until max iteration is reached
    while iteration_number <= train_config["max_iterations"]:

        # Resetting the counter if an iteration has been completed
        if train_counter >= train_nb_examples:
            train_counter = 0
            iteration_number += 1

        # Starting evaluation on dev corpus if an iteration has been completed
        if iteration_number - 1 not in train_logger and iteration_number - 1 != 0:

            logging.info("End iteration {}".format(iteration_number - 1))
            logging.info("Evaluating on dev corpus")

            params = {
                model_args["pl_dropout"]: 1.0
            }

            dev_counter = 0

            labels_corr = 0
            labels_pred = 0

            done = set()

            while dev_counter < dev_nb_examples:

                x_id, x_len, y_pred, y_target = sess.run([batch_dev[0], batch_dev[1], model_dev.prediction,
                                                          batch_dev[3]], feed_dict=params)

                dev_counter += train_config["batch_size"]
                cur_percentage = (float(dev_counter) / dev_nb_examples) * 100

                for seq_id_, seq_len_, unary_scores_, y_target_ in zip(x_id, x_len, y_pred, y_target):

                    seq_id_str = seq_id_.decode("UTF-8")

                    if seq_id_str in done:
                        continue
                    else:
                        done.add(seq_id_str)

                    # Decoding with Viterbi
                    unary_scores_ = unary_scores_[:seq_len_]
                    viterbi_sequence,\
                        viterbi_score = tf.contrib.crf.viterbi_decode(unary_scores_,
                                                                      sess.run(model_dev.transitions_params))

                    # Counting incorrect and correct predictions
                    for label_pred, label_gs in zip(viterbi_sequence, y_target_):
                        target_id = np.argmax(label_gs)

                        labels_pred += 1
                        if data_object.label_mapping[target_id] == data_object.label_mapping[label_pred]:
                            labels_corr += 1

                # Logging progress
                if dev_counter % display_every_n_dev == 0 or cur_percentage >= 100:
                    logging.info("* processed={} ({:5.2f}%)".format(
                        dev_counter,
                        round(cur_percentage, 2),
                    ))

            # Computing token accuracy
            accuracy = float(labels_corr) / labels_pred
            logging.info("Accuracy: {}".format(accuracy))

            # Adding iteration score to train logger object
            train_logger.add_score(iteration_number - 1, accuracy)

        # Setting dropout to 0.5 for learning
        params = {
            model_args["pl_dropout"]: train_config["dropout_rate"]
        }

        # Optimizing with one mini-batch
        _, loss = sess.run([model_train.optimize, model_train.loss], feed_dict=params)

        # Incrementing counter and computing completion
        train_counter += 256
        cur_percentage = (float(train_counter) / train_nb_examples) * 100

        # Logging training progress
        if train_counter % display_every_n_train == 0 or cur_percentage >= 100:
            logging.info("* epoch={} ({:5.2f}%), loss={:7.4f}, processed={}".format(
                iteration_number,
                round(cur_percentage, 2),
                loss,
                train_counter
            ))

    # Stopping everything gracefully
    logging.info("Stopping everything gracefully (or at least trying to)")

    coord.request_stop()

    for item in queue_list_train:
        item.close(cancel_pending_enqueues=True)

    for item in queue_list_dev:
        item.close(cancel_pending_enqueues=True)

    for item in threads_train:
        coord.join(item)

    for item in threads_dev:
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
