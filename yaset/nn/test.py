import json
import logging
import math
import os

import numpy as np
import tensorflow as tf

from .helpers import get_best_model
from .models.lstm import BiLSTMCRF
from ..data.reader import TestData


def read_and_decode_test(filename_queue, feature_columns):
    """
    Read and decode one example from a TFRecords file
    :param feature_columns: list of feature columns
    :param filename_queue: filename queue containing the TFRecords filenames
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
            "x_chars": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "x_chars_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        for col in feature_columns:
            sequence_features["x_att_{}".format(col)] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

        # Parsing contextual and sequential features
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        sequence_length = tf.cast(context_parsed["x_length"], tf.int32)
        chars = tf.reshape(sequence_parsed["x_chars"], tf.stack([sequence_length, -1]))

        # Preparing tensor list, casting values to 32 bits when necessary
        tensor_list = [
            context_parsed["x_id"],
            tf.cast(context_parsed["x_length"], tf.int32),
            tf.cast(sequence_parsed["x_tokens"], dtype=tf.int32),
            tf.cast(chars, dtype=tf.int32),
            tf.cast(sequence_parsed["x_chars_len"], dtype=tf.int32),
        ]

        for col in feature_columns:
            tensor_list.append(tf.cast(sequence_parsed["x_att_{}".format(col)], dtype=tf.int32))

        return tensor_list


def test_model(working_dir, model_dir, data_object: TestData, data_params, train_params, model_params, n_jobs=1):
    """
    Apply model on test data
    :param working_dir: current working directory
    :param model_dir: yaset model path
    :param data_object: TestData object
    :param n_jobs: number of cores to use
    :return: nothing
    """

    # Setting some TensorFlow session parameters
    config_tf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_tf.intra_op_parallelism_threads = n_jobs
    config_tf.inter_op_parallelism_threads = n_jobs

    # Load data characteristics from log file
    train_data_char = json.load(open(os.path.join(model_dir, "data_char.json")))

    logging.info("Building computation graph")

    # Clearing TensorFlow computation graph
    logging.debug("-> Resetting TensorFlow graph")
    tf.reset_default_graph()

    # Creating TensorFlow thread coordinator
    logging.debug("-> Creating coordinator")
    coord = tf.train.Coordinator()

    nb_examples = data_object.test_stats.nb_instances

    tfrecords_file_path = os.path.join(os.path.abspath(working_dir), "data.tfrecords")

    # Building 'dev' input pipeline sub-graph
    logging.debug("-> Building input pipeline")
    queue_runner_list, queue_list, \
        batch = _build_test_pipeline(tfrecords_file_path,
                                     data_object.feature_columns,
                                     batch_size=64,
                                     nb_instances=nb_examples)

    # Network parameters for **kwargs usage
    model_args = {

        **train_params,
        **model_params,


        "word_embedding_matrix_shape": train_data_char.get("embedding_matrix_shape"),
        "char_count": len(data_object.char_mapping),

        "pl_dropout": tf.placeholder(tf.float32),

        # "char_embedding_matrix_shape": [len(data_object.char_mapping),
        #                                 train_params.get("char_embedding_size")],
        "char_lstm_num_hidden": train_params.get("char_hidden_layer_size"),

        "output_size": len(train_data_char["label_mapping"])
    }

    # Creating main computation sub-graph
    logging.debug("-> Instantiating NN model")
    with tf.name_scope('train'):
        if train_params["model_type"] == "bilstm-char-crf":
            model = BiLSTMCRF(batch, reuse=False, test=True, **model_args)
        else:
            raise Exception("The model type ou specified does not exist: {}".format(train_params["model_type"]))

    # Initialization Op
    with tf.device('/cpu:0'):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Retrieving model filename based on training statistics
    tf_model_saver_path = os.path.join(model_dir, "tfmodels")
    train_stats_file = os.path.join(model_dir, "train_stats.json")
    best_filename = os.path.join(tf_model_saver_path, get_best_model(train_stats_file))

    saver = tf.train.Saver()

    # Creating TensorFlow Session object
    logging.debug("-> Creating TensorFlow session and initializing graph")
    sess = tf.Session(config=config_tf)

    # Initializing variables and embedding matrix
    sess.run(init)

    # Restoring model
    logging.info("Loading saved model into TensorFlow session")
    saver.restore(sess, best_filename)

    # Launching threads and starting TensorFlow queue runners
    logging.debug("-> Launching threads")
    threads = [item.create_threads(sess, coord=coord, start=True) for item in queue_runner_list]

    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    logging.info("Processing data !")

    counter = 0

    params = {
        model_args["pl_dropout"]: 0.0
    }

    display_every_n = math.ceil((nb_examples // 32) * 0.05) * 32

    if display_every_n == 0:
        display_every_n = 32

    pred_sequences = dict()
    done = set()

    while counter < nb_examples:

        x_id, x_len, y_pred = sess.run([batch[0], batch[1], model.prediction], feed_dict=params)

        counter += 64
        cur_percentage = (float(counter) / nb_examples) * 100

        for seq_id_, seq_len_, unary_scores_ in zip(x_id, x_len, y_pred):

            seq_id_str = seq_id_.decode("UTF-8")

            if seq_id_str in done:
                continue
            else:
                done.add(seq_id_str)

            unary_scores_ = unary_scores_[:seq_len_]

            # Tiling and adding START and END tokens
            start_unary_scores = [[-1000.0] * unary_scores_.shape[1] + [0.0, -1000.0]]
            end_unary_tensor = [[-1000.0] * unary_scores_.shape[1] + [-1000.0, 0.0]]

            tile = np.tile(np.array([-1000.0, -1000.0], dtype=np.float32), [unary_scores_.shape[0], 1])

            tiled_tensor = np.concatenate([unary_scores_, tile], 1)

            tensor_start_end = np.concatenate([start_unary_scores, tiled_tensor, end_unary_tensor], 0)

            viterbi_sequence, \
                viterbi_score = tf.contrib.crf.viterbi_decode(tensor_start_end,
                                                              sess.run(model.transition_params))

            pred_sequences[seq_id_str] = viterbi_sequence[1:-1]

        # Logging progress
        if counter % display_every_n == 0 or cur_percentage >= 100:
            logging.info("* processed={} ({:.2f}%)".format(
                counter,
                round(cur_percentage, 2),
            ))

    target_output_file = os.path.join(working_dir, "output.conll")
    data_object.write_predictions_to_file(target_output_file, pred_sequences)
    logging.info("Writing prediction to file")

    # Stopping everything gracefully
    logging.info("Stopping everything gracefully (or at least trying to)")

    logging.debug("* Requesting stop")
    coord.request_stop()

    logging.debug("* Closing pipeline queues")
    for item in queue_list:
        item.close(cancel_pending_enqueues=True)

    logging.debug("* Closing pipeline threads")
    for item in threads:
        coord.join(item)

    sess.close()


def _build_test_pipeline(tfrecords_file_path, feature_columns, batch_size=None, nb_instances=None):
    """
    Build the test pipeline
    :param tfrecords_file_path: test TFRecords file path
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
        tensor_list = read_and_decode_test(filename_queue, feature_columns)

        dtypes = [tf.string, tf.int32, tf.int32, tf.int32, tf.int32]
        shapes = [[], [], [None], [None, None], [None]]

        for _ in feature_columns:
            dtypes.append(tf.int32)
            shapes.append([None])

        # Main queue
        padding_queue = tf.PaddingFIFOQueue(nb_instances, dtypes=dtypes, shapes=shapes)

        # Enqueue and dequeue Ops + queue runner creation
        enqueue_op = padding_queue.enqueue(tensor_list)
        batch = padding_queue.dequeue_many(batch_size)
        queue_runner_list.append(tf.train.QueueRunner(padding_queue, [enqueue_op] * 1))

        return queue_runner_list, [filename_queue, padding_queue], batch