import logging
import math
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from .helpers import TrainLogger, conll_evaluation, compute_bucket_boundaries
from .models.lstm import BiLSTMCRF
from ..data.reader import TrainData
from ..tools import ensure_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_and_decode(filename_queue, feature_columns):
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
            "y": tf.FixedLenSequenceFeature([], dtype=tf.int64),
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
            tf.cast(sequence_parsed["y"], dtype=tf.int32)
        ]

        for col in feature_columns:
            tensor_list.append(tf.cast(sequence_parsed["x_att_{}".format(col)], dtype=tf.int32))

        return tensor_list


def _build_train_pipeline(tfrecords_file_path, feature_columns, buckets=None, batch_size=None,
                          nb_instances=None):
    """
    Build the train pipeline. Sequences are grouped into buckets for faster training.
    :param tfrecords_file_path: train TFRecords file path
    :param buckets: train buckets
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
        tensor_list = read_and_decode(filename_queue, feature_columns)

        dtypes = [tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
        for _ in feature_columns:
            dtypes.append(tf.int32)

        # Random shuffle queue, allow for randomization of training instances (maximum size: 50% of nb. instances)
        shuffle_queue = tf.RandomShuffleQueue(nb_instances, nb_instances//2, dtypes=dtypes)

        # Enqueue and dequeue Ops + queue runner creation
        enqueue_op_shuffle_queue = shuffle_queue.enqueue(tensor_list)
        inputs = shuffle_queue.dequeue()

        queue_runner_list.append(tf.train.QueueRunner(shuffle_queue, [enqueue_op_shuffle_queue] * 4))

        shapes = [[], [], [None], [None, None], [None], [None]]
        for _ in feature_columns:
            shapes.append([None])

        if buckets:
            # Bucketing according to bucket boundaries passed as arguments
            length, batch = tf.contrib.training.bucket_by_sequence_length(inputs[1], inputs, batch_size,
                                                                          sorted(buckets),
                                                                          num_threads=4,
                                                                          capacity=32,
                                                                          shapes=shapes,
                                                                          dynamic_pad=True)
        else:

            padding_queue = tf.PaddingFIFOQueue(nb_instances, dtypes=dtypes, shapes=shapes)
            enqueue_op_padding_queue = padding_queue.enqueue(inputs)
            batch = padding_queue.dequeue_many(batch_size)

            queue_runner_list.append(tf.train.QueueRunner(padding_queue, [enqueue_op_padding_queue] * 4))

        return queue_runner_list, [filename_queue, shuffle_queue], batch


def _build_dev_pipeline(tfrecords_file_path, feature_columns, batch_size=None, nb_instances=None):
    """
    Build the dev pipeline
    :param tfrecords_file_path: dev TFRecords file path
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
        tensor_list = read_and_decode(filename_queue, feature_columns)

        dtypes = [tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
        shapes = [[], [], [None], [None, None], [None], [None]]

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


def train_model(working_dir, embedding_object, data_object: TrainData, train_params, model_params):

    config_tf = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_tf.intra_op_parallelism_threads = train_params["cpu_cores"]
    config_tf.inter_op_parallelism_threads = train_params["cpu_cores"]
    config_tf.gpu_options.allow_growth = True

    logging.info("Building computation graph")

    # Clearing TensorFlow computation graph
    logging.debug("* Resetting TensorFlow graph")
    tf.reset_default_graph()

    # Creating TensorFlow thread coordinator
    logging.debug("* Creating coordinator")
    coord = tf.train.Coordinator()

    train_bucket_boundaries = None

    if train_params["bucket_use"]:
        # Computing bucket boundaries for bucketing
        logging.debug("* Computing bucket boundaries")
        train_bucket_boundaries = compute_bucket_boundaries(
            data_object.train_stats.sequence_lengths, train_params["batch_size"])
        logging.debug("* Bucket boundaries for train instances: {}".format(sorted(train_bucket_boundaries)))

    # Fetching 'train' and 'dev' instance counts
    train_nb_examples = data_object.train_stats.nb_instances
    dev_nb_examples = data_object.dev_stats.nb_instances

    # Computing TFRecords file paths
    tfrecords_train_file_path = os.path.join(os.path.abspath(working_dir), "tfrecords", "train.tfrecords")
    tfrecords_dev_file_path = os.path.join(os.path.abspath(working_dir), "tfrecords", "dev.tfrecords")

    # Building 'train' input pipeline sub-graph
    logging.debug("* Building 'train' input pipeline")
    queue_runner_list_train, queue_list_train,\
        batch_train = _build_train_pipeline(tfrecords_train_file_path,
                                            data_object.feature_columns,
                                            buckets=train_bucket_boundaries,
                                            batch_size=train_params["batch_size"],
                                            nb_instances=train_nb_examples)

    # Building 'dev' input pipeline sub-graph
    logging.debug("* Building 'dev' input pipeline")
    queue_runner_list_dev, queue_list_dev,\
        batch_dev = _build_dev_pipeline(tfrecords_dev_file_path,
                                        data_object.feature_columns,
                                        batch_size=train_params["batch_size"],
                                        nb_instances=dev_nb_examples)

    # Network parameters for **kwargs usage
    model_args = {

        # Train and model params
        **train_params,
        **model_params,

        # Embedding shapes
        "word_embedding_matrix_shape": embedding_object.embedding_matrix.shape,
        "char_count": len(data_object.char_mapping),
        # "char_embedding_matrix_shape": [len(data_object.char_mapping), model_params["char_embedding_size"]],

        # Misc
        "pl_dropout": tf.placeholder(tf.float32),
        "pl_emb": tf.placeholder(tf.float32, [embedding_object.embedding_matrix.shape[0],
                                              embedding_object.embedding_matrix.shape[1]]),
        "output_size": len(data_object.label_mapping),

        "train_nb_instances": train_nb_examples,
        "pl_global_counter": tf.placeholder(tf.int32)
    }

    model_train = None
    model_dev = None

    # Creating main computation sub-graph
    logging.debug("* Instantiating NN model ('train')")
    with tf.name_scope('train'):
        if train_params["model_type"] == "bilstm-char-crf":
            model_train = BiLSTMCRF(batch_train, reuse=False, test=False, **model_args)
        else:
            raise Exception("The model type ou specified does not exist: {}".format(train_params["model_type"]))

    # Creating dev computation sub-graph, setting reuse to 'true' for weight sharing
    logging.debug("* Instantiating NN model ('dev')")
    with tf.name_scope('dev'):
        if train_params["model_type"] == "bilstm-char-crf":
            model_dev = BiLSTMCRF(batch_dev, reuse=True, test=False, **model_args)
        else:
            raise Exception("The model type ou specified does not exist: {}".format(train_params["model_type"]))

    # Initialization Op
    with tf.device('/cpu:0'):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # TensorFlow model saver
    tf_model_saver_path = os.path.join(os.path.abspath(working_dir), "tfmodels")
    tf_model_saving_name = os.path.join(tf_model_saver_path, "model.ckpt")
    ensure_dir(tf_model_saver_path)

    saver = tf.train.Saver(max_to_keep=0)

    # Creating TensorFlow Session object
    logging.debug("* Creating TensorFlow session and initializing computation graph (variables + embeddings)")
    sess = tf.Session(config=config_tf)

    # Initializing variables and embedding matrix
    sess.run(init)
    sess.run(model_train.embedding_tokens_init, {model_args["pl_emb"]: embedding_object.embedding_matrix})

    # Launching threads and starting TensorFlow queue runners
    logging.debug("* Launching threads and TensorFlow queue runners")
    threads_train = [item.create_threads(sess, coord=coord, start=True) for item in queue_runner_list_train]
    threads_dev = [item.create_threads(sess, coord=coord, start=True) for item in queue_runner_list_dev]

    _ = tf.train.start_queue_runners(sess=sess, coord=coord)

    logging.info("Zajiganié !")

    iteration_number = 1
    train_counter_global = 0

    train_logger = TrainLogger()
    train_logger_dump_filename = os.path.join(os.path.abspath(working_dir), "train_stats.json")

    # Looping until max iteration is reached
    while iteration_number <= train_params["max_iterations"]:

        train_counter_global += _do_one_iteration(train_nb_examples, train_params, model_params,
                                                  model_args, train_counter_global, model_train,
                                                  sess, iteration_number)

        # Resetting the counter if an iteration has been completed
        iteration_number += 1

        # Starting evaluation on dev corpus if an iteration has been completed
        if iteration_number - 1 not in train_logger and iteration_number - 1 != 0:

            logging.info("End iteration {}".format(iteration_number - 1))
            logging.info("Evaluating on dev corpus")

            do_break = _evaluate_on_dev(model_args, dev_nb_examples, sess, batch_dev, model_dev, train_params,
                                        data_object, saver, tf_model_saving_name, iteration_number, train_logger,
                                        tf_model_saver_path)

            if do_break:
                break

    logging.info("Iteration scores\n\n{}\n".format(train_logger.get_score_table()))

    logging.info("Saving model characteristics")

    logging.debug("* Dumping train logger")
    train_logger.save_to_file(train_logger_dump_filename)

    logging.debug("* Dumping data characteristics")
    target_data_characteristics_file = os.path.join(working_dir, 'data_char.json')
    data_object.dump_data_characteristics(target_data_characteristics_file, embedding_object)

    # Stopping everything gracefully
    logging.info("Stopping everything gracefully (or at least trying to)")

    logging.debug("* Requesting stop")
    coord.request_stop()

    logging.debug("* Closing 'train' pipeline queues")
    for item in queue_list_train:
        item.close(cancel_pending_enqueues=True)

    logging.debug("* Closing 'dev' pipeline queues")
    for item in queue_list_dev:
        item.close(cancel_pending_enqueues=True)

    logging.debug("* Closing 'train' pipeline threads")
    for item in threads_train:
        coord.join(item)

    logging.debug("* Closing 'dev' pipeline threads")
    for item in threads_dev:
        coord.join(item)

    sess.close()


def _do_one_iteration(train_nb_examples, train_params, model_params, model_args,
                      train_counter_global, model_train, sess, iteration_number):

    # Computing the 5% threeshold for logging
    display_every_n_train = math.ceil((train_nb_examples //
                                       train_params["batch_size"]) * 0.05) * train_params["batch_size"]

    if display_every_n_train == 0:
        display_every_n_train = train_params["batch_size"]

    train_counter = 0

    while train_counter < train_nb_examples:

        # Setting dropout for learning (defined by user)
        params = {
            model_args["pl_dropout"]: model_params["dropout_rate"],
            model_args["pl_global_counter"]: train_counter_global
        }

        # Processing one mini-batch
        _, loss = sess.run([model_train.optimize, model_train.loss_crf], feed_dict=params)

        # Incrementing counter and computing completion
        train_counter += train_params["batch_size"]
        train_counter_global += train_params["batch_size"]

        cur_percentage = (float(train_counter) / train_nb_examples) * 100

        # Logging training progress
        if train_counter % display_every_n_train == 0 or cur_percentage >= 100:
            if cur_percentage >= 100:
                logging.info("* epoch={} ({:5.2f}%), loss={:7.4f}, processed_iter={}, processed_global={}".format(
                    iteration_number,
                    round(100.0, 2),
                    -loss,
                    train_counter,
                    train_counter_global
                ))
            else:
                logging.info("* epoch={} ({:5.2f}%), loss={:7.4f}, processed_iter={}, processed_global={}".format(
                    iteration_number,
                    round(cur_percentage, 2),
                    -loss,
                    train_counter,
                    train_counter_global
                ))

    return train_counter


def _evaluate_on_dev(model_args, dev_nb_examples, sess, batch_dev, model_dev, train_params, data_object,
                     saver, tf_model_saving_name, iteration_number, train_logger, tf_model_saver_path):

    display_every_n_dev = math.ceil((dev_nb_examples //
                                     train_params["batch_size"]) * 0.05) * train_params["batch_size"]

    if display_every_n_dev == 0:
        display_every_n_dev = train_params["batch_size"]

    params = {
        model_args["pl_dropout"]: 0.0
    }

    dev_counter = 0
    done = set()
    metric_payload = list()

    while dev_counter < dev_nb_examples:

        x_id, x_len, y_pred, y_target = sess.run([batch_dev[0], batch_dev[1], model_dev.prediction,
                                                  batch_dev[5]], feed_dict=params)

        dev_counter += train_params["batch_size"]
        cur_percentage = (float(dev_counter) / dev_nb_examples) * 100

        for seq_id_, seq_len_, unary_scores_, y_target_ in zip(x_id, x_len, y_pred, y_target):

            curr_seq_metric_payload = list()

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
                                                          sess.run(model_dev.transition_params))

            # Counting incorrect and correct predictions
            for label_pred, label_gs in zip(viterbi_sequence[1:-1], y_target_):
                curr_seq_metric_payload.append({
                    "gs": data_object.inv_label_mapping[label_gs],
                    "pred": data_object.inv_label_mapping[label_pred]
                })

            metric_payload.append(curr_seq_metric_payload)

        # Logging progress
        if dev_counter % display_every_n_dev == 0 or cur_percentage >= 100:
            if cur_percentage >= 100:
                logging.info("* processed={} ({:5.2f}%)".format(
                    dev_nb_examples,
                    round(100.0, 2),
                ))
            else:
                logging.info("* processed={} ({:5.2f}%)".format(
                    dev_counter,
                    round(cur_percentage, 2),
                ))

                # Computing token accuracy
    pred_labels = list()
    gs_labels = list()

    for seq in metric_payload:
        for tok in seq:
            pred_labels.append(tok["pred"])
            gs_labels.append(tok["gs"])

    accuracy = accuracy_score(gs_labels, pred_labels)

    if train_params["dev_metric"] == "accuracy":

        logging.info("Accuracy: {}".format(accuracy))
        logging.debug("* nb. pred.: {:,}".format(len(pred_labels)))

        score = accuracy

    elif train_params["dev_metric"] == "conll":

        precision, recall, f1 = conll_evaluation(metric_payload)

        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Precision: {}".format(precision))
        logging.info("Recall: {}".format(recall))
        logging.info("F1: {}".format(f1))

        score = f1

    else:
        raise Exception("The 'dev' metric you specified does not exist: {}".format(train_params["dev_metric"]))

    logging.debug("* nb. pred.: {:,}".format(len(pred_labels)))

    model_name = saver.save(sess, tf_model_saving_name, global_step=iteration_number - 1)

    # Adding iteration score to train logger object
    train_logger.add_iteration_score(iteration_number - 1, score)
    train_logger.add_iteration_model_filename(iteration_number - 1, model_name)

    logging.info("Model has been saved at: {}".format(model_name))

    if iteration_number - 1 != 1:
        logging.info("Cleaning model directory (saving space)")
        _delete_models(train_logger.get_removable_iterations(), tf_model_saver_path)

    # Quitting main loop if patience is reached
    if train_logger.check_patience(train_params["patience"]):
        logging.info("Patience reached, quitting main loop")
        return True
    else:
        return False


def _delete_models(indices, model_dir):
    """
    Clean model snapshot directory
    :param indices: iteration IDs
    :param model_dir: model path
    :return: nothing
    """

    # Building regular expression
    regex = re.compile("model.ckpt-({})\.".format("|".join([str(i) for i in indices])))

    # Deleting files
    for root, dirs, files in os.walk(os.path.abspath(model_dir)):
        for filename in files:
            if regex.match(filename):
                os.remove(os.path.join(root, filename))



