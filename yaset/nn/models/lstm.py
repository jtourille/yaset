import functools
import logging

import numpy as np
import tensorflow as tf


def lazy_property(func):
    """
    Decorator taken from the blog post located at https://danijar.com/structuring-your-tensorflow-models/
    :param func: function you want to decorate
    :return: decorated function
    """

    attribute = '_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper


class BiLSTMCRF:
    """
    Neural Network model based on Lample et al. (2016)
    """

    def __init__(self, batch, reuse=False, test=False, **kwargs):

        self.reuse = reuse
        self.test = test

        self.train_config = dict()
        for k, v in kwargs.items():
            self.train_config[k] = v

        self.use_char_embeddings = self.train_config["use_char_embeddings"]
        # self.char_embedding_size = self.train_config["char_embedding_matrix_shape"][1]
        # self.char_lstm_num_hidden = self.train_config["char_hidden_layer_size"]

        if not self.test:
            self.global_counter = self.train_config["pl_global_counter"]

        self.output_size = self.train_config["output_size"]
        self.lstm_hidden_size = self.train_config["hidden_layer_size"]

        self.pl_dropout = self.train_config["pl_dropout"]

        # If not in dev not test phase
        if not self.reuse and not self.test:
            self.pl_emb = self.train_config["pl_emb"]

        self.x_tokens = batch[2]
        self.x_tokens_len = batch[1]

        self.x_chars = batch[3]
        self.x_chars_len = batch[4]

        # Using a dummy placeholder for test set
        if not test:
            self.y = batch[5]
        else:
            self.y = tf.placeholder(tf.int32, shape=[None, None])

        logging.debug("-> Creating matrices")

        if self.train_config["store_matrices_on_gpu"]:
            device_str = "/gpu:0"
        else:
            device_str = "/cpu:0"

        with tf.device(device_str):
            with tf.variable_scope('matrices', reuse=self.reuse):

                self.W = tf.get_variable('embedding_matrix_words',
                                         dtype=tf.float32,
                                         shape=[self.train_config["word_embedding_matrix_shape"][0],
                                                self.train_config["word_embedding_matrix_shape"][1]],
                                         initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                         trainable=self.train_config["trainable_word_embeddings"])

                self.transition_params = tf.get_variable('transition_params',
                                                         dtype=tf.float32,
                                                         shape=[self.train_config["output_size"] + 2,
                                                                self.train_config["output_size"] + 2],
                                                         initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                                         trainable=True)

                if self.use_char_embeddings:
                    self.C = tf.get_variable('embedding_matrix_chars',
                                             dtype=tf.float32,
                                             initializer=self._get_weight(
                                                 self.train_config["char_count"],
                                                 self.train_config["char_embedding_size"]),
                                             trainable=True)

            if not self.reuse and not self.test:
                self.embedding_tokens_init = self.W.assign(self.pl_emb)

        with tf.device(device_str):

            logging.debug("-> Embedding lookups")

            self.embed_words

            if self.use_char_embeddings:
                self.embed_chars

        if self.use_char_embeddings:
            self.char_representation

        logging.debug("-> Forward and Backward representations")

        self.main_lstm

        logging.debug("-> Predictions")

        self.prediction

        # If not in dev nor test phase
        if not self.reuse and not self.test:
            logging.debug("-> Optimization")
            self.optimize

    @lazy_property
    def embed_words(self):
        """
        Word embedding lookup
        :return: tf.nn.embedding_lookup object
        """

        embed_words = tf.nn.embedding_lookup(self.W, self.x_tokens, name='lookup_tokens')

        return embed_words

    @lazy_property
    def embed_chars(self):
        """
        Character embedding lookup
        :return: tf.nn.embedding_lookup object
        """

        embed_chars = tf.nn.embedding_lookup(self.C, self.x_chars, name='lookup_chars')

        return embed_chars

    @lazy_property
    def char_representation(self):
        """
        Build a character based representation of the tokens
        :return: character based representation [batch_size, seq_size, char_lstm_num_hidden * 2]
        """

        char_lstm_num_hidden = self.train_config["char_hidden_layer_size"]

        # Reshaping token lengths tensor []
        reshaped_len = tf.reshape(self.x_chars_len, [-1])

        # Reshaping character embedding batch for LSTM processing [batch_size * seq_length, char_length, char_emb_size]
        input_chars = tf.reshape(self.embed_chars,
                                 [tf.shape(self.embed_chars)[0] * tf.shape(self.embed_chars)[1],
                                  tf.shape(self.embed_chars)[2],
                                  self.train_config["char_embedding_size"]])

        with tf.variable_scope('char_representation', reuse=self.reuse):

            lstm_cell_fw = tf.contrib.rnn.LSTMCell(char_lstm_num_hidden, state_is_tuple=True)
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(char_lstm_num_hidden, state_is_tuple=True)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw,
                cell_bw=lstm_cell_bw,
                dtype=tf.float32,
                sequence_length=reshaped_len,
                inputs=input_chars)

        # Concatenating forward and backward outputs
        char_vector = tf.concat([states[0].h, states[1].h], 1)

        # Reshaping the output [batch_size, seq_len, char_lstm_num_hidden * 2]
        final_output = tf.reshape(char_vector,
                                  [tf.shape(self.embed_chars)[0], tf.shape(self.embed_chars)[1],
                                   char_lstm_num_hidden * 2])

        return final_output

    @lazy_property
    def main_lstm(self):

        # Depending on the use of characters in the final representation
        if self.use_char_embeddings:
            # Concatenate forward word embedding and forward character embedding representations
            input_tensor = tf.concat([self.embed_words, self.char_representation], 2)
        else:
            # Use only forward word embedding representation
            input_tensor = self.embed_words

        input_tensor = tf.nn.dropout(input_tensor, 1.0 - self.pl_dropout)

        with tf.variable_scope('main_lstm', reuse=self.reuse):

            lstm_cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw,
                cell_bw=lstm_cell_bw,
                dtype=tf.float32,
                sequence_length=self.x_tokens_len,
                inputs=input_tensor)

        return tf.concat(outputs, 2)

    @lazy_property
    def prediction(self):
        """
        Predict unary scores
        :return: unary scores [batch_size, seq_len, num_labels]
        """

        with tf.variable_scope('prediction', reuse=self.reuse):

            # Initializing weights and bias for last layer
            last_layer_weights = tf.get_variable('last_layer_weights',
                                                 initializer=self._get_weight(2 * self.lstm_hidden_size,
                                                                              2 * self.lstm_hidden_size))

            last_layer_bias = tf.get_variable('last_layer_bias',
                                              initializer=self._get_bias(2 * self.lstm_hidden_size))

            # Preparing input tensor for last layer
            tensor_bef_last_layer = tf.reshape(self.main_lstm, [-1, 2 * self.lstm_hidden_size])

            # Last layer computation (tanh activation)
            last_layer = tf.add(tf.matmul(tensor_bef_last_layer, last_layer_weights), last_layer_bias)
            last_layer = tf.tanh(last_layer)

            # Initializing weights and bias for projection layer
            projection_weights = tf.get_variable('projection_weights',
                                                 initializer=self._get_weight(2 * self.lstm_hidden_size,
                                                                              self.output_size))
            projections_bias = tf.get_variable('projection_bias',
                                               initializer=self._get_bias(self.output_size))

            # Projecting
            prediction = tf.add(tf.matmul(last_layer, projection_weights), projections_bias)

            # Reshaping output [batch_size, batch_len, nb_labels]
            prediction = tf.reshape(prediction, [-1, tf.shape(self.x_tokens)[1], self.output_size])

        return prediction

    @lazy_property
    def loss_crf(self):
        """
        CRF based loss.
        :return: loss
        """

        # Reshaping seq_len tensor [seq_len, 1]
        seq_length_reshaped = tf.reshape(self.x_tokens_len, [tf.shape(self.x_tokens_len)[0], -1])

        # Computing loss by scanning mini-batch tensor
        out = tf.scan(self.loss_crf_scan, [self.prediction,
                                           seq_length_reshaped,
                                           self.y], back_prop=True, infer_shape=True, initializer=0.0)

        # Division by batch_size
        loss_crf = tf.divide(tf.reduce_sum(out), tf.cast(tf.shape(self.x_tokens)[0], dtype=tf.float32))

        return loss_crf

    def loss_crf_scan(self, _, current_input):
        """
        Scan function for log likelihood computation
        :param _: previous output
        :param current_input: current tensor line
        :return: sequence log likelihood
        """

        # TILING

        # Create tiling for "start" and "end" scores
        tile = tf.tile(tf.constant(-1000.0, shape=[1, 2], dtype=tf.float32), [tf.shape(current_input[0])[0], 1])

        # Add two scores for each token in each sequence
        tiled_tensor = tf.concat([current_input[0], tile], 1)

        # -----------------------------------------------------------
        # ADDING START TOKEN

        cur_nb_class = current_input[0].get_shape().as_list()[1]

        # Create start and end token unary scores
        start_unary_scores = [[-1000.0] * cur_nb_class + [0.0, -1000.0]]
        end_unary_tensor = [[-1000.0] * cur_nb_class + [-1000.0, 0.0]]

        # Concatenate start unary scores to the tiled vector
        tensor_start = tf.concat([start_unary_scores, tiled_tensor], 0)

        # -----------------------------------------------------------
        # ADDING END TOKEN

        # Creating mask to fetch elements of the sequence
        mask = tf.sequence_mask(
            (tf.cast(tf.reshape(current_input[1], [-1]), dtype=tf.int32) + 1) * tf.shape(tensor_start)[1],
            tf.shape(tensor_start)[1] * tf.shape(tensor_start)[0],
            dtype=tf.int32)

        # Flattening unary scores and partitioning
        unary_scores_reshaped = tf.reshape(tensor_start, [1, -1])
        slices = tf.dynamic_partition(unary_scores_reshaped, mask, 2)

        # Reshaping slice one
        slice_1 = tf.reshape(slices[1], [-1, tf.shape(tensor_start)[1]])

        # Concatenating and reshaping
        tensor_start_end = tf.concat([slice_1, end_unary_tensor], 0)
        tensor_start_end_reshaped = tf.reshape(tensor_start_end,
                                               [1, tf.shape(tensor_start_end)[0], tf.shape(tensor_start_end)[1]])

        # Setting shape to tensor
        tensor_start_end_reshaped.set_shape([1, None, cur_nb_class + 2])

        # -----------------------------------------------------------
        # ADDING START AND END LABELS

        # Creating mask for target
        mask_y = tf.sequence_mask(
            (tf.cast(tf.reshape(current_input[1], [-1]), dtype=tf.int32)),
            tf.shape(current_input[0])[0],
            dtype=tf.int32
        )

        # Flattening label tensor and partitioning
        y_reshaped = tf.reshape(current_input[2], [1, -1])
        slices_y = tf.dynamic_partition(y_reshaped, mask_y, 2)

        # Concatenating and reshaping
        new_y = tf.concat([[cur_nb_class], slices_y[1], [cur_nb_class+1]], axis=0)
        new_y_reshaped = tf.reshape(new_y, [1, -1])

        # -----------------------------------------------------------
        # COMPUTING LOG LIKELIHOOD

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(tensor_start_end_reshaped, new_y_reshaped,
                                                              current_input[1],
                                                              transition_params=self.transition_params)

        return tf.reduce_sum(log_likelihood)

    @lazy_property
    def optimize(self):
        """
        SGD with gradient clipping of 5
        :return:
        """

        if self.train_config["opt_decay_use"]:

            learning_rate = tf.train.exponential_decay(self.train_config["opt_lr"], self.global_counter,
                                                       self.train_config["train_nb_instances"],
                                                       self.train_config["opt_decay_rate"],
                                                       staircase=True)
        else:
            learning_rate = self.train_config["opt_lr"]

        if self.train_config["opt_algo"] == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.train_config["opt_algo"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise Exception("The optimization algorithm you specified does not exist")

        if self.train_config["opt_gc_use"]:

            if self.train_config["opt_gc_type"] == "clip_by_norm":
                gradients, variables = zip(*optimizer.compute_gradients(-self.loss_crf))
                gradients, _ = tf.clip_by_global_norm(gradients, self.train_config["opt_gs_val"])
                train_op = optimizer.apply_gradients(zip(gradients, variables))

            elif self.train_config["opt_gc_type"] == "clip_by_value":
                gvs = optimizer.compute_gradients(-self.loss_crf)
                capped_gvs = [(tf.clip_by_value(grad, -self.train_config["opt_gs_val"],
                                                self.train_config["opt_gs_val"]), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs)
            else:
                raise Exception("The gradient clipping method you specified does not exist: {}".format(
                    self.train_config["opt_gc_type"]
                ))

        else:
            train_op = optimizer.minimize(-self.loss_crf)

        return train_op

    @staticmethod
    def _get_weight(in_size, out_size):
        """
        Weight matrix initialization following Xavier initialization
        :param in_size: input size
        :param out_size: output size
        :return: weight matrix
        """

        min_val = -np.divide(np.sqrt(6), np.sqrt(np.add(in_size, out_size)))
        max_val = np.divide(np.sqrt(6), np.sqrt(np.add(in_size, out_size)))

        return tf.random_uniform([in_size, out_size], minval=min_val, maxval=max_val)

    @staticmethod
    def _get_bias(out_size):
        """
        Bias matrix initialization
        :param out_size: output size
        :return: bias matrix
        """

        return tf.constant(0.0, shape=[out_size])
