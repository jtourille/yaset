import functools
import tensorflow as tf
import numpy as np
import logging


def lazy_property(function):

    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class BiLSTMCRF:

    def __init__(self, batch, reuse=False, test=False, **kwargs):

        self.reuse = reuse
        self.test = test

        self.use_char_embeddings = kwargs["use_char_embeddings"]
        self.char_embedding_size = kwargs["char_embedding_matrix_shape"][1]
        self.char_lstm_num_hidden = kwargs["char_lstm_num_hidden"]

        self.x_tokens_len = batch[1]
        self.x_tokens_fw = batch[2]

        self.x_tokens_bw = tf.reverse_sequence(self.x_tokens_fw, self.x_tokens_len, seq_dim=1)

        # -----------------------------------------------------------

        if self.use_char_embeddings:

            self.x_chars_fw = batch[3]
            self.x_chars_len = batch[4]

            # We reverse the sequence for the backward computation
            # 1 - Reshape the matrix in [batch_size * sequence_length, character_length]
            self.x_chars_fw_reshaped = tf.reshape(self.x_chars_fw,
                                                  [tf.shape(self.x_chars_fw)[0] * tf.shape(self.x_chars_fw)[1],
                                                   tf.shape(self.x_chars_fw)[2]])
            # 2 - Flatten the character length tensor
            self.x_chars_len_reshaped = tf.reshape(self.x_chars_len, [-1])
            # 3 - Reverse the character sequences
            self.x_chars_bw = tf.reverse_sequence(self.x_chars_fw_reshaped, self.x_chars_len_reshaped, seq_dim=1)
            # 4 - Reshape back the matrix to [batch_size, sequence_length, character_length]
            self.x_chars_bw = tf.reshape(self.x_chars_bw, [tf.shape(self.x_chars_fw)[0],
                                                           tf.shape(self.x_chars_fw)[1],
                                                           tf.shape(self.x_chars_fw)[2]])

        # -----------------------------------------------------------

        if not test:
            self.y = batch[5]
        else:
            self.y = tf.placeholder(tf.int32, shape=[None, None])

        self.pl_dropout = kwargs["pl_dropout"]

        if not self.reuse and not self.test:
            self.pl_emb = kwargs["pl_emb"]

        self.lstm_hidden_size = kwargs["lstm_hidden_size"]
        self.output_size = kwargs["output_size"]

        logging.debug("-> Matrices")

        with tf.device('/cpu:0'):
            with tf.variable_scope('matrices', reuse=self.reuse):

                self.W = tf.get_variable('embedding_matrix_words',
                                         dtype=tf.float32,
                                         shape=[kwargs["word_embedding_matrix_shape"][0],
                                                kwargs["word_embedding_matrix_shape"][1]],
                                         initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                         trainable=kwargs["trainable_word_embeddings"])

                self.transition_params = tf.get_variable('transition_params',
                                                         dtype=tf.float32,
                                                         shape=[kwargs["output_size"], kwargs["output_size"]],
                                                         initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                                         trainable=True)

                if self.use_char_embeddings:
                    self.C = tf.get_variable('embedding_matrix_chars',
                                             dtype=tf.float32,
                                             initializer=self._get_weight(kwargs["char_embedding_matrix_shape"][0],
                                                                          kwargs["char_embedding_matrix_shape"][1]),
                                             trainable=True)

            if not self.reuse and not self.test:
                self.embedding_tokens_init = self.W.assign(self.pl_emb)

        with tf.device("/cpu:0"):

            logging.debug("-> Embedding lookups")

            self.embed_words_fw
            self.embed_words_bw

            if self.use_char_embeddings:
                self.embed_chars_fw
                self.embed_chars_bw

        if self.use_char_embeddings:
            self.char_representation

        logging.debug("-> Forward and Backward representations")

        self.forward_representation
        self.backward_representation

        logging.debug("-> Predictions")

        self.prediction

        logging.debug("-> Loss")
        with tf.variable_scope('loss', reuse=self.reuse):
            self.loss = self.loss_and_transitions

        logging.debug("-> Optimization")
        if not self.reuse and not self.test:
            self.optimize

    @lazy_property
    def embed_words_fw(self):

        with tf.device('/cpu:0'):
            embed_words = tf.nn.embedding_lookup(self.W, self.x_tokens_fw, name='lookup_tokens_fw')

        return embed_words

    @lazy_property
    def embed_words_bw(self):

        with tf.device('/cpu:0'):
            embed_words = tf.nn.embedding_lookup(self.W, self.x_tokens_bw, name='lookup_tokens_bw')

        return embed_words

    @lazy_property
    def embed_chars_fw(self):

        with tf.device('/cpu:0'):
            embed_chars = tf.nn.embedding_lookup(self.C, self.x_chars_fw, name='lookup_chars_fw')

        return embed_chars

    @lazy_property
    def embed_chars_bw(self):

        with tf.device('/cpu:0'):
            embed_chars = tf.nn.embedding_lookup(self.C, self.x_chars_bw, name='lookup_chars_bw')

        return embed_chars

    @lazy_property
    def char_representation(self):

        embed_fw = self.embed_chars_fw
        embed_bw = self.embed_chars_bw

        reshaped_len = tf.reshape(self.x_chars_len, [-1])

        # Reshaping character embedding batch for LSTM processing [batch_size * seq_length, char_length, char_emb_size]
        input_chars_fw = tf.reshape(embed_fw,
                                    [tf.shape(embed_fw)[0] * tf.shape(embed_fw)[1],
                                     tf.shape(embed_fw)[2],
                                     self.char_embedding_size])

        input_chars_bw = tf.reshape(embed_bw,
                                    [tf.shape(embed_bw)[0] * tf.shape(embed_bw)[1],
                                     tf.shape(embed_bw)[2],
                                     self.char_embedding_size])

        with tf.variable_scope('chars_forward'):
            fw_cell_chars = tf.contrib.rnn.LSTMCell(self.char_lstm_num_hidden, state_is_tuple=True, reuse=self.reuse)

            outputs_fw_chars, states_fw_chars = tf.nn.dynamic_rnn(cell=fw_cell_chars,
                                                                  inputs=input_chars_fw,
                                                                  dtype=tf.float32,
                                                                  sequence_length=reshaped_len)

        with tf.variable_scope('chars_backward'):
            bw_cell_chars = tf.contrib.rnn.LSTMCell(self.char_lstm_num_hidden, state_is_tuple=True, reuse=self.reuse)

            outputs_bw_chars, states_bw_chars = tf.nn.dynamic_rnn(cell=bw_cell_chars,
                                                                  inputs=input_chars_bw,
                                                                  dtype=tf.float32,
                                                                  sequence_length=reshaped_len)

        # Setting the 0 padding values to 1
        len_clip = tf.clip_by_value(reshaped_len, 1, tf.shape(embed_fw)[2])

        # Creating partitions
        partitions = tf.one_hot(len_clip - 1, depth=tf.shape(embed_fw)[2], dtype=tf.int32)

        # Gathering outputs
        select_fw = tf.dynamic_partition(outputs_fw_chars, partitions, 2)
        select_bw = tf.dynamic_partition(outputs_bw_chars, partitions, 2)

        # Concatenating forward and backward outputs
        char_vector = tf.concat([select_fw[1], select_bw[1]], 1)

        # Reshaping the output
        final_output = tf.reshape(char_vector,
                                  [tf.shape(embed_fw)[0], tf.shape(embed_fw)[1], self.char_lstm_num_hidden * 2])

        return final_output

    @lazy_property
    def forward_representation(self):

        if self.use_char_embeddings:
            vector = tf.concat([self.embed_words_fw, self.char_representation], 2)
        else:
            vector = self.embed_words_fw

        with tf.variable_scope('forward_representation', reuse=self.reuse):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.pl_dropout)

            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=vector,
                                               dtype=tf.float32,
                                               sequence_length=self.x_tokens_len)

        return outputs

    @lazy_property
    def backward_representation(self):

        if self.use_char_embeddings:
            char_vector_bw = tf.reverse_sequence(self.char_representation, self.x_tokens_len, seq_dim=1)
            vector = tf.concat([self.embed_words_bw, char_vector_bw], 2)
        else:
            vector = self.embed_words_bw

        with tf.variable_scope('backward_representation', reuse=self.reuse):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.pl_dropout)

            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=vector,
                                               dtype=tf.float32,
                                               sequence_length=self.x_tokens_len)

        return outputs

    @lazy_property
    def prediction(self):

        with tf.variable_scope('prediction', reuse=self.reuse):
            weight = tf.get_variable('prediction_weights',
                                     initializer=self._get_weight(2 * self.lstm_hidden_size, 2 * self.lstm_hidden_size))
            bias = tf.get_variable('prediction_bias',
                                   initializer=self._get_bias(2 * self.lstm_hidden_size))

            proj_weight = tf.get_variable('projection_weights',
                                          initializer=self._get_weight(2 * self.lstm_hidden_size, self.output_size))
            proj_bias = tf.get_variable('projection_bias',
                                        initializer=self._get_bias(self.output_size))

            final_vector = tf.concat([self.forward_representation, self.backward_representation], 2)
            final_vector = tf.reshape(final_vector, [-1, 2 * self.lstm_hidden_size])

            prediction = tf.add(tf.matmul(final_vector, weight), bias)
            prediction = tf.tanh(prediction)

            prediction = tf.add(tf.matmul(prediction, proj_weight), proj_bias)

            prediction = tf.reshape(prediction, [-1, tf.shape(self.x_tokens_fw)[1], self.output_size])

        return prediction

    @lazy_property
    def loss_and_transitions(self):

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.prediction, self.y, self.x_tokens_len,
                                                              transition_params=self.transition_params)

        loss = tf.reduce_mean(-log_likelihood)

        return loss

    @lazy_property
    def optimize_adam(self):

        optimizer = tf.train.AdamOptimizer(0.001)

        return optimizer.minimize(self.loss)

    @lazy_property
    def optimize_adadelta(self):

        optimizer = tf.train.AdadeltaOptimizer(0.001)

        return optimizer.minimize(self.loss)

    @lazy_property
    def optimize(self):
        """
        SGD with gradient clipping of 5
        :return:
        """

        optimizer = tf.train.GradientDescentOptimizer(0.005)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        return train_op

    @staticmethod
    def _get_weight(in_size, out_size):
        min_val = -np.divide(np.sqrt(6), np.sqrt(np.add(in_size, out_size)))
        max_val = np.divide(np.sqrt(6), np.sqrt(np.add(in_size, out_size)))

        return tf.random_uniform([in_size, out_size], minval=min_val, maxval=max_val)

    @staticmethod
    def _get_bias(out_size):

        return tf.constant(0.0, shape=[out_size])
