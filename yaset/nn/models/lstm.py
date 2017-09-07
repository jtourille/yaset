import functools
import tensorflow as tf
import numpy as np


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

    def __init__(self, batch, reuse=False, **kwargs):

        self.x_tokens_len = batch[0]
        self.x_tokens_fw = batch[1]
        self.x_tokens_bw = tf.reverse_sequence(self.x_tokens_fw, self.x_tokens_len, seq_dim=1)
        self.y = batch[2]

        self.reuse = reuse

        self.pl_dropout = kwargs["pl_dropout"]
        self.pl_emb = kwargs["pl_emb"]
        self.lstm_hidden_size = kwargs["lstm_hidden_size"]
        self.output_size = kwargs["output_size"]

        # self.transitions_params = tf.Variable([self.output_size, self.output_size], dtype=np.float32)
        self.transitions_params = tf.get_variable('transition_params',
                                                  dtype=tf.float32,
                                                  shape=[self.output_size,self.output_size],
                                                  initializer=tf.random_uniform_initializer(0., 1.),
                                                  trainable=True)

        with tf.device('/cpu:0'):
            # Initializing main word embedding matrix
            self.W = tf.get_variable('embedding_matrix_word',
                                     dtype=tf.float32,
                                     shape=[kwargs["word_embedding_matrix_shape"][0],
                                            kwargs["word_embedding_matrix_shape"][1]],
                                     initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                     trainable=True)
            self.embedding_tokens_init = self.W.assign(self.pl_emb)

        with tf.device("/cpu:0"):

            self.embed_words_fw
            self.embed_words_bw

            self.forward_representation
            self.backward_representation

            self.prediction
            self.loss
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
    def forward_representation(self):

        with tf.variable_scope('forward_representation', reuse=self.reuse):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.pl_dropout)

            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=self.embed_words_fw,
                                               dtype=tf.float32,
                                               sequence_length=self.x_tokens_len)

        return outputs

    @lazy_property
    def backward_representation(self):

        with tf.variable_scope('backward_representation', reuse=self.reuse):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.pl_dropout)

            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=self.embed_words_bw,
                                               dtype=tf.float32,
                                               sequence_length=self.x_tokens_len)

        return outputs

    @lazy_property
    def prediction(self):

        with tf.variable_scope('prediction', reuse=self.reuse):

            weight = tf.get_variable('pred_w',
                                     initializer=self._get_weight(2 * self.lstm_hidden_size, self.output_size))
            bias = tf.get_variable('pred_b',
                                   initializer=self._get_bias(self.output_size))

            final_vector = tf.concat([self.forward_representation, self.backward_representation], 2)
            final_vector = tf.reshape(final_vector, [-1, 2 * self.lstm_hidden_size])

            prediction = tf.add(tf.matmul(final_vector, weight), bias)
            prediction = tf.reshape(prediction, [-1, tf.shape(self.x_tokens_fw)[1], self.output_size])

        return prediction

    @lazy_property
    def loss(self):

        target_reshaped = tf.to_int32(tf.argmax(self.y, 2))
        log_likelihood, self.transitions_params = tf.contrib.crf.crf_log_likelihood(self.prediction, target_reshaped,
                                                                                    self.x_tokens_len)

        loss = tf.reduce_mean(-log_likelihood)

        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        # cross_entropy = tf.reduce_mean(cross_entropy)

        return loss

    @lazy_property
    def optimize(self):

        optimizer = tf.train.AdamOptimizer(0.001)
        gvs = optimizer.compute_gradients(self.loss)
        # capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        return optimizer.apply_gradients(gvs)

    @staticmethod
    def _get_weight(in_size, out_size):
        min_val = -np.divide(np.sqrt(6), np.sqrt(np.add(in_size, out_size)))
        max_val = np.divide(np.sqrt(6), np.sqrt(np.add(in_size, out_size)))

        return tf.random_uniform([in_size, out_size], minval=min_val, maxval=max_val)

    @staticmethod
    def _get_bias(out_size):

        return tf.constant(0.0, shape=[out_size])
