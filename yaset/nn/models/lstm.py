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

        self.use_char_embeddings = kwargs["use_char_embeddings"]
        self.char_embedding_size = kwargs["char_embedding_matrix_shape"][1]
        self.char_lstm_num_hidden = kwargs["char_lstm_num_hidden"]

        self.x_tokens_len = batch[1]
        self.x_tokens_fw = batch[2]

        self.x_tokens_bw = tf.reverse_sequence(self.x_tokens_fw, self.x_tokens_len, seq_dim=1)

        self.output_size = kwargs["output_size"]

        if not self.test:
            self.opt_algo = kwargs["opt_algo"]
            self.opt_gc_use = kwargs["opt_gc_use"]
            self.opt_gc_val = kwargs["opt_gc_val"]
            self.opt_lr = kwargs["opt_lr"]

        # -----------------------------------------------------------
        # Character embeddings

        if self.use_char_embeddings:

            # Fetching the char-related tensors from input batch
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

        # Using a dummy placeholder for test set
        if not test:
            self.y = batch[5]
        else:
            self.y = tf.placeholder(tf.int32, shape=[None, None])

        self.pl_dropout = kwargs["pl_dropout"]

        # If not in dev not test phase
        if not self.reuse and not self.test:
            self.pl_emb = kwargs["pl_emb"]

        self.lstm_hidden_size = kwargs["lstm_hidden_size"]
        self.output_size = kwargs["output_size"]

        logging.debug("-> Creating matrices")

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
                                                         shape=[kwargs["output_size"] + 2, kwargs["output_size"] + 2],
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

        # If not in dev nor test phase
        if not self.reuse and not self.test:
            logging.debug("-> Optimization")
            self.optimize

    @lazy_property
    def embed_words_fw(self):
        """
        Forward embedding lookup
        :return: tf.nn.embedding_lookup object
        """

        with tf.device('/cpu:0'):
            embed_words = tf.nn.embedding_lookup(self.W, self.x_tokens_fw, name='lookup_tokens_fw')

        return embed_words

    @lazy_property
    def embed_words_bw(self):
        """
        Backward embedding lookup
        :return: tf.nn.embedding_lookup object
        """

        with tf.device('/cpu:0'):
            embed_words = tf.nn.embedding_lookup(self.W, self.x_tokens_bw, name='lookup_tokens_bw')

        return embed_words

    @lazy_property
    def embed_chars_fw(self):
        """
        Forward character lookup
        :return: tf.nn.embedding_lookup object
        """

        with tf.device('/cpu:0'):
            embed_chars = tf.nn.embedding_lookup(self.C, self.x_chars_fw, name='lookup_chars_fw')

        return embed_chars

    @lazy_property
    def embed_chars_bw(self):
        """
        Backward character lookup
        :return: tf.nn.embedding_lookup object
        """

        with tf.device('/cpu:0'):
            embed_chars = tf.nn.embedding_lookup(self.C, self.x_chars_bw, name='lookup_chars_bw')

        return embed_chars

    @lazy_property
    def char_representation(self):
        """
        Build a character based representation of the tokens
        :return: character based representation [batch_size, seq_size, char_lstm_num_hidden * 2]
        """

        # Retrieving character embeddings
        embed_fw = self.embed_chars_fw
        embed_bw = self.embed_chars_bw

        # Reshaping token lengths tensor []
        reshaped_len = tf.reshape(self.x_chars_len, [-1])

        # Reshaping character embedding batch for LSTM processing [batch_size * seq_length, char_length, char_emb_size]
        # Forward
        input_chars_fw = tf.reshape(embed_fw,
                                    [tf.shape(embed_fw)[0] * tf.shape(embed_fw)[1],
                                     tf.shape(embed_fw)[2],
                                     self.char_embedding_size])

        # Backward
        input_chars_bw = tf.reshape(embed_bw,
                                    [tf.shape(embed_bw)[0] * tf.shape(embed_bw)[1],
                                     tf.shape(embed_bw)[2],
                                     self.char_embedding_size])

        # Character LSTM forward pass
        with tf.variable_scope('chars_forward'):
            fw_cell_chars = tf.contrib.rnn.LSTMCell(self.char_lstm_num_hidden, state_is_tuple=True, reuse=self.reuse)

            outputs_fw_chars, states_fw_chars = tf.nn.dynamic_rnn(cell=fw_cell_chars,
                                                                  inputs=input_chars_fw,
                                                                  dtype=tf.float32,
                                                                  sequence_length=reshaped_len)

        # Character LSTM backward pass
        with tf.variable_scope('chars_backward'):
            bw_cell_chars = tf.contrib.rnn.LSTMCell(self.char_lstm_num_hidden, state_is_tuple=True, reuse=self.reuse)

            outputs_bw_chars, states_bw_chars = tf.nn.dynamic_rnn(cell=bw_cell_chars,
                                                                  inputs=input_chars_bw,
                                                                  dtype=tf.float32,
                                                                  sequence_length=reshaped_len)

        # Setting the 0 padding values to 1 to create the mask
        len_clip = tf.clip_by_value(reshaped_len, 1, tf.shape(embed_fw)[2])

        # Creating partition mask
        partitions = tf.one_hot(len_clip - 1, depth=tf.shape(embed_fw)[2], dtype=tf.int32)

        # Gathering outputs from forward and backward passes
        select_fw = tf.dynamic_partition(outputs_fw_chars, partitions, 2)
        select_bw = tf.dynamic_partition(outputs_bw_chars, partitions, 2)

        # Concatenating forward and backward outputs
        char_vector = tf.concat([select_fw[1], select_bw[1]], 1)

        # Reshaping the output [batch_size, seq_len, char_lstm_num_hidden * 2]
        final_output = tf.reshape(char_vector,
                                  [tf.shape(embed_fw)[0], tf.shape(embed_fw)[1], self.char_lstm_num_hidden * 2])

        return final_output

    @lazy_property
    def forward_representation(self):
        """
        Build forward representation of the sequence
        :return: forward LSTM output
        """

        # Depending on the use of characters in the final representation
        if self.use_char_embeddings:
            # Concatenate forward word embedding and forward character embedding representations
            fw_tensor = tf.concat([self.embed_words_fw, self.char_representation], 2)
        else:
            # Use only forward word embedding representation
            fw_tensor = self.embed_words_fw

        # Building forward representation
        with tf.variable_scope('forward_representation', reuse=self.reuse):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)

            # Applying dropout on input embeddings
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.pl_dropout)

            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=fw_tensor,
                                               dtype=tf.float32,
                                               sequence_length=self.x_tokens_len)

        return outputs

    @lazy_property
    def backward_representation(self):
        """
        Build forward representation of the sequence
        :return: forward LSTM output
        """

        # Depending on the use of characters in the final representation
        if self.use_char_embeddings:
            # Concatenate backward word embedding and forward character embedding representations
            # Reversing character based representation sequences
            char_vector_bw = tf.reverse_sequence(self.char_representation, self.x_tokens_len, seq_dim=1)
            bw_tensor = tf.concat([self.embed_words_bw, char_vector_bw], 2)
        else:
            # Use only backward word embedding representation
            bw_tensor = self.embed_words_bw

        # Building backward representation
        with tf.variable_scope('backward_representation', reuse=self.reuse):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size, state_is_tuple=True)

            # Applying dropout on input embeddings
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.pl_dropout)

            outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                               inputs=bw_tensor,
                                               dtype=tf.float32,
                                               sequence_length=self.x_tokens_len)

        return outputs

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
            tensor_bef_last_layer = tf.concat([self.forward_representation, self.backward_representation], 2)
            tensor_bef_last_layer = tf.reshape(tensor_bef_last_layer, [-1, 2 * self.lstm_hidden_size])

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
            prediction = tf.reshape(prediction, [-1, tf.shape(self.x_tokens_fw)[1], self.output_size])

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
        loss_crf = tf.divide(tf.reduce_sum(out), tf.cast(tf.shape(self.x_tokens_fw)[0], dtype=tf.float32))

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

        if self.opt_algo == "adam":
            optimizer = tf.train.AdamOptimizer(self.opt_lr)
        elif self.opt_algo == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.opt_lr)
        else:
            raise Exception("The optimization algorithm you specified does not exist")

        if self.opt_gc_use:
            gvs = optimizer.compute_gradients(-self.loss_crf)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)
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
