import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops, rnn_cell
from module import sequence_mask, layer_normalization, get_optimizer


class Model:
    def __init__(self, num_units, num_heads, vocab_size,
                 keep_prob=0.9, num_layers=6, max_len=100,
                 max_graident_norm=5, learning_rate=0.0001,
                 learning_rate_decay_factor=0.95,
                 forward_only=False):
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.max_gradient_norm = max_graident_norm
        self.forward_only = forward_only

        # len_inp * batch_size
        self.src_tok = tf.placeholder(dtype=tf.int32, shape=[None, None])
        # len_out * batch_size
        self.tgt_tok = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.src_mask = tf.placeholder(dtype=tf.bool, shape=[None, None])
        self.tgt_mask = tf.placeholder(dtype=tf.bool, shape=[None, None])

        self.len_inp = tf.shape(self.src_tok)[1]
        self.len_out = tf.shape(self.tgt_tok)[1]
        self.batch_size = tf.shape(self.src_tok)[0]

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.create_model()

    def _get_pos_embedding(self, len_inp, batch_size):
        src_pos = tf.tile(tf.reshape(tf.range(len_inp), [1, -1]), [batch_size, 1])
        enc_inp_pos = embedding_ops.embedding_lookup(self.emb_pos, src_pos)
        return enc_inp_pos

    def _get_embedding(self, flag_src):
        if flag_src:
            return embedding_ops.embedding_lookup(self.enc_emb, self.src_tok) * (self.num_units ** 0.5)
        else:
            return embedding_ops.embedding_lookup(self.dec_emb, self.tgt_tok) * (self.num_units ** 0.5)

    def _multi_head(self, queries, keys, query_mask, key_mask, num_heads, block_feature=False, scope='multihead', reuse=None):
        with vs.variable_scope(scope, reuse=reuse):
            # batch_size * seq_size_q * num_units
            Q = rnn_cell._linear(tf.reshape(queries,
                                            [-1, self.num_units]),
                                 self.num_units, True, 1.0, scope='Q')
            Q = tf.reshape(Q, tf.shape(queries))
            # batch_size * seq_size_k * num_units
            K = rnn_cell._linear(tf.reshape(keys,
                                            [-1, self.num_units]),
                                 self.num_units, True, 1.0, scope='K')
            K = tf.reshape(K, tf.shape(keys))
            V = rnn_cell._linear(tf.reshape(keys,
                                            [-1, self.num_units]),
                                 self.num_units, True, 1.0, scope='V')
            V = tf.reshape(V, tf.shape(keys))
            Q_ = tf.pack(tf.split(2, num_heads, Q))  # num_heads *  batch_size * seq_size_q *num_units/num_heads
            K_ = tf.pack(tf.split(2, num_heads, K))  # num_heads * batch_size * seq_size_k * num_units/num_heads
            V_ = tf.pack(tf.split(2, num_heads, V))  # num_heads * batch_size * seq_size_k * num_units/num_heads
            len_q = tf.shape(queries)[1]
            len_k = tf.shape(keys)[1]

            # Compute weight
            weights = tf.batch_matmul(Q_, tf.transpose(K_, [0,1,3,2])) \
                      / ((self.num_units/num_heads) ** 0.5)    # num_heads * batch_size * seq_size_q * seq_size_k
            key_mask = tf.tile(tf.reshape(key_mask, [1, -1, 1, len_k]), [num_heads, 1, len_q, 1])
            weights = tf.select(key_mask, weights, tf.ones_like(weights) * (-2**32 + 1))

            if block_feature:
                diag_vals = tf.ones_like(weights[0, 0, :, :]) # seq_size_q * seq_size_k
                mask = tf.cast(tf.batch_matrix_band_part(diag_vals, -1, 0), tf.bool)
                mask = tf.tile(tf.reshape(mask, [1, 1, len_q, len_k]), [num_heads, tf.shape(queries)[0], 1, 1])
                weights = tf.select(mask, weights, tf.ones_like(weights) * (-2 ** 32 + 1))

            weights = tf.reshape(tf.nn.softmax(tf.reshape(weights, [-1, len_k])),
                                 [num_heads, -1, len_q, len_k])
            # num_heads * batch_size * seq_size_q * num_units/num_heads
            ctx = tf.batch_matmul(weights,  V_)

            ctx *= tf.reshape(tf.cast(query_mask, tf.float32), [-1, len_q, 1]) # num_heads * batch_size * seq_size_q * num_units/num_heads
            ctx = tf.concat(2, tf.unpack(ctx))  # batch_size * seq_size_q * num_units
            ctx = rnn_cell._linear(tf.reshape(ctx, [-1, self.num_units]), self.num_units, True, 1.0, scope='context')
            ctx = tf.reshape(ctx, [-1, len_q, self.num_units])
            drop_ctx = tf.nn.dropout(ctx, keep_prob=self.keep_prob)
            # Add and Normalization
            res = layer_normalization(drop_ctx + queries)
        return  res, weights

    def _feed_forward(self, inputs, num_units, scope="Feed_Forward", reuse=None):
        '''
        :param inputs: batch_size * seq_size_q * num_units
        :return:  batch_size * seq_size_q *  num_units
        '''
        with vs.variable_scope(scope, reuse=reuse):
            W1 = tf.get_variable("W1", [self.num_units, num_units],
                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            b1 = tf.get_variable("b1", [num_units])
            W2 = tf.get_variable("W2", [num_units, self.num_units],
                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            b2 = tf.get_variable("b2", [self.num_units])
            outputs1 = tf.nn.relu(tf.matmul(tf.reshape(inputs, [-1, self.num_units]), W1) + b1)
            outputs2 = tf.matmul(outputs1, W2) + b2
            outputs2 = tf.reshape(outputs2, tf.shape(inputs))
            #res = outputs2
            res = layer_normalization(outputs2 + inputs)
        return res

    def _add_embedding(self):
        self.enc_emb = tf.get_variable("Enc_Embedding", [self.vocab_size, self.num_units])
        self.dec_emb = tf.get_variable("Dec_Embedding", [self.vocab_size, self.num_units])

        position_enc = np.array([[pos / np.power(10000, 2. * i / self.num_units)
                                  for i in range(self.num_units)]
                                 for pos in range(200)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        self.emb_pos = tf.convert_to_tensor(position_enc, dtype=tf.float32)

    def create_model(self):
        with vs.variable_scope('Model'):
            with vs.variable_scope('Input'):
                self._add_embedding()
                self.enc_pos = self._get_pos_embedding(self.len_inp, self.batch_size)
                self.dec_pos = self._get_pos_embedding(self.len_out, self.batch_size)
                self.enc_emb = self._get_embedding(1)
                self.dec_emb = self._get_embedding(0)
                self.enc_inp = self.enc_emb + self.enc_pos
                self.dec_inp = self.dec_emb + self.dec_pos
                self.enc_inp = tf.nn.dropout(self.enc_inp, keep_prob=self.keep_prob)
                self.dec_inp = tf.nn.dropout(self.dec_inp, keep_prob=self.keep_prob)
            with vs.variable_scope('Encoder'):
                inp = self.enc_inp
                for i in xrange(self.num_layers):
                    with vs.variable_scope('Encoder_Layer%d' % i):
                        sub1, self.enc_w = self._multi_head(inp, inp,
                                                self.src_mask, self.src_mask, self.num_heads)
                        inp = self._feed_forward(sub1, num_units=4 * self.num_units)
                self.enc_output = inp
            with vs.variable_scope('Decoder'):
                out = self.dec_inp
                for i in xrange(self.num_layers):
                    with vs.variable_scope('Decoder_Layer%d' % i):
                        sub1, self.dec_w = self._multi_head(out, out,
                                                self.tgt_mask, self.tgt_mask,
                                                self.num_heads,
                                                block_feature=True,
                                                scope='self_attention')
                        sub2, self.dec_w_2 = self._multi_head(sub1, self.enc_output,
                                                self.tgt_mask, self.src_mask,
                                                self.num_heads,
                                                scope='vanilla_attention')
                        out = self._feed_forward(sub2, num_units=4 * self.num_units)
                self.dec_output = out

            with vs.variable_scope("Logistic"):
                doshape = tf.shape(self.dec_output)
                batch_size, T = doshape[0], doshape[1]
                do2d = tf.reshape(self.dec_output, [-1, self.num_units])
                logits = rnn_cell._linear(do2d, self.vocab_size, True, 1.0)
                self.outputs = tf.nn.softmax(logits)
                targets_no_GO = tf.slice(self.tgt_tok, [0, 1], [-1, -1])
                masks_no_GO = tf.slice(self.tgt_mask, [0, 1], [-1, -1])
                # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
                labels = tf.reshape(tf.pad(targets_no_GO,[[0,0],[0,1]]), [-1])
                labels = tf.one_hot(labels, depth=self.vocab_size)
                labels = tf.reshape(0.9 * labels + 0.1 / self.vocab_size, [batch_size * T, -1])
                self.labels = labels
                mask = tf.reshape(tf.pad(masks_no_GO, [[0, 0], [0, 1]]), [-1])
                losses = tf.nn.softmax_cross_entropy_with_logits(logits, labels) * tf.cast(mask, tf.float32)
                losses2d = tf.reshape(losses, tf.pack([batch_size, T]))
                self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)
            params = tf.trainable_variables()
            if not self.forward_only:
                opt = get_optimizer('adam')(self.learning_rate)
                gradients = tf.gradients(self.losses, params)
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.max_gradient_norm)
                self.gradient_norm = tf.global_norm(gradients)
                self.param_norm = tf.global_norm(params)
                self.updates = opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def train(self, session, source_tokens, target_tokens, source_mask, target_mask):
        input_feed = {}
        input_feed[self.src_tok] = np.transpose(source_tokens)
        input_feed[self.tgt_tok] = np.transpose(target_tokens)
        input_feed[self.src_mask] = np.transpose(source_mask)
        input_feed[self.tgt_mask] = np.transpose(target_mask)
        output_feed = [self.updates, self.gradient_norm,
                       self.losses, self.param_norm]
        outputs = session.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3]

    def test(self, session, source_tokens, target_tokens, source_mask, target_mask):
        input_feed = {}
        input_feed[self.src_tok] = np.transpose(source_tokens)
        input_feed[self.tgt_tok] = np.transpose(target_tokens)
        input_feed[self.src_mask] = np.transpose(source_mask)
        input_feed[self.tgt_mask] = np.transpose(target_mask)
        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]
