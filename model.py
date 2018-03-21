import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops, rnn_cell
from module import sequence_mask, layer_normalization, get_optimizer


class Model:
    def __init__(self, num_units, num_heads, vocab_size,
                 keep_prob=0.9, num_layers=6, max_len=100,
                 max_graident_norm=5, learning_rate=0.0001,
                 forward_only=False):
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.max_gradient_norm = max_graident_norm
        self.learning_rate = learning_rate
        self.forward_only = forward_only
        # len_inp * batch_size
        self.src_tok = tf.placeholder(dtype=tf.int32, shape=[None, None])
        # len_out * batch_size
        self.tgt_tok = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.len_inp = tf.shape(self.src_tok)[0]
        self.len_out = tf.shape(self.tgt_tok)[0]
        self.batch_size = tf.shape(self.src_tok)[1]
        self.create_model()


    def _get_pos_embedding(self, len_inp, batch_size):
        src_pos = tf.tile(tf.reshape(tf.range(len_inp), [-1, 1]), [batch_size])
        enc_inp_pos = embedding_ops.embedding_lookup(self.emb_pos, src_pos)
        return enc_inp_pos

    def _get_embedding(self, flag_src):
        if flag_src:
            return embedding_ops.embedding_lookup(self.emb, self.src_tok)
        else:
            return embedding_ops.embedding_lookup(self.emb, self.tgt_tok)

    def _multi_head(self, queries, keys, query_mask, key_mask, num_heads, scope='multihead', reuse=None):
        '''
        :param queries: seq_size_q * batch_size * num_units
        :param keys: seq_size_k * batch_size * num_units
        :param values: seq_size_k * batch_size * num_units
        :param query_mask: seq_size_q * batch_size
        :param key_mask: seq_size_k * batch_size
        :return:  seq_size_q * batch_size * num_units
        '''
        with vs.variable_scope(scope=scope, reuse=reuse):
            # Linear Transformation
            Q = rnn_cell._linear(tf.reshape(queries,
                                            [-1, self.num_units]),
                                 self.num_units, True, 1.0, scope='Q')
            Q = tf.reshape(Q, tf.shape(queries))
            K = rnn_cell._linear(tf.reshape(keys,
                                            [-1, self.num_units]),
                                 self.num_units, True, 1.0, scope='K')
            K = tf.reshape(K, tf.shape(keys))
            V = rnn_cell._linear(tf.reshape(keys,
                                            [-1, self.num_units]),
                                 self.num_units, True, 1.0, scope='V')
            V = tf.reshape(V, tf.shape(keys))
            Q_ = tf.pack(tf.split(2, num_heads, Q))  # num_heads * seq_size_q * batch_size * num_units/num_heads
            K_ = tf.pack(tf.split(2, num_heads, K))  # num_heads * seq_size_k * batch_size * num_units/num_heads
            V_ = tf.pack(tf.split(2, num_heads, V))  # num_heads * seq_size_k * batch_size * num_units/num_heads
            len_q = tf.shape(queries)[0]
            # Compute weight
            weights = tf.batch_matmul(tf.transpose(Q_, [0,2,1,3]),
                                      tf.transpose(K_, [0,2,3,1])) \
                      / ((self.num_units/num_heads) ** 0.5)    # num_heads * batch_size * seq_size_q * seq_size_k
            weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=3, keep_dims=True)) # num_heads * batch_size * seq_size_q * seq_size_k
            weights = tf.transpose(weights, [0, 2, 1, 3]) * tf.cast(tf.transpose(key_mask), tf.float32) # num_heads * seq_size_q * batch_size * seq_size_k
            weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=3, keep_dims=True)) # num_heads * seq_size_q * batch_size * seq_size_k
            ctx = tf.transpose(tf.batch_matmul(tf.transpose(weights, [0, 2, 1, 3]),
                                               tf.transpose(V_, [0, 2, 1, 3])),
                               [0, 2, 1, 3])            # num_heads * batch_size * seq_size_q * num_units/num_heads
            ctx *= tf.reshape(query_mask, [len_q, -1, 1]) # num_heads * seq_size_q * batch_size * num_units/num_heads
            ctx = tf.concat(2, tf.unpack(ctx))  # seq_size_q * batch_size * num_units
            ctx = rnn_cell._linear(tf.reshape(ctx, [-1, self.num_units]), self.num_units, True, 1.0, scope='context')
            drop_ctx = tf.nn.dropout(ctx, keep_prob=self.keep_prob)
            # Add and Normalization
            res = layer_normalization(drop_ctx + queries)
        return res

    def _feed_forward(self, inputs, num_units, scope="Feed_Forward", reuse=None):
        '''
        :param inputs: seq_size_q * batch_size * num_units
        :return:  seq_size_q * batch_size * num_units
        '''
        with vs.variable_scope(scope=scope, reuse=reuse):
            W1 = tf.get_variable("W1", [self.num_units, num_units],
                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            b1 = tf.get_variable("b1", [num_units])
            W2 = tf.get_variable("W2", [num_units, self.num_units],
                                 initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            b2 = tf.get_varibale("b2", [self.num_units])
            outputs1 = tf.matmul(tf.reshape(inputs, [-1, self.num_units]), W1) + b1
            outputs2 = tf.matmul(outputs1, W2) + b2
            outputs2 = tf.reshape(outputs2, tf.shape(inputs))
            res = layer_normalization(outputs2 + inputs)
        return res

    def _add_embedding(self):
        self.emb = tf.get_variable("Embedding", [self.vocab_size, self.num_units])
        self.emb_pos = tf.get_variable('Embedding_Pos', [self.max_len, self.num_units])


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
                self.src_mask = tf.sign(tf.abs(self.src_tok))
                self.tgt_mask = tf.sign(tf.abs(self.tgt_tok))
            with vs.variable_scope('Encoder'):
                inp = self.enc_inp
                for i in xrange(self.num_layers):
                    with vs.variable_scope('Encoder_Layer%d' % i):
                        sub1 = self._multi_head(inp, inp,
                                                self.src_mask, self.src_mask, self.num_heads)
                        inp = self._feed_forward(sub1, num_units=4 * self.num_units)
                self.enc_output = inp
            with vs.variable_scope('Decoder'):
                out = self.dec_inp
                for i in xrange(self.num_layers):
                    with vs.variable_scope('Decoder_Layer%d' % i):
                        sub1 = self._multi_head(out, out,
                                                self.tgt_mask, self.tgt_mask, self.num_heads)
                        sub2 = self._multi_head(sub1, self.enc_output,
                                                self.tgt_mask, self.src_mask, self.num_heads)
                        out = self._feed_forward(sub2, num_units=4 * self.num_units)
                self.dec_output = out
            with vs.variable_scope("Logistic"):
                doshape = tf.shape(self.dec_output)
                T, batch_size = doshape[0], doshape[1]
                do2d = tf.reshape(self.dec_output, [-1, self.num_units])
                logits2d = rnn_cell._linear(do2d, self.vocab_size, True, 1.0)
                outputs2d = tf.nn.log_softmax(logits2d)
                self.outputs = tf.reshape(outputs2d, tf.pack([T, batch_size, self.vocab_size]))
                labels1d = tf.reshape(self.tgt_tok, [-1])
                mask1d = tf.reshape(self.tgt_mask, [-1])
                losses1d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2d, labels1d) * tf.to_float(mask1d)
                losses2d = tf.reshape(losses1d, tf.pack([T, batch_size]))
                self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)
            self.global_step = tf.Variable(0, trainable=False)
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

    def train(self, session, source_tokens, target_tokens):
        input_feed = {}
        input_feed[self.src_tok] = source_tokens
        input_feed[self.tgt_tok] = target_tokens
        output_feed = [self.updates, self.gradient_norm,
                       self.losses, self.param_norm, self.tgt_mask]
        outputs = session.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3], outputs[4]

    def test(self, session, source_tokens, target_tokens):
        input_feed = {}
        input_feed[self.src_tok] = source_tokens
        input_feed[self.tgt_tok] = target_tokens
        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]
