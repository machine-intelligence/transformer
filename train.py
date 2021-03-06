# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os
import codecs
from tqdm import tqdm
from collections import OrderedDict

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            else:  # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            self.tensors_of_interest = OrderedDict()

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

            # Load vocabulary
            en2idx, idx2en = load_en_vocab()
            de2idx, idx2de = load_de_vocab()

            # Encoder
            with tf.variable_scope("encoder"):
                # Embedding
                self.enc, new_tensors = embedding(
                    self.x,
                    vocab_size=len(en2idx),
                    num_units=hp.hidden_units,
                    scale=True,
                    scope="enc_embed")

                self.tensors_of_interest['English-Word-Embedding'] = new_tensors['Embedding']

                # Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(
                        self.x,
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")
                else:
                    positional_embedding, new_tensors = embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                    self.tensors_of_interest['English-Positional-Embedding'] = new_tensors['Embedding']

                    self.enc += positional_embedding

                # Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead Attention
                        self.enc, new_tensors = multihead_attention(
                            queries=self.enc,
                            memory=self.enc,
                            num_units=hp.hidden_units,
                            num_heads=hp.num_heads,
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training,
                            causality=False,
                            num_block=i)
                        self.tensors_of_interest.update(new_tensors)

                        # Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):
                # Embedding
                self.dec, new_tensors = embedding(
                    self.decoder_inputs,
                    vocab_size=len(de2idx),
                    num_units=hp.hidden_units,
                    scale=True,
                    scope="dec_embed")

                self.tensors_of_interest['German-Word-Embedding'] = new_tensors['Embedding']

                # Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(
                        self.decoder_inputs,
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="dec_pe")
                else:
                    positional_embedding, new_tensors = embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="dec_pe")

                    self.tensors_of_interest['German-Positional-Embedding'] = new_tensors['Embedding']

                    self.dec += positional_embedding

                # Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead Attention ( self-attention)
                        self.dec, _ = multihead_attention(queries=self.dec,
                                                          memory=self.dec,
                                                          num_units=hp.hidden_units,
                                                          num_heads=hp.num_heads,
                                                          dropout_rate=hp.dropout_rate,
                                                          is_training=is_training,
                                                          causality=True,
                                                          scope="self_attention")

                        # Multihead Attention ( vanilla attention)
                        self.dec, _ = multihead_attention(queries=self.dec,
                                                          memory=self.enc,
                                                          num_units=hp.hidden_units,
                                                          num_heads=hp.num_heads,
                                                          dropout_rate=hp.dropout_rate,
                                                          is_training=is_training,
                                                          causality=False,
                                                          scope="vanilla_attention")

                        # Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(de2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(de2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

if __name__ == '__main__':
    # Construct graph
    g = Graph(is_training=True)
    print("Graph loaded")

    # Start session
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)

    with sv.managed_session() as sess:

        checkpoint = tf.train.latest_checkpoint(hp.logdir)
        if checkpoint:
            print("Checkpoint found! Restoring...")
            sv.saver.restore(sess, checkpoint)

        for epoch in range(1, hp.num_epochs + 1):
            print("Starting epoch", epoch)
            print()
            if sv.should_stop():
                break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")
