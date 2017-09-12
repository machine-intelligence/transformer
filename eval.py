# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image, ImageDraw, ImageFont

batches_to_visualize = 1

def visualizeEncoderAttention(sources, tensors_of_interest, batch_index: int):
    x_step_size = 100
    y_step_size = 100
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 180, 255), (200, 200, 200), (255, 100, 0), (100, 0, 255))
    assert len(colors) == hp.num_heads
    attention_strength_threshold = 0.3

    for sentence_index, sentence in enumerate(sources):
        print("Source sentence:", sentence)

        im = Image.new('RGB', (x_step_size * (hp.maxlen + 1), y_step_size * hp.num_blocks), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(im, 'RGBA')

        # If you're not on paperspace, you get to have nicer text with the following:
        # noinspection PyBroadException
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            draw.font = font
        except:
            pass

        for layer in range(hp.num_blocks):
            attn_signal_strength = tensors_of_interest['Attention-Signal-Strength%s' % layer]
            residual_signal_strength = tensors_of_interest['Residual-Signal-Strength%s' % layer]
            activation = tensors_of_interest['Activation%s' % layer]

            for head_index in range(hp.num_heads):
                for q_index in range(hp.maxlen):
                    for k_index in range(hp.maxlen):
                        color = colors[head_index]
                        attention_strength = activation[head_index * hp.batch_size + sentence_index][q_index][k_index]
                        if attention_strength > attention_strength_threshold:
                            draw.line(
                                (((q_index + 1) * x_step_size, im.size[1] - (layer + 1) * y_step_size),
                                 ((k_index + 1) * x_step_size, im.size[1] - layer * y_step_size)),
                                fill=color + (int(255 * attention_strength),),
                                width=3)

                    residual_variance = residual_signal_strength[sentence_index][q_index] ** 2
                    attention_variance = attn_signal_strength[sentence_index][q_index] ** 2
                    # print("Variances:", residual_variance, attention_variance)
                    strength_ratio = residual_variance / (residual_variance + attention_variance)
                    scale = strength_ratio * 10.0

                    dot_x = (q_index + 1) * x_step_size
                    dot_y = im.size[1] - (layer + 1) * y_step_size
                    draw.ellipse((dot_x - scale, dot_y - scale, dot_x + scale, dot_y + scale), fill=(0, 0, 0, 127))

        for i, word in enumerate(sentence.split()):
            text_size_x, text_size_y = draw.textsize(word)
            draw.text((((i + 1) * x_step_size) - text_size_x / 2.0, im.size[1] - 20), text=word, fill=(0, 0, 0, 255))

        del draw
        im.save("fig/Activation-Batch-{}-Sentence-{}.png".format(batch_index, sentence_index), "PNG")

def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Sources, Targets = load_test_data()
    en2idx, idx2en = load_en_vocab()
    de2idx, idx2de = load_de_vocab()

    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # Inference
            if not os.path.exists('results'):
                os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):

                    # Get mini-batches
                    x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    # Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        tensors = [g.preds] + list(g.tensors_of_interest.values())
                        tensors_out = sess.run(tensors, {g.x: x, g.y: preds})
                        _preds = tensors_out[0]
                        preds[:, j] = _preds[:, j]

                        print([idx2de[idx] for idx in preds[0]])

                        # For the first few batches, we save figures giving the attention structure in the encoder.
                        if j == 0 and i < batches_to_visualize:
                            tensor_keys = [None] + list(g.tensors_of_interest.keys())  # Add a null key at the start so it lines up with the tensors_out list
                            visualizeEncoderAttention(sources=sources, tensors_of_interest={key: value for key, value in zip(tensor_keys, tensors_out)}, batch_index=i)

                    # Write to file
                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        got = " ".join(idx2de[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source + "\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()

                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)

                # Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100 * score))

if __name__ == '__main__':
    eval()
    print("Done")
