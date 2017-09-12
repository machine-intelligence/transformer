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

def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Sources, Targets = load_test_data()
    en2idx, idx2en = load_en_vocab()
    de2idx, idx2de = load_de_vocab()

#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]

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
                        tensor_keys = [None] + list(g.tensors_of_interest.keys())  # Add a null key at the start so it lines up with the tensors_out list
                        tensors_out = sess.run(tensors, {g.x: x, g.y: preds})
                        _preds = tensors_out[0]
                        preds[:, j] = _preds[:, j]

                        print([idx2de[idx] for idx in preds[0]])

                        if j == 0:
                            for sentence_idx in range(len(sources)):
                                print("Source:", sources[sentence_idx])
                                print("Target:", targets[sentence_idx])

                                x_step_size = 100
                                y_step_size = 100
                                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 180, 255), (200, 200, 200), (255, 100, 0), (100, 0, 255)]
                                im = Image.new('RGB', (x_step_size * (hp.maxlen + 1), y_step_size * hp.num_blocks), color=(255, 255, 255, 255))
                                draw = ImageDraw.Draw(im, 'RGBA')
                                # font = ImageFont.truetype("arial.ttf", 15)
                                # draw.font = font

                                for layer in range(hp.num_blocks):
                                    inp = tensors_out[tensor_keys.index('Input%s' % layer)][sentence_idx]
                                    inspect = []
                                    for t in range(10):
                                        target_word = inp[t]

                                        temp_x = []
                                        for word_idx in range(len(idx2en)):
                                            index_vector = [0 for _ in range(0, t)] + [word_idx] + [0 for _ in range(t + 1, 10)]
                                            temp_x.append(index_vector)

                                        results = sess.run(g.embedding, {g.x: np.array(temp_x)})
                                        relevant_projected_words = np.swapaxes(results, 0, 1)[t]
                                        scores = np.dot(relevant_projected_words, target_word)
                                        best_match = idx2en[np.argmax(scores)]
                                        inspect.append(best_match)
                                    print("Layer", layer, "=", " ".join(inspect))

                                    attn_signal_strength = tensors_out[tensor_keys.index('Attention-Signal-Strength%s' % layer)]
                                    residual_signal_strength = tensors_out[tensor_keys.index('Residual-Signal-Strength%s' % layer)]

                                    activation = tensors_out[tensor_keys.index('Activation%s' % layer)]
                                    for q_index in range(hp.maxlen):
                                        for k_index in range(hp.maxlen):
                                            for head_index in range(hp.num_heads):
                                                color = colors[head_index]
                                                strength = activation[sentence_idx + head_index * hp.batch_size][q_index][k_index]
                                                if strength > 0.3:
                                                    draw.line(
                                                        ((q_index + 1) * x_step_size,
                                                         im.size[1] - (layer + 1) * y_step_size,
                                                         (k_index + 1) * x_step_size,
                                                         im.size[1] - layer * y_step_size),
                                                        fill=color + (int(255 * strength),),
                                                        width=3)

                                        residual_variance = residual_signal_strength[sentence_idx][q_index][0] ** 2
                                        attention_variance = attn_signal_strength[sentence_idx][q_index][0] ** 2
                                        # print("Variances:", residual_variance, attention_variance)
                                        strength_ratio = residual_variance / (residual_variance + attention_variance)
                                        scale = strength_ratio * 10.0
                                        # print(strength_ratio)
                                        dot_x = (q_index + 1) * x_step_size
                                        dot_y = im.size[1] - (layer + 1) * y_step_size
                                        draw.ellipse((dot_x - scale, dot_y - scale, dot_x + scale, dot_y + scale), fill=(0, 0, 0, 127))

                                for i, word in enumerate(sources[sentence_idx].split()):
                                    text_size_x, text_size_y = draw.textsize(word)
                                    draw.text((((i + 1) * x_step_size) - text_size_x / 2.0, im.size[1] - text_size_y), text=word, fill=(0, 0, 0, 255))

                                del draw
                                im.save("%s-Activation.png" % sentence_idx, "PNG")

                    # print(g.tensors_of_interest)
                    return

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
