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
from PIL import Image, ImageDraw

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

                    print("Source:", sources[0])
                    print("Target:", targets[0])

                    # Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        tensors = [g.preds] + list(g.tensors_of_interest.values())
                        tensors_out = sess.run(tensors, {g.x: x, g.y: preds})
                        _preds = tensors_out[0]
                        preds[:, j] = _preds[:, j]

                        print([idx2de[idx] for idx in preds[0]])

                        x_step_size = 100
                        y_step_size = 100
                        im = Image.new('RGB', (x_step_size * 11, y_step_size * 6), color=(255, 255, 255, 255))
                        draw = ImageDraw.Draw(im, 'RGBA')

                        tensor_keys = [None] + list(g.tensors_of_interest.keys())  # Add a null key at the start so it lines up with the tensors_out list
                        for layer in range(6):
                            tensor = tensors_out[tensor_keys.index('Activation%s' % layer)]
                            for q_index in range(tensor[0].shape[0]):
                                for k_index in range(tensor[0].shape[1]):
                                    draw.line(
                                        ((q_index + 1) * x_step_size,
                                         im.size[1] - (layer + 1) * y_step_size,
                                         (k_index + 1) * x_step_size,
                                         im.size[1] - layer * y_step_size),
                                        fill=(0, 0, 255, int(255 * tensor[0][q_index][k_index])),
                                        width=3)
                        del draw
                        im.save("Activation.png", "PNG")
                        return

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
