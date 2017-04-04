# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import multi_task_model

import subprocess
import stat


#tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
#tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "data/fake_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "music_model_tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 10000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 12,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", 'joint', "Options: joint; intent; tagging")
tf.app.flags.DEFINE_boolean("eval", False, "Evaluation mode")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
  print ('Please indicate max sequence length. Exit')
  exit()

if FLAGS.task is None:
  print ('Please indicate task to run. Available options: intent; tagging; joint')
  exit()

task = dict({'intent':0, 'tagging':0, 'joint':0})
if FLAGS.task == 'intent':
  task['intent'] = 1
elif FLAGS.task == 'tagging':
  task['tagging'] = 1
elif FLAGS.task == 'joint':
  task['intent'] = 1
  task['tagging'] = 1
  task['joint'] = 1

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]
#_buckets = [(3, 10), (10, 25)]

def create_model(session, source_vocab_size, target_vocab_size, label_vocab_size):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = multi_task_model.MultiTaskModel(
          source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
          forward_only=False,
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)
  with tf.variable_scope("model", reuse=True):
    model_test = multi_task_model.MultiTaskModel(
          source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
          forward_only=True,
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  print(ckpt)
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model_train, model_test

def eval():
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].iteritems():
    print ('%s: %s' % (k, str(v)))
  print("Preparing data in %s" % FLAGS.data_dir)
  vocab_path = ''
  tag_vocab_path = ''
  label_vocab_path = ''
  in_seq_train, out_seq_train, label_train, in_seq_dev, out_seq_dev, label_dev, in_seq_test, out_seq_test, label_test, vocab_path, tag_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)

  with tf.Session() as sess:
    model, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab))

    data_set = [[[]]]
    token_ids = data_utils.prepare_one_data(FLAGS.data_dir, FLAGS.in_vocab_size, 'Who’s the original singer of Blessings')
    slot_ids = [0 for i in range(len(token_ids))]
    data_set[0][0].append(token_ids)
    data_set[0][0].append(slot_ids)
    data_set[0][0].append([0])
    encoder_inputs, tags, tag_weights, sequence_length, labels = model_test.get_one(
                  data_set, 0, 0)
    if task['joint'] == 1:
      _, _, tagging_logits, classification_logits = model.joint_step(sess, encoder_inputs, tags, tag_weights, labels,
                                                                     sequence_length, 0, False)

    classification = [np.argmax(classification_logit) for classification_logit in classification_logits]
    tagging_logit = [np.argmax(tagging_logit) for tagging_logit in tagging_logits]

    out_vocab_path = os.path.join(FLAGS.data_dir, "out_vocab_%d.txt" % FLAGS.out_vocab_size)
    label_path = os.path.join(FLAGS.data_dir, "label.txt")
    out_vocab, _ = data_utils.initialize_vocabulary(out_vocab_path)
    label_vocab, _ = data_utils.initialize_vocabulary(label_path)
    def inverse_lookup(vocab, id):
      for key, value in vocab.iteritems():
        if value == id:
          return key
    classification_word = [inverse_lookup(label_vocab, c) for c in classification]
    tagging_word = [inverse_lookup(out_vocab, t) for t in tagging_logit]
    print("Who’s the original singer of Blessings")
    print(classification_word)
    print(tagging_word)

def main(_):
  eval()

if __name__ == "__main__":
  tf.app.run()


