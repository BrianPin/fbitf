import os
import sys
import numpy as np
import tensorflow as tf
import math
import time
import argparse
import random
import data_set
FLAGS = None

train_data = data_set.FizzBuzzDataSet(10000)
validate_data = data_set.FizzBuzzDataSet(1000)
test_data = data_set.FizzBuzzDataSet(1000)


class FizzBuzz(object):
    BIT_WIDTH = data_set.BIT_WIDTH
    NUM_CLASS = data_set.NUM_CLASS

    def __init__(self):
        pass

    @staticmethod
    def inference(num_as_bits, hidden1_units, hidden2_units):
        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal([FizzBuzz.BIT_WIDTH,
                                                       hidden1_units],
                                                      stddev=1.0/math.sqrt(float(FizzBuzz.BIT_WIDTH))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(num_as_bits, weights) + biases)

        with tf.name_scope('hidden2'):
            weights = tf.Variable(tf.truncated_normal([hidden1_units,
                                                       hidden2_units],
                                                      stddev=1.0/math.sqrt(float(hidden1_units))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(tf.truncated_normal([hidden2_units, FizzBuzz.NUM_CLASS],
                                                      stddev=1.0/math.sqrt(hidden2_units)),
                                  name='weights')
            biases = tf.Variable(tf.zeros([FizzBuzz.NUM_CLASS]), name='biases')
            logits = tf.matmul(hidden2, weights) + biases

        return logits

    @staticmethod
    def training(loss, learning_rate):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    @staticmethod
    def loss(logits, labels):
        """Calculates the loss from the logits and the labels.
        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size].

        Returns:
            loss: Loss tensor of type float.
        """
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    @staticmethod
    def evaluation(logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs(batch_size):
    number_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FizzBuzz.BIT_WIDTH),
                                        name='number')
    label_placeholder = tf.placeholder(tf.int32, shape=(batch_size), 
                                       name='label')
    return number_placeholder, label_placeholder


def fill_feed_dict(data_sets, number_placeholder, label_placeholder):
    x, y = data_sets.next_batch(FLAGS.batch_size)
    feed_dict = {
        number_placeholder: x,
        label_placeholder: y,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            number_placeholder,
            labels_placeholder,
            data_sets):
    true_count = 0
    steps_per_epoch = data_sets.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_sets,
                                   number_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d Num correct: %d Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    with tf.Graph().as_default():
        number_placeholder, label_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = FizzBuzz.inference(number_placeholder,
                                    FLAGS.hidden1,
                                    FLAGS.hidden2)
        loss = FizzBuzz.loss(logits, label_placeholder)

        train_op = FizzBuzz.training(loss, FLAGS.learning_rate)
        eval_correct = FizzBuzz.evaluation(logits, label_placeholder)
        summary = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init_op)
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(train_data, 
                                       number_placeholder, 
                                       label_placeholder)

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 10 == 0:
                print('Step %d loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        number_placeholder,
                        label_placeholder,
                        train_data)
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        number_placeholder,
                        label_placeholder,
                        validate_data)
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        number_placeholder,
                        label_placeholder,
                        test_data)


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='log',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
