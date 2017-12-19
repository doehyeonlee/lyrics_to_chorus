from cnn.no_padding_1conv import NoPadding1Conv
from extract_features import tensor_from_ssm, labels_from_label_array
from util.load_data import load_ssm_string, load_segment_borders

import argparse
import math
import numpy as np
import tensorflow as tf
from random import random
from time import time
from sklearn.utils import shuffle
from os import path
from tensorflow.contrib import slim


def tdiff(timestamp: float) -> float:
    return time() - timestamp


def k(value: int) -> float:
    return float(value) / 1000


def add_to_buckets(buckets, bucket_id, tensor, labels):
    if bucket_id not in buckets:
        buckets[bucket_id] = ([], [])
    X, Y = buckets[bucket_id]
    X.append(tensor)
    Y.append(labels)


def compact_buckets(buckets):
    # Compacting buckets and printing statistics
    largest_bucket = (0, 0)
    for bucket_id in buckets:
        X, Y = buckets[bucket_id]
        buckets[bucket_id] = (np.vstack(X), np.concatenate(Y))
        bucket_len = buckets[bucket_id][1].shape[0]
        print("  max: %3d len: %d" % (2 ** bucket_id, bucket_len))
        if bucket_len > largest_bucket[1]:
            largest_bucket = (bucket_id, bucket_len)

    # Quick fix until I figure out how to process different sized buckets
    largest_bucket_content = buckets[largest_bucket[0]]
    buckets = dict()
    buckets[largest_bucket[0]] = largest_bucket_content
    return buckets

def feed(training_data, batch_size):
    X, Y = training_data
    X, Y = shuffle(X, Y)
    size = Y.shape[0]

    pointer = 0

    while pointer+batch_size < size:
        yield X[pointer:pointer+batch_size], Y[pointer:pointer+batch_size]
        pointer += batch_size
    yield X[pointer:], Y[pointer:]


def main(args):
    print("Starting training with parameters:", vars(args))

    # Load the data
    print("Loading data...")
    timestamp = time()
    ssm_string_data = load_ssm_string(args.data)
    segment_borders = load_segment_borders(args.data)
    print("Done in %.1fs" % tdiff(timestamp))

    # Figure out the maximum ssm size
    print("Gathering dataset statistics...")
    timestamp = time()
    max_ssm_size = 0
    counter = 0
    for ssm_obj in ssm_string_data.itertuples():
        counter += 1
        max_ssm_size = max(max_ssm_size, ssm_obj.ssm.shape[0])
    print("Done in %.1fs (%.2fk items, max ssm size: %d)" % (tdiff(timestamp), k(counter), max_ssm_size))

    # Producing training set
    train_buckets = dict()
    test_buckets = dict()
    print("Producing training set...")
    counter = 0
    filtered = 0
    timestamp = time()
    for borders_obj in segment_borders.itertuples():
        counter += 1
        ssm = ssm_string_data.loc[borders_obj.id].ssm
        ssm_size = ssm.shape[0]

        # Reporting
        if counter % 10000 == 0:
            print("  processed %3.0fk items (%4.1fs, filt.: %4.1fk)" % (k(counter), tdiff(timestamp), k(filtered)))
            timestamp = time()

        # Filter out too small or too large ssm
        if ssm_size < args.min_ssm_size or ssm_size > args.max_ssm_size:
            filtered += 1
            continue

        # Sentences are grouped into buckets to improve performance
        bucket_id = int(math.ceil(math.log2(ssm_size)))
        ssm_tensor = tensor_from_ssm(ssm, 2**bucket_id, args.window_size)
        ssm_labels = labels_from_label_array(borders_obj.borders, ssm_size)

        # 10% goes to test set
        if random() > 0.9:
            add_to_buckets(test_buckets, bucket_id, ssm_tensor, ssm_labels)
        else:
            add_to_buckets(train_buckets, bucket_id, ssm_tensor, ssm_labels)
    del ssm_string_data
    del segment_borders

    # Compacting buckets and printing statistics
    print("Training set buckets:")
    train_buckets = compact_buckets(train_buckets)
    print("Test set buckets:")
    test_buckets = compact_buckets(test_buckets)

    # Define the neural network
    nn = NoPadding1Conv(window_size=args.window_size, ssm_size=2 ** next(train_buckets.keys().__iter__()))

    # Defining optimisation problem
    g_global_step = tf.train.get_or_create_global_step()
    g_train_op = slim.optimize_loss(
        loss=nn.g_loss, global_step=g_global_step, learning_rate=None,
        optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)

    # Logging
    summary_writer = tf.summary.FileWriter(
        logdir=path.join(args.output, 'log'), graph=tf.get_default_graph())
    g_summary = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        # Checkpoint restore / variable initialising
        save_path = path.join(args.output, 'model.ckpt')
        latest_checkpoint = tf.train.latest_checkpoint(args.output)
        if latest_checkpoint is None:
            print("Initializing variables")
            timestamp = time()
            tf.get_variable_scope().set_initializer(tf.random_normal_initializer(mean=0.0, stddev=0.01))
            tf.global_variables_initializer().run()
            print("Done in %.2fs" % tdiff(timestamp))
        else:
            print("Restoring from checkpoint variables")
            timestamp = time()
            saver.restore(sess=sess, save_path=latest_checkpoint)
            print("Done in %.2fs" % tdiff(timestamp))

        print()
        timestamp = time()
        global_step_v = 0
        avg_loss = 0.0

        # Training loop
        for epoch in range(args.max_epoch):
            for bucket_id in train_buckets:
                for batch_X, batch_Y in feed(train_buckets[bucket_id], args.batch_size):
                    # Single training step
                    summary_v, global_step_v, loss_v, _ = sess.run(
                        fetches=[g_summary, g_global_step, nn.g_loss, g_train_op],
                        feed_dict={nn.g_in: batch_X, nn.g_labels: batch_Y, nn.g_dprob: 0.8})
                    summary_writer.add_summary(summary=summary_v, global_step=global_step_v)
                    avg_loss += loss_v

                    # Reporting
                    if global_step_v % args.report_period == 0:
                        print("Iter %d" % global_step_v)
                        print("  epoch %.0f, avg.loss %.2f, time per iter %.2fs" % (
                            epoch, avg_loss / args.report_period, tdiff(timestamp) / args.report_period
                        ))
                        timestamp = time()
                        avg_loss = 0.0

                        # Evaluation
                        micro_tp = 0
                        total = 0
                        for bucket_id in train_buckets:
                            for test_X, test_Y in feed(test_buckets[bucket_id], args.batch_size):
                                total += test_X.shape[0]
                                micro_tp += test_X.shape[0] - nn.g_incorrect.eval(feed_dict={nn.g_in: test_X, nn.g_labels: test_Y, nn.g_dprob: 1.0})
                        print("  precision: %.2f%%" % ((micro_tp / total) * 100))

                    # Checkpointing
                    if global_step_v % 1000 == 0:
                        real_save_path = saver.save(sess=sess, save_path=save_path, global_step=global_step_v)
                        print("Saved the checkpoint to: %s" % real_save_path)

        real_save_path = saver.save(sess=sess, save_path=save_path, global_step=global_step_v)
        print("Saved the checkpoint to: %s" % real_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the lyrics segmentation cnn')
    parser.add_argument('--data', required=True,
                        help='The directory with the data')
    parser.add_argument('--output', required=True,
                        help='Output path')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--window-size', type=int, default=2,
                        help='The size of the window')
    parser.add_argument('--min-ssm-size', type=int, default=5,
                        help='Minimum size of the ssm matrix')
    parser.add_argument('--max-ssm-size', type=int, default=128,
                        help='Maximum size of the ssm matrix')
    parser.add_argument('--report-period', type=int, default=1000,
                        help='When to report stats')

    args = parser.parse_args()
    main(args)
