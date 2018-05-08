import os
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from egan.dcae import DCAE

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epochs", 20, "Number of epochs to train [20]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term for Adam [0.5]")
flags.DEFINE_integer("num", -1, "Number of images to train/test on [-1]")
flags.DEFINE_integer("batch_size", 64, "Batch size [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("z_dim", 100, "Latent space dimension of the autoencoder [100]")
flags.DEFINE_string("dataset", "celebA", "The name of the dataset [celebA]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of input images file names [*.jpg]")
flags.DEFINE_string("checkpoint_dir", "dcae_checkpoint", "Checkpoint directory name [dcae_checkpoint]")
flags.DEFINE_string("test_dir", "dcae_tests", "Tests directory name [dcae_tests]")
flags.DEFINE_string("data_dir", "data", "Root directory of dataset [data]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for cropping, False if not [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.test_dir):
        os.makedirs(FLAGS.test_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcae = DCAE(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            crop=FLAGS.crop,
            dataset_name=FLAGS.dataset,
            data_dir=FLAGS.data_dir,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            num=FLAGS.num,
            z_dim=FLAGS.z_dim,
            batch_size=FLAGS.batch_size,
            test_num=FLAGS.batch_size,
        )

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        if FLAGS.train:
            dcae.train(FLAGS)
        else:
            if not dcae.load(FLAGS.checkpoint_dir)[0]:
                raise Exception('Train before testing!')

            dcae.test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
