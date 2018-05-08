import glob
import os
import random
import time

import numpy as np
import scipy.misc
import tensorflow as tf

from egan.ops import BatchNorm, linear, conv2d, deconv2d
from egan.util import conv_out_size_same, get_image, save_images, image_manifold_size

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class DCAE(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True, output_height=64, output_width=64,
                 dataset_name=None, data_dir='', input_fname_pattern='*.jpg', checkpoint_dir=None, num=-1,
                 z_dim=100, f_dim=64, test_num=64, batch_size=64):
        """
        num:      Number of points to train on
        test_num: Number of images to display during tests while training
        z_dim:    Dimension of latent space
        f_dim:    Dimension of filters in first conv layer
        """
        self.sess = sess
        self.test_num = test_num

        self.crop = crop
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name is None:
            self.data = []
        else:
            self.data = glob.glob(os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern))
            if num > 0:
                self.data = self.data[:num]

        self.c_dim = 3  # Assume 3 channels per image
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.batch_size = batch_size

        self.e_bn0 = BatchNorm(name='e_bn0')
        self.e_bn1 = BatchNorm(name='e_bn1')
        self.e_bn2 = BatchNorm(name='e_bn2')
        self.e_bn3 = BatchNorm(name='e_bn3')

        self.d_bn0 = BatchNorm(name='d_bn0')
        self.d_bn1 = BatchNorm(name='d_bn1')
        self.d_bn2 = BatchNorm(name='d_bn2')
        self.d_bn3 = BatchNorm(name='d_bn3')

        self.build_model()

    def build_model(self):
        if self.crop:
            input_dims = [self.batch_size, self.output_height, self.output_width, self.c_dim]
        else:
            input_dims = [self.batch_size, self.input_height, self.input_width, self.c_dim]

        self.x = tf.placeholder(tf.float32, shape=input_dims, name='real_images')
        x = self.x

        self.z = self.encoder(x)
        self.x_ = self.decoder(self.z)

        self.tester = self.tester(x)
        self.z_test = self.encoder(x, train=False)

        self.z_sum = histogram_summary('z', self.z)
        self.x__sum = image_summary('x_', self.x_)

        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.x-self.x_))
        self.loss_sum = scalar_summary("loss", self.loss)

        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1
        ).minimize(self.loss, var_list=self.e_vars+self.d_vars)

        tf.global_variables_initializer().run()
        self.sum = merge_summary([self.z_sum, self.x__sum, self.loss_sum])
        self.writer = SummaryWriter(os.path.join(os.getcwd(), 'dcae_logs'), self.sess.graph)

        test_files = self.data[:self.test_num]
        test_x = np.array([get_image(test_file,
                                     input_height=self.input_height,
                                     input_width=self.input_width,
                                     resize_height=self.output_height,
                                     resize_width=self.output_width,
                                     crop=self.crop) for test_file in test_files]).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print('Loaded model weights')
        else:
            print('Could not load model weights')

        for epoch in range(config.epochs):
            batch_idxs = len(self.data) // self.batch_size
            random.shuffle(self.data)

            for idx in range(batch_idxs):
                batch_files = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_x = np.array([get_image(batch_file,
                                              input_height=self.input_height,
                                              input_width=self.input_width,
                                              resize_height=self.output_height,
                                              resize_width=self.output_width,
                                              crop=self.crop) for batch_file in batch_files]).astype(np.float32)

                _, summary_str = self.sess.run([optim, self.sum], feed_dict={self.x: batch_x})
                self.writer.add_summary(summary_str, counter)
                loss = self.loss.eval({self.x: batch_x})

                counter += 1
                print('Epoch: [{:2d}/{:2d}] [{:4d}/{:4d}], time: {:4.4f}, loss: {:.8f}'.format(
                    epoch+1, config.epochs, idx+1, batch_idxs, time.time() - start_time, loss))

                if np.mod(counter, 100) == 1:
                    tests, loss = self.sess.run([self.tester, self.loss], feed_dict={self.x: test_x})
                    save_images(tests, image_manifold_size(tests.shape[0]),
                                os.path.join(config.test_dir, 'train_{:02d}_{:04d}.png'.format(epoch+1, idx+1)))
                    print('[Test] loss: {:.8f}'.format(loss))

                if np.mod(counter, 500) == 2:
                    self.save(self.checkpoint_dir, counter)

        self.save(self.checkpoint_dir, counter)

    def test(self, config):
        batch_idxs = len(self.data) // self.batch_size

        for idx in range(batch_idxs):
            test_files = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
            test_x = np.array([get_image(test_file,
                                         input_height=self.input_height,
                                         input_width=self.input_width,
                                         resize_height=self.output_height,
                                         resize_width=self.output_width,
                                         crop=self.crop) for test_file in test_files]).astype(np.float32)

            tests, zs = self.sess.run([self.tester, self.z_test], feed_dict={self.x: test_x})
            tests = (tests+1.0) / 2.0
            for img, z, file in zip(tests, zs, test_files):
                name = os.path.splitext(os.path.basename(file))[0]
                path = os.path.join(config.test_dir, 'test_{}.png'.format(name))
                path_z = os.path.join(config.test_dir, 'test_{}_z.txt'.format(name))
                scipy.misc.imsave(path, img)
                np.savetxt(path_z, z)

    def encoder(self, image, train=True):
        with tf.variable_scope('encoder') as scope:
            if not train:
                scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            h0 = conv2d(image, self.f_dim, name='e_h0')
            h0 = tf.nn.tanh(self.e_bn0(h0, train=train))

            h1 = conv2d(h0, self.f_dim*2, name='e_h1')
            h1 = tf.nn.relu(self.e_bn1(h1, train=train))

            h2 = conv2d(h1, self.f_dim*4, name='e_h2')
            h2 = tf.nn.relu(self.e_bn2(h2, train=train))

            h3 = conv2d(h2, self.f_dim*8, name='e_h3')
            h3 = tf.nn.relu(self.e_bn3(h3, train=train))

            h4 = linear(tf.reshape(h3, [-1, self.f_dim*8*s_h16*s_w16]), self.z_dim, 'e_h4_lin')

            return tf.nn.tanh(h4)

    def decoder(self, z, train=True):
        with tf.variable_scope('decoder') as scope:
            if not train:
                scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z_ = linear(z, self.f_dim*8*s_h16*s_w16, 'd_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.f_dim*8])
            h0 = tf.nn.relu(self.d_bn0(h0, train=train))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.f_dim*4], name='d_h1')
            h1 = tf.nn.relu(self.d_bn1(h1, train=train))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.f_dim*2], name='d_h2')
            h2 = tf.nn.relu(self.d_bn2(h2, train=train))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.f_dim], name='d_h3')
            h3 = tf.nn.relu(self.d_bn3(h3, train=train))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='d_h4')

            return tf.nn.tanh(h4)

    def tester(self, image):
        z = self.encoder(image, train=False)
        return self.decoder(z, train=False)

    @property
    def model_dir(self):
        return '{}_{}_{}'.format(self.dataset_name, self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = 'dcae.model'
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print('Reading checkpoints...')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print('Reading {} was a success'.format(ckpt_name))
            return True, counter
        else:
            print('Failed to find a checkpoint')
            return False, 0

