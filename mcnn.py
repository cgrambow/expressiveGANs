import glob
import os
import random
import time

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Activation, Dropout
import numpy as np

from util import get_image


class MCNN(object):
    def __init__(self, input_height=108, input_width=108, crop=True, output_height=64, output_width=64,
                 train_dataset_name='celebA', test_dataset_name=None, data_dir='', input_fname_pattern='*.jpg',
                 attribute_file_name='list_attr_celeba.txt', model_path='weights.h5', aux_model_path='weights_aux.h5',
                 train_size=-1, test_size=-1):
        self.crop = crop

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.data_dir = data_dir
        self.input_fname_pattern = input_fname_pattern
        self.model_path = model_path
        self.aux_model_path = aux_model_path

        self.data = glob.glob(os.path.join(self.data_dir, self.train_dataset_name, self.input_fname_pattern))
        random.shuffle(self.data)
        if train_size > 0:
            self.data = self.data[:train_size]

        if self.test_dataset_name is None:
            self.test_data = []
        else:
            self.test_data = glob.glob(os.path.join(self.data_dir, self.test_dataset_name, self.input_fname_pattern))
            random.shuffle(self.test_data)
            if test_size > 0:
                self.test_data = self.test_data[:test_size]
        self.c_dim = 3  # Assume 3 channels per image

        self.attribute_dict = {}
        self.load_attributes(os.path.join(self.data_dir, attribute_file_name))

        self.xt = None
        self.yt = None

        self.model = None
        self.aux_model = None
        self.build_model()
        self.build_aux()

    def build_model(self):
        if self.crop:
            image_dims = (self.output_height, self.output_width, self.c_dim)
        else:
            image_dims = (self.input_height, self.input_width, self.c_dim)

        inputs = Input(shape=image_dims)

        # Shared layers
        x = Conv2D(75, (7, 7))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3))(x)

        x = Conv2D(200, (5, 5))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3))(x)

        # Layers for gender group
        gender = Conv2D(300, (3, 3), padding='same')(x)
        gender = BatchNormalization()(gender)
        gender = Activation('relu')(gender)
        gender = MaxPooling2D((5, 5))(gender)
        gender = Flatten()(gender)
        gender = Dense(512, activation='relu')(gender)
        gender = Dropout(0.5)(gender)
        gender = Dense(512, activation='relu')(gender)
        gender = Dropout(0.5)(gender)
        # male
        gender = Dense(1, activation='sigmoid')(gender)

        # Layers for nose group
        nose = Conv2D(300, (3, 3), padding='same')(x)
        nose = BatchNormalization()(nose)
        nose = Activation('relu')(nose)
        nose = MaxPooling2D((5, 5))(nose)
        nose = Flatten()(nose)
        nose = Dense(512, activation='relu')(nose)
        nose = Dropout(0.5)(nose)
        nose = Dense(512, activation='relu')(nose)
        nose = Dropout(0.5)(nose)
        # big nose, pointy nose
        nose = Dense(2, activation='sigmoid')(nose)

        # Layers for mouth group
        mouth = Conv2D(300, (3, 3), padding='same')(x)
        mouth = BatchNormalization()(mouth)
        mouth = Activation('relu')(mouth)
        mouth = MaxPooling2D((5, 5))(mouth)
        mouth = Flatten()(mouth)
        mouth = Dense(512, activation='relu')(mouth)
        mouth = Dropout(0.5)(mouth)
        mouth = Dense(512, activation='relu')(mouth)
        mouth = Dropout(0.5)(mouth)
        # big lips, smiling, lipstick, mouth slightly open
        mouth = Dense(4, activation='sigmoid')(mouth)

        # Layers for eyes group
        eyes = Conv2D(300, (3, 3), padding='same')(x)
        eyes = BatchNormalization()(eyes)
        eyes = Activation('relu')(eyes)
        eyes = MaxPooling2D((5, 5))(eyes)
        eyes = Flatten()(eyes)
        eyes = Dense(512, activation='relu')(eyes)
        eyes = Dropout(0.5)(eyes)
        eyes = Dense(512, activation='relu')(eyes)
        eyes = Dropout(0.5)(eyes)
        # arched eyebrows, bags under eyes, bushy eyebrows, narrow eyes, eyeglasses
        eyes = Dense(5, activation='sigmoid')(eyes)

        # Layers for face group
        face = Conv2D(300, (3, 3), padding='same')(x)
        face = BatchNormalization()(face)
        face = Activation('relu')(face)
        face = MaxPooling2D((5, 5))(face)
        face = Flatten()(face)
        face = Dense(512, activation='relu')(face)
        face = Dropout(0.5)(face)
        face = Dense(512, activation='relu')(face)
        face = Dropout(0.5)(face)
        # attractive, blurry, oval face, pale skin, young, heavy makeup
        face = Dense(6, activation='sigmoid')(face)

        # Layers for rest
        rest = Conv2D(300, (3, 3), padding='same')(x)
        rest = BatchNormalization()(rest)
        rest = Activation('relu')(rest)
        rest = MaxPooling2D((5, 5))(rest)
        rest = Flatten()(rest)

        # Layers for around head group
        aroundhead = Dense(512, activation='relu')(rest)
        aroundhead = Dropout(0.5)(aroundhead)
        aroundhead = Dense(512, activation='relu')(aroundhead)
        aroundhead = Dropout(0.5)(aroundhead)
        # black hair, blond hair, brown hair, gray hair, balding, receding hairline, bangs, straight hair, wavy hair
        aroundhead = Dense(9, activation='sigmoid')(aroundhead)

        # Layers for facial hair group
        facialhair = Dense(512, activation='relu')(rest)
        facialhair = Dropout(0.5)(facialhair)
        facialhair = Dense(512, activation='relu')(facialhair)
        facialhair = Dropout(0.5)(facialhair)
        # 5 o'clock shadow, mustache, no beard, sideburns, goatee
        facialhair = Dense(5, activation='sigmoid')(facialhair)

        # Layers for cheeks group
        cheeks = Dense(512, activation='relu')(rest)
        cheeks = Dropout(0.5)(cheeks)
        cheeks = Dense(512, activation='relu')(cheeks)
        cheeks = Dropout(0.5)(cheeks)
        # high cheekbones, rosy cheeks
        cheeks = Dense(2, activation='sigmoid')(cheeks)

        # Layers for fat group
        fat = Dense(512, activation='relu')(rest)
        fat = Dropout(0.5)(fat)
        fat = Dense(512, activation='relu')(fat)
        fat = Dropout(0.5)(fat)
        # chubby
        fat = Dense(1, activation='sigmoid')(fat)

        outputs = keras.layers.concatenate([
            gender,
            nose,
            mouth,
            eyes,
            face,
            aroundhead,
            facialhair,
            cheeks,
            fat
        ])

        self.model = Model(inputs=inputs,
                           outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def build_aux(self):
        inputs = Input(shape=(35,))
        outputs = Dense(35, activation='sigmoid', use_bias=False)(inputs)

        self.aux_model = Model(inputs=inputs,
                               outputs=outputs)
        self.aux_model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, epochs=22, batch_size=100, aux=False):
        if aux:
            print(self.aux_model.summary())
        else:
            print(self.model.summary())

        if self.test_data:
            self.xt = np.array([get_image(file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=self.crop) for file in self.test_data]).astype(np.float32)
            yt_names = [os.path.basename(file) for file in self.test_data]
            yt_tuples = [self.attribute_dict[name] for name in yt_names]
            self.yt = [np.empty((len(self.test_data),n)).astype(np.float32) for n in (1, 2, 4, 5, 6, 9, 5, 2, 1)]
            for i, attribute_groups in enumerate(yt_tuples):
                for g, group in enumerate(attribute_groups):
                    self.yt[g][i] = group
            self.yt = np.concatenate(self.yt, axis=1)

        counter = 1
        start_time = time.time()
        could_load = self.load()
        if could_load:
            if could_load == 2:
                print('Resuming training from loaded model')
            else:
                print('Resuming training from loaded base model (no aux)')
        else:
            print('Training new model')

        for epoch in range(epochs):
            batch_idxs = len(self.data) // batch_size
            random.shuffle(self.data)

            for idx in range(batch_idxs):
                batch_files = self.data[idx*batch_size:(idx+1)*batch_size]
                batch_x = [get_image(batch_file,
                                     input_height=self.input_height,
                                     input_width=self.input_width,
                                     resize_height=self.output_height,
                                     resize_width=self.output_width,
                                     crop=self.crop) for batch_file in batch_files]
                x = np.array(batch_x).astype(np.float32)

                batch_names = [os.path.basename(batch_file) for batch_file in batch_files]
                batch_y = [self.attribute_dict[name] for name in batch_names]
                # This has to match the number of attributes in each group (see load_attributes):
                y = [np.empty((batch_size,n)).astype(np.float32) for n in (1, 2, 4, 5, 6, 9, 5, 2, 1)]
                for i, attribute_groups in enumerate(batch_y):
                    for g, group in enumerate(attribute_groups):
                        y[g][i] = group
                y = np.concatenate(y, axis=1)

                if aux:
                    scores = self.model.predict_on_batch(x)
                    train_loss = self.aux_model.train_on_batch(scores, y)
                else:
                    train_loss = self.model.train_on_batch(x, y)

                counter += 1
                print('Epoch: [{:2d}/{:2d}] [{:4d}/{:4d}], time: {:4.4f}, loss: {:.8f}'.format(
                    epoch+1, epochs, idx+1, batch_idxs, time.time() - start_time, train_loss))

                if np.mod(counter, 500) == 2:
                    self.save(aux=aux)

            if self.xt is not None:
                accuracy = self.evaluate(self.xt, self.yt, batch_size=batch_size, aux=aux)
                print('Test accuracy: {:.2f}'.format(accuracy*100.0))

    def predict(self, xt, batch_size=None, aux=False):
        ytp = self.model.predict(xt, batch_size=batch_size)
        if aux:
            ytp = self.aux_model.predict(ytp, batch_size=batch_size)
        ytp[ytp>=0.5] = 1
        ytp[ytp<0.5] = 0
        return ytp

    def evaluate(self, xt, yt, batch_size=None, aux=False):
        ytp = self.predict(xt, batch_size=batch_size, aux=aux)
        return 1.0 - np.sum(np.abs(yt - ytp)) / yt.size

    def save(self, aux=False):
        self.model.save_weights(self.model_path)
        if aux:
            self.aux_model.save_weights(self.aux_model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            if os.path.exists(self.aux_model_path):
                self.aux_model.load_weights(self.aux_model_path)
                return 2
            else:
                return True
        else:
            return False

    def load_attributes(self, path):
        """
        Groups are:
        0: (male)
        1: (big_nose, pointy_nose)
        2: (big_lips, smiling, wearing_lipstick, mouth_slightly_open)
        3: (arched_eyebrows, bags_under_eyes, bushy_eyebrows, narrow_eyes, eyeglasses)
        4: (attractive, blurry, oval_face, pale_skin, young, heavy_makeup)
        5: (black_hair, blond_hair, brown_hair, gray_hair, bald, receding_hairline, bangs, straight_hair, wavy_hair)
        6: (5_o_clock_shadow, mustache, no_beard, sideburns, goatee)
        7: (high_cheekbones, rosy_cheeks)
        8: (chubby)
        """
        with open(path) as f:
            train_names = set(os.path.basename(d) for d in self.data)
            test_names = set(os.path.basename(d) for d in self.test_data)

            f.readline()
            # Get attributes and their column indices
            a = {label: idx for idx, label in enumerate(f.readline().strip().lower().split())}

            for line in f:
                line_split = line.strip().split()
                img_name = line_split[0]

                # Only save the data we are training or testing on
                if img_name in train_names or img_name in test_names:
                    vals = [1 if v == '1' else 0 for v in line_split[1:]]
                    self.attribute_dict[img_name] = (
                        np.array(
                            [vals[a['male']]]
                        ),
                        np.array(
                            [vals[a['big_nose']],
                             vals[a['pointy_nose']]]
                        ),
                        np.array(
                            [vals[a['big_lips']],
                             vals[a['smiling']],
                             vals[a['wearing_lipstick']],
                             vals[a['mouth_slightly_open']]]
                        ),
                        np.array(
                            [vals[a['arched_eyebrows']],
                             vals[a['bags_under_eyes']],
                             vals[a['bushy_eyebrows']],
                             vals[a['narrow_eyes']],
                             vals[a['eyeglasses']]]
                        ),
                        np.array(
                            [vals[a['attractive']],
                             vals[a['blurry']],
                             vals[a['oval_face']],
                             vals[a['pale_skin']],
                             vals[a['young']],
                             vals[a['heavy_makeup']]]
                        ),
                        np.array(
                            [vals[a['black_hair']],
                             vals[a['blond_hair']],
                             vals[a['brown_hair']],
                             vals[a['gray_hair']],
                             vals[a['bald']],
                             vals[a['receding_hairline']],
                             vals[a['bangs']],
                             vals[a['straight_hair']],
                             vals[a['wavy_hair']]]
                        ),
                        np.array(
                            [vals[a['5_o_clock_shadow']],
                             vals[a['mustache']],
                             vals[a['no_beard']],
                             vals[a['sideburns']],
                             vals[a['goatee']]]
                        ),
                        np.array(
                            [vals[a['high_cheekbones']],
                             vals[a['rosy_cheeks']]]
                        ),
                        np.array(
                            [vals[a['chubby']]]
                        )
                    )
