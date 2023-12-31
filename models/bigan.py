# Bidirectional GAN

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Dropout, concatenate
from keras.models import Sequential, Model
from keras.optimizers.legacy import Adam
from keras.utils import image_dataset_from_directory

import tensorflow as tf

class BiGAN():
    def __init__(self) -> None:
        # input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1 # gray scale images
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.batch_size = 32
        self.latent_dim = 100

        optimizer = Adam(2e-4, 0.5)

        self.generator_losses = []
        self.discriminator_losses = []
        self.discriminator_acc = []

        # building and compiling the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # building generator
        self.generator = self.build_generator()

        # building encoder
        self.encoder = self.build_encoder()

        self.discriminator.trainable = False

        # generate images from sampled noise
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # encode image
        img1 = Input(shape=self.img_shape)
        z = self.encoder(img1)

        # latent -> img is fake, img -> latent is valid
        fake = self.discriminator([noise, img])
        valid = self.discriminator([z, img1])

        # combined model generator + discriminator (cgan), trains to fool the discriminator
        self.combined = Model([noise, img1], [fake, valid])
        self.combined.compile(
            loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer
        )

    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        print('Encoder Summary:')
        print(model.summary())

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)
    
    def build_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        print('Generator Summary:')
        print(model.summary())

        noise = Input(shape=(self.latent_dim,))
        gen_img = model(noise)

        return Model(noise, gen_img)
    
    def build_discriminator(self):
        z = Input(shape=(self.latent_dim,))
        img = Input(shape=self.img_shape)
        d_ip = concatenate([z, Flatten()(img)])

        d = Dense(1024)(d_ip)
        d = LeakyReLU(0.2)(d)
        d = Dropout(0.5)(d)
        d = Dense(1024)(d)
        d = LeakyReLU(0.2)(d)
        d = Dropout(0.5)(d)
        d = Dense(1024)(d)
        d = LeakyReLU(0.2)(d)
        d = Dropout(0.5)(d)
        validity = Dense(1, activation='sigmoid')(d)

        return Model([z, img], validity)
    
    def train(self, epochs, batch_size=128, sample_interval=50):
        # load data
        dataset = image_dataset_from_directory(
            '../data/', batch_size=32, image_size=(64, 64)
        )

        # ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for batch_images, batch_labels in dataset:
                batch_images = batch_images / 127.5 - 1.0
                gray_images = tf.image.rgb_to_grayscale(batch_images)
                gray_images = tf.image.resize(gray_images, [64, 64])
                
                # Train Discriminator

                # select random batch of images and encode
                indices = np.random.randint(0, len(gray_images), batch_size)
                idx = tf.constant(indices, dtype=tf.int32)
                imgs = tf.gather(gray_images, idx)
                z = self.encoder.predict(imgs)

                # noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # generate batch of new images
                gen_imgs = self.generator.predict(noise)

                # training the discriminator on real and fake images
                d_loss_real = self.discriminator.train_on_batch([z, imgs], valid)
                d_loss_fake = self.discriminator.train_on_batch([noise, gen_imgs], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train Generator

                g_loss = self.combined.train_on_batch([noise, imgs], [valid, fake])

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

                self.generator_losses.append(g_loss[0])
                self.discriminator_losses.append(d_loss[0])
                self.discriminator_acc.append(100*d_loss[1])

                if epoch % sample_interval == 0 or epoch == epochs - 1:
                    self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # rescaling images
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, ax = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                ax[i, j].axis('off')
                cnt = cnt + 1
        fig.savefig(f'../generated_images_bigan/{epoch}.jpg')
        plt.close()

    def plot_losses_and_accuracy(self, epochs):
        epochs_range = np.linspace(1, epochs, len(self.generator_losses))

        fig, ax1 = plt.subplots(figsize=(24, 8), dpi=100)

        ax1.plot(epochs_range, self.generator_losses, label='Generator Loss', color='crimson')
        ax1.plot(epochs_range, self.discriminator_losses, label='Discriminator Loss', color='dodgerblue')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y')
        ax1.legend(loc=(1.15, 0.8))

        ax2 = ax1.twinx()
        ax2.plot(epochs_range, self.discriminator_acc, label='Discriminator Accuracy', color='khaki')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='y')
        ax2.legend(loc=(1.15, 0.3))

        fig.tight_layout()
        fig.savefig('../generated_images_bigan/losses_and_accuracy_bigan.jpg')
        plt.close(fig)


if __name__ == '__main__':
    bigan = BiGAN()
    bigan.train(epochs=50, batch_size=32, sample_interval=10)
    bigan.plot_losses_and_accuracy(epochs=50)