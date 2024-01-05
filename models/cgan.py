# Conditional GAN

import numpy as np 
import matplotlib.pyplot as plt 

from keras.layers import Input, Dense, Flatten, Dropout, Reshape, BatchNormalization, LeakyReLU, Embedding, multiply
from keras.models import Sequential, Model
from keras.optimizers.legacy import Adam
from keras.utils import image_dataset_from_directory

import tensorflow as tf

class CGAN():
    def __init__(self) -> None:
        # input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1 # gray scale images
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2 # number of labels
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

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        self.discriminator.trainable = False

        valid = self.discriminator([img, label])

        # combined model generator + discriminator, trains to fool the discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(
            loss=['binary_crossentropy'],
            optimizer=optimizer
        )

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        print('Generator Summary:')
        print(model.summary())

        noise = Input(shape=(self.latent_dim, ))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_ip = multiply([noise, label_embedding])
        img = model(model_ip)

        return Model([noise, label], img)
    
    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        print('Discriminator Summary:')
        print(model.summary())

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        print(label_embedding.shape)
        print(flat_img.shape)

        model_ip = multiply([flat_img, label_embedding])
        validity = model(model_ip)

        return Model([img, label], validity)
    
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

                # select random batch of images
                indices = np.random.randint(0, len(gray_images), batch_size)
                idx = tf.constant(indices, dtype=tf.int32)
                imgs = tf.gather(gray_images, idx)
                labels = tf.gather(batch_labels, idx).numpy()

                # noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                random_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

                # generate batch of new images
                gen_imgs = self.generator.predict([noise, random_labels])

                # training the discriminator on real and fake images
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train Generator

                # condition on labels
                sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

                # train the generator on noise
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                self.generator_losses.append(g_loss)
                self.discriminator_losses.append(d_loss[0])
                self.discriminator_acc.append(100*d_loss[1])

                if epoch % sample_interval == 0 or epoch == epochs -1:
                    self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([[i % self.num_classes] for i in range(r * c)])

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # rescaling images
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, ax = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                ax[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                ax[i, j].set_title("Label %s" % sampled_labels[cnt])
                ax[i, j].axis('off')
                cnt = cnt + 1
        fig.savefig(f'../generated_images_cgan/{epoch}.jpg')
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
        fig.savefig('../generated_images_cgan/losses_and_accuracy_cgan.jpg')
        plt.close(fig)


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=50, batch_size=32, sample_interval=10)
    cgan.plot_losses_and_accuracy(epochs=50)