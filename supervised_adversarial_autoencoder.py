import tensorflow as tf
import numpy as np
import datetime
import os
import time 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import sin, cos, sqrt
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.metrics import confusion_matrix
from sklearn import mixture 


flags = tf.app.flags
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

# Get the Fashion-mnist data
mnist = input_data.read_data_sets('./Data', one_hot=True)

# Parameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
z_dim = 100
batch_size = 100
n_epochs = 1000
learning_rate = 0.00001
beta1 = 0.9
n_labels = 10
results_path = './Results/Supervised'
distributionName = str(z_dim) + 'GMM'

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
y_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_labels], name='Labels')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
manual_decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim + n_labels], name='Decoder_input')


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_{6}_Supervised". \
        format(datetime.datetime.now(), z_dim, learning_rate, batch_size, n_epochs, beta1, distributionName)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    other_results = results_path + folder_name + '/others'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
        os.mkdir(other_results)
    return tensorboard_path, saved_model_path, log_path, other_results


def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    nx, ny = 10, 10
    tam = 10 
    FONT_SIZE = 5
    # random_inputs = np.random.randn(10, z_dim) #* 5.
    # print('Original input:\t ' + str(random_inputs))
    # print(random_inputs.shape)
    # # random_inputs = np.random.randint(0, 10, size=[10, z_dim])
    random_inputs = np.random.randint(0, 10, size=[tam])
    random_inputs = gaussian_mixture(tam, z_dim, label_indices=random_inputs)
    # print('Random inputs:\t ' + str(random_inputs))
    # print(random_inputs.shape)
    # print('GMM inputs:\t ' + str(GMM))
    # print(GMM.shape)

    sample_y = np.identity(10)
    print('sample y ' + str(sample_y))
    print(sample_y.shape)

    # random_inputs = GMM
    # random_inputs = np.identity(10)
    print('Random input ' + str(random_inputs))
    print(random_inputs.shape)

    plt.subplot()
    # gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
    gs = gridspec.GridSpec(nx, ny)

    i = 0
    for r in random_inputs:
        for t in sample_y:
            # print('R value ' + str(r))
            # print('T value ' + str(t))
            r, t = np.reshape(r, (1, z_dim)), np.reshape(t, (1, n_labels))
            dec_input = np.concatenate((t, r), 1)
            # print('decode input ' + str(dec_input))
            # print('R value ' + str(r))
            # print('T value ' + str(t))

            x = sess.run(op, feed_dict={manual_decoder_input: dec_input})
            ax = plt.subplot(gs[i])
            
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')

            # ax.set_ylabel(str(r), fontsize=FONT_SIZE, rotation=180)
            # if i < 1:
                # ax.set_ylabel('Input %s' % str(r), fontsize=FONT_SIZE, rotation=180)
                # ax.set_xlabel('Label %s' % str(t), fontsize=FONT_SIZE)
            
            

            # ax.set_xlabel('input %s' % str(r))
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_aspect('auto')
            i += 1
    plt.show()

def generate_image_grid_gmm(sess, op):
    nx, ny = z_dim, n_labels
    # Training Gaussian Mixture Model
    gmm = train_gaussian_mixture(n_labels, z_dim, 1000)
    # Generate random vector of each Gaussian 
    # random_inputs = gmm.sample(n_samples=z_dim)[0]
    random_inputs = np.identity(z_dim)
    print('Random input ' + str(random_inputs))
    print(random_inputs.shape)
    # Generate image for each class 
    sample_y = np.identity(n_labels)
    print('sample y ' + str(sample_y))
    print(sample_y.shape)

    plt.subplot()

    if z_dim > 10:
        nx, ny = 10, n_labels
        temp = random_inputs[range(0, z_dim, 10)]
        # temp2 = sample_y[range(0, z_dim, 10)]
        print('temporal y ' + str(temp))
        print(temp.shape)
        # print('temporal2 y ' + str(temp2))
        # print(temp2.shape)
        random_inputs = temp
        # sample_y = temp2

    # gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
    gs = gridspec.GridSpec(nx, ny)

    i = 0
    for r in random_inputs:
        for t in sample_y:
            r, t = np.reshape(r, (1, z_dim)), np.reshape(t, (1, n_labels))
            dec_input = np.concatenate((t, r), 1)
            x = sess.run(op, feed_dict={manual_decoder_input: dec_input})
            ax = plt.subplot(gs[i]) 
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')

            # ax.set_ylabel(str(r), fontsize=5, rotation=180)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
            i += 1

    plt.show()


def generate_decode_output(sess, op):

    input_value = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    input_value = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    input_label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    input_value, input_label = np.reshape(input_value, (1, z_dim)), np.reshape(input_label, (1, n_labels))
    print('input value ' + str(input_value))
    print('input label ' + str(input_label))
    dec_input = np.concatenate((input_label, input_value), 1)
    print('decode input ' + str(dec_input))

    x = sess.run(op, feed_dict={manual_decoder_input: dec_input})
    img = np.array(x.tolist()).reshape(28, 28)
    plt.imshow(img, cmap='gray')

    plt.show()

def save_confusion_matrix(classes_sz, conf_values, filename, num_samples):
    classes = np.arange(classes_sz)
    cmap = plt.cm.Blues
    plt.figure()

    plt.imshow(conf_values, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix of ' + str(num_samples) + ' samples')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conf_values.max() / 2.
    for i, j in itertools.product(range(conf_values.shape[0]), range(conf_values.shape[1])):
        plt.text(j, i, format(conf_values[i, j], fmt), horizontalalignment="center", color="white" if conf_values[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(filename)

def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # weights = tf.get_variable("weights", shape=[n1, n2],
                                  # initializer=tf.random_normal_initializer(mean=0., stddev=0.1))
        weights = tf.get_variable("weights", shape=[n1, n2], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :param supervised: True -> returns output without passing it through softmax,
                       False -> returns output after passing it through softmax.
    :return: tensor which is the classification output and a hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        return latent_variable


def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(dense(x, z_dim + n_labels, n_l2, 'd_dense_1'))
        d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output


def discriminator(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        dc_den1 = tf.nn.relu(dense(x, z_dim, n_l1, name='dc_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_output')
        return output

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    # if n_dim != 2:
        # raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z

def train_gaussian_mixture(n_gaussians, zdim, n_samples):

    # Generate vector for each centroid 
    mu = np.identity(zdim)
    mutemp = mu[range(0, zdim, 10)] #10x100
    # idx = range(1, n_gaussians+1, 2)
    idx = range(1, n_gaussians+1, 2)
    mutemp[idx] *= -1
    # Generate random samples to train GMM
    for i in range(n_gaussians):
        if i % 2 == 0:
            continue
        x_temp = np.vstack([np.random.randn(n_samples, zdim) + mutemp[i-1],
                            np.random.randn(n_samples, zdim) + mutemp[i]])

    gmm = mixture.GaussianMixture(n_components=n_gaussians, covariance_type='full')
    gmm.fit(x_temp)

    return gmm

def print_confusion_matrix():
    session = tf.Session()
    # Get the true classifications for the test-set.
    cls_true = np.argmax(mnist.test.labels, 1)
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred, feed_dict=feed_dict)
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    # Print the confusion matrix as text.
    # print(cm)
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Make various adjustments to the plot.
    plt.tight_layout()
    # plt.colorbar()
    tick_marks = np.arange(C)
    plt.xticks(tick_marks, range(C))
    plt.yticks(tick_marks, range(C))
    plt.xlabel('Predicted')
    plt.ylabel('True')

def train(train_model=True):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        # Concat class label and the encoder output
        decoder_input = tf.concat([y_input, encoder_output], 1)
        decoder_output = decoder(decoder_input)

    with tf.variable_scope(tf.get_variable_scope()):
        d_real = discriminator(real_distribution)
        d_fake = discriminator(encoder_output, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder(manual_decoder_input, reuse=True)

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Discriminator Loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()
    dc_var = [var for var in all_variables if 'dc_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]
    d_var = [var for var in all_variables if 'd_' in var.name]
    
    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(dc_loss, var_list=dc_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)

    init = tf.global_variables_initializer()

    # Reshape images to display them
    input_images = tf.reshape(x_input, [-1, 28, 28, 1])
    generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

    # Visualize gradients 
    discriminator_grad = tf.gradients(dc_loss, dc_var)
    discriminator_grad = list(zip(discriminator_grad, dc_var))

    generator_grad = tf.gradients(generator_loss, en_var)
    generator_grad = list(zip(generator_grad, en_var))

    autoencoder_grad = tf.gradients(autoencoder_loss, d_var)
    autoencoder_grad = list(zip(autoencoder_grad, d_var))

    # Tensorboard visualization
    tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    tf.summary.histogram(name='Real Distribution', values=real_distribution)
    for grad, var in discriminator_grad:
        tf.summary.histogram(var.name + '/gradient', grad)
    for grad, var in autoencoder_grad:
        tf.summary.histogram(var.name + '/gradient', grad)
    for grad, var in generator_grad:
        if grad is not None:
            tf.summary.histogram(var.name + '/gradient', grad)

    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Training Gaussian Mixture Model
    gmm = train_gaussian_mixture(n_labels, z_dim, 10000)
    # Confusion matrix params
    gmm_labels = []
    gmm_values = []
    starting = 0
    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        if train_model:
            tensorboard_path, saved_model_path, log_path, other_results = form_results()
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            for i in range(n_epochs):
                n_batches = int(mnist.train.num_examples / batch_size)
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                for b in range(1, n_batches + 1):
                    # Default conf. 1 gaussian normal distribution
                    # z_real_dist = np.random.randn(batch_size, z_dim) * 5.
                    # A mixture of 10 2D Gaussians
                    # z_values = np.random.randint(0, 10, size=[batch_size])    # []
                    # z_real_dist = gaussian_mixture(batch_size, z_dim, label_indices=z_values)

                    # A mixture of 10 Guassians in R10
                    gmm_samples = gmm.sample(n_samples=batch_size)
                    z_real_dist = gmm_samples[0]
                    if starting < 1:
                        gmm_values = np.array(gmm_samples[0])
                        gmm_labels = np.array(gmm_samples[1])
                        starting+=1
                    else:
                        gmm_values = np.concatenate([gmm_values, gmm_samples[0]])
                        gmm_labels = np.concatenate([gmm_labels, gmm_samples[1]])                    # print 'real dist ' + str(z_real_dist)

                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x, y_input: batch_y})
                    sess.run(discriminator_optimizer,
                             feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                    sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    if b % 50 == 0:
                        a_loss, d_loss, g_loss, summary = sess.run(
                            [autoencoder_loss, dc_loss, generator_loss, summary_op],
                            feed_dict={x_input: batch_x, x_target: batch_x,
                                       real_distribution: z_real_dist, y_input: batch_y})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {}, iteration: {}".format(i, b))
                        print("Autoencoder Loss: {}".format(a_loss))
                        print("Discriminator Loss: {}".format(d_loss))
                        print("Generator Loss: {}".format(g_loss))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Autoencoder Loss: {}\n".format(a_loss))
                            log.write("Discriminator Loss: {}\n".format(d_loss))
                            log.write("Generator Loss: {}\n".format(g_loss))
                    step += 1

                saver.save(sess, save_path=saved_model_path, global_step=step)

            # Saving Confusion Matrix
            gmm_predicted = gmm.predict(gmm_values)
            # true, predicted, labels, sample weights
            conf_values = confusion_matrix(gmm_labels, gmm_predicted)
            confMtx_name = other_results + "/{0}.png".format(datetime.datetime.now())

            save_confusion_matrix(n_labels, conf_values, confMtx_name, gmm_values.shape[0])

        else:
            # Get the latest results folder
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' +
                                                                     all_results[-1] + '/Saved_models/'))
            # generate_image_grid(sess, op=decoder_image)
            generate_image_grid_gmm(sess, op=decoder_image)
            # print_confusion_matrix()
            # generate_decode_output(sess, op=decoder_image)


if __name__ == '__main__':
    exeTimeFile = 'ExeTime_S-AAE_'
    if FLAGS.train:
        print("Starting Training ----------------------------------------")
        time.sleep(3)
        exeTimeFile += 'Train.txt'
    else:
        exeTimeFile +='Test.txt'

    start_time = time.time()
    train(train_model=FLAGS.train)
    with open(exeTimeFile, 'w') as file:
        file.write(str((time.time() - start_time)))
