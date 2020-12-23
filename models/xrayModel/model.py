from __future__ import absolute_import
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import datetime
import math
import sys
from PIL import Image

class Model(tf.keras.Model):
    def __init__(self):
        """
        Architechture for CNN
        """
        super(Model, self).__init__()

        self.batch_size = 50
        self.num_classes = 2

        # Initialize all hyperparameters
        self.learning_rate = 0.0008
        self.num_epochs = 2
        self.is_testing = False

        # Hyperparameters dealing with how data is read from memory

        # Stop after ith segment. Assign -1 to allow all data segments to be read
        self.stop_at_segment = 1

        # Sets the size of the training and testing data memory segments
        self.train_segment_size = 1000
        self.test_segment_size = 1000
        

        # Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([25,25,1,100], stddev=.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([15,15,100,50], stddev=.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3,3,50,25], stddev=.1))

        self.D1 = tf.keras.layers.Dense(25, activation="relu", 
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=1e-3),
                                        bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                        activity_regularizer=tf.keras.regularizers.l2(1e-3))
        self.D2 = tf.keras.layers.Dense(25, activation="relu", 
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=1e-3),
                                        bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                        activity_regularizer=tf.keras.regularizers.l2(1e-3))
        self.D3 = tf.keras.layers.Dense(25, activation="relu", 
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=1e-3),
                                        bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                        activity_regularizer=tf.keras.regularizers.l2(1e-3))
        self.D4 = tf.keras.layers.Dense(25, activation="relu", 
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=1e-3),
                                        bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                        activity_regularizer=tf.keras.regularizers.l2(1e-3))
        self.D5 = tf.keras.layers.Dense(25, activation="relu", 
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=1e-3),
                                        bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                        activity_regularizer=tf.keras.regularizers.l2(1e-3))
        self.D6 = tf.keras.layers.Dense(self.num_classes)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images for training
        :param is_testing: a boolean that determines whether to use dropout
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it is (batch_size, 2)
        """

        layer1Output = tf.nn.conv2d(inputs, self.filter1, strides=9, padding='SAME')
        (mean1, variance1) = tf.nn.moments(layer1Output, [0,1,2])
        layer1Output = tf.nn.batch_normalization(layer1Output, mean1, variance1, variance_epsilon=0.00000001, offset=None, scale=None)
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.max_pool(layer1Output, 3, 3, padding='SAME')

        layer2Output = tf.nn.conv2d(layer1Output, self.filter2, strides=7, padding='SAME')
        (mean2, variance2) = tf.nn.moments(layer2Output, [0,1,2])
        tf.nn.batch_normalization(layer2Output, mean2, variance2, variance_epsilon=0.00000001, offset=None, scale=None)
        layer2Output = tf.nn.relu(layer2Output)
        layer2Output = tf.nn.max_pool(layer2Output, 2, 2, padding='SAME')

        layer3Output = tf.nn.conv2d(layer2Output, self.filter3, strides=1, padding='SAME')
        (mean5, variance5) = tf.nn.moments(layer3Output, [0,1,2])
        tf.nn.batch_normalization(layer3Output, mean5, variance5, variance_epsilon=0.00000001, offset=None, scale=None)
        layer3Output = tf.nn.relu(layer3Output)

        layer3Output = tf.reshape(layer3Output, [len(inputs), -1])
        layer3Output = self.D1(layer3Output)

        if not self.is_testing:
            tf.nn.dropout(layer3Output, rate=0.6)
        layer3Output = self.D2(layer3Output)

        if not self.is_testing:
            tf.nn.dropout(layer3Output, rate=0.5)
        layer3Output = self.D3(layer3Output)

        if not self.is_testing:
            tf.nn.dropout(layer3Output, rate=0.5)
        layer3Output = self.D4(layer3Output)

        if not self.is_testing:
            tf.nn.dropout(layer3Output, rate=0.5)
        layer3Output = self.D5(layer3Output)

        logits = self.D6(layer3Output)
    
        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: None
    '''
    BATCH_SZ = model.batch_size

    inds = np.arange(0, np.shape(train_inputs)[0])
    np.random.shuffle(inds)
    train_inputs = train_inputs[inds]
    train_labels = train_labels[inds]
    
    steps = 0
    for i in range(0, np.shape(train_inputs)[0], BATCH_SZ):
        steps += 1
        image = train_inputs[i:i + BATCH_SZ]
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = train_labels[i:i + BATCH_SZ]
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        loss = None
        with tf.GradientTape() as tape:
            predictions = model.call(image)
            loss = model.loss(predictions, label)

        train_acc = model.accuracy(predictions, label)
        #print("Loss: {} | Accuracy on training set after {} steps: {}".format(str(loss.numpy())[ : 6], steps, train_acc))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels, setType):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :param setType: validation or test string
    :return: test accuracy
    """
    BATCH_SZ = model.batch_size
    accs = []
    steps = 0

    model.is_testing = True
    for i in range(0, np.shape(test_inputs)[0], BATCH_SZ):
        steps += 1
        image = test_inputs[i:i + BATCH_SZ]
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = test_labels[i:i + BATCH_SZ]
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        predictions = model.call(image)
        loss = model.loss(predictions, label)
        acc = model.accuracy(predictions, label)
        print("Loss: {} | Accuracy on {} set after {} steps: {}".format(str(loss.numpy())[ : 6], setType, steps, acc))
        accs.append(acc)
    model.is_testing = False
    return tf.reduce_mean(tf.convert_to_tensor(accs))


def main():
    '''
    Executes training and testing steps
    
    :return: None
    '''
    model = Model()
    if len(sys.argv) != 2:

        flag = False
        (inp_train, lab_train, pn, pp, end) = None, None, 0, 0, None
        if model.stop_at_segment == 1:
            flag = True
            (inp_train, lab_train, pn, pp, end) = get_data("../data/train", segment=model.train_segment_size)

        for epoch in range(model.num_epochs):
            pn = 0
            pp = 0
            mem_seg = 0
            end = False
            while not end:
                mem_seg += 1
                if not flag:
                    (inp_train, lab_train, pn, pp, end) = None, None, pn, pp, None
                    (inp_train, lab_train, pn, pp, end) = get_data("../data/train", 
                                                                segment=model.train_segment_size, positionN=pn, positionP=pp)
                if end:
                    break
                #print("\nTRAIN DATA SEGMENT: {} | EPOCH: {}\n".format(mem_seg, epoch + 1))
                train(model, inp_train, lab_train)
                if not model.stop_at_segment == -1 and mem_seg == model.stop_at_segment:
                    end = True
            if epoch < model.num_epochs - 1:
                    print("\nCURRENT TEST ACCURACY\n")
                    end2 = False
                    ppn = 0
                    ppp = 0
                    accum = []
                    while not end2:
                        (inp_test, lab_test, ppn, ppp, end2) = get_data("../data/test", 
                                                        segment=model.test_segment_size, positionN=ppn, positionP=ppp)
                        accum.append(test(model, inp_test, lab_test, setType="test"))
                    curr = tf.reduce_mean(tf.convert_to_tensor(accum))
                    print("\nAGGREGATE TEST ACCURACY: {}\n".format(curr.numpy()))


    inp_train =  None
    lab_train = None
    
    images = []

    if len(sys.argv) != 2:
            model.save_weights("weights", save_format='tf')
    else:
        image = Image.open(sys.argv[1])
        image = image.resize((150, 150))
        image = np.array(image).astype('float32') / 255.0
        image = np.reshape(image, (-1, 1, 150, 150))
        image = np.transpose(image, axes=[0,2,3,1])
        images.append(image)
        
        images = np.array(images)

        model(np.zeros((1,150,150,1)))
        model.load_weights("weights")
        logits = model.call(images)

        pred = tf.argmax(logits, 1)

        if pred[0] == 1:
            print("Normal")
            sys.stdout.flush()
        else:
            print("Pneumonia")
            sys.stdout.flush()


if __name__ == '__main__':
    main()
