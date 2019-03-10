import pandas as pd
import tensorflow as tf
import numpy as np


class Tensorflow:
    train_data_set = []
    test_data_set = []
    X_training_data = []
    Y_training_data = []

    X_testing_data = []
    Y_testing_data = []

    l = 0.01
    epochs = 1500
    input_size = 8
    feature_size = 8
    label_size = 8
    hidden_layer_nodes = 3

    def load_data_set(self):
        self.X_training_data = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,1]]
        self.Y_training_data = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,1]]


    def train(self, X, Y):

        weights1 = tf.get_variable(name="weights1", shape=[self.feature_size, self.hidden_layer_nodes],
                                   initializer=tf.contrib.layers.xavier_initializer())
        biases1 = tf.get_variable(name="biases1", shape=[self.hidden_layer_nodes], initializer=tf.zeros_initializer())
        layer_1_output = tf.nn.sigmoid(tf.matmul(X, weights1) + biases1)

        weights2 = tf.get_variable(name="weights2", shape=[self.hidden_layer_nodes, self.label_size],
                                   initializer=tf.contrib.layers.xavier_initializer())
        biases2 = tf.get_variable(name="biases2", shape=[self.label_size], initializer=tf.zeros_initializer())
        prediction = tf.nn.sigmoid(tf.matmul(layer_1_output, weights2) + biases2)

        cost = tf.reduce_mean(tf.squared_difference(prediction, Y))
        correct = tf.equal(tf.round(prediction), Y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        optimizer = tf.train.AdamOptimizer(self.l).minimize(cost)
        return [optimizer, cost, accuracy]

    def run(self, optimizer, X, Y, cost, accuracy):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                session.run(optimizer, feed_dict={X: self.X_training_data, Y: self.Y_training_data})
                training_accuracy = session.run(accuracy, feed_dict={X: self.X_training_data, Y: self.Y_training_data})
                # testing_accuracy = session.run(accuracy, feed_dict={X: self.X_testing_data, Y: self.Y_testing_data})
                training_cost = session.run(cost, feed_dict={X: self.X_training_data, Y: self.Y_training_data})
                #testing_cost = session.run(cost, feed_dict={X: self.X_training_data, Y: self.Y_training_data})
                print("Epoch: {} - Training Cost: {} Training Accuracy: {}"
                      .format(epoch, training_cost, training_accuracy))
            final_cost_training = session.run(cost, feed_dict={X: self.X_training_data, Y: self.Y_training_data})
            #final_cost_testing = session.run(cost, feed_dict={X: self.X_testing_data, Y: self.Y_testing_data})
            final_accuracy_training = session.run(accuracy,
                                                  feed_dict={X: self.X_training_data, Y: self.Y_training_data})
            #final_accuracy_testing = session.run(accuracy, feed_dict={X: self.X_testing_data, Y: self.Y_testing_data})

            print("Final Training cost: {}".format(final_cost_training))
            #print("Final Testing cost: {}".format(final_cost_testing))
            print("Final Training Accuracy: {}".format(final_accuracy_training))
            #print("Final Testing Accuracy: {}".format(final_accuracy_testing))


def main():
    tensorflow = Tensorflow()
    tensorflow.load_data_set()
    X = tf.placeholder(tf.float32, shape=(None, tensorflow.feature_size))
    Y = tf.placeholder(tf.float32, shape=(None, tensorflow.label_size))
    optimizer, cost, accuracy = tensorflow.train(X, Y)
    tensorflow.run(optimizer, X, Y, cost, accuracy)


if __name__ == '__main__':
    main()
