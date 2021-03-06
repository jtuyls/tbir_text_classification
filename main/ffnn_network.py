import os
import math
import time
import sklearn.metrics
import numpy as np
import tensorflow as tf


class Network(object):

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def build_network(self, input_size, X_, keep_prob):

        # setup hidden layer
        with tf.name_scope("hidden1"):
            hidden1_units = 50
            W = tf.Variable(tf.truncated_normal(shape=[input_size, hidden1_units],
                                                stddev=1.0 / math.sqrt(float(input_size))),
                            name="weights")
            B = tf.Variable(tf.zeros([hidden1_units]),
                            name="biases")
            hidden1 = tf.nn.relu(tf.matmul(X_, W) + B)

            hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

        with tf.name_scope("hidden2"):
            hidden2_units = 400
            W = tf.Variable(tf.truncated_normal(shape=[hidden1_units, hidden2_units],
                                                stddev=1.0 / math.sqrt(float(input_size))),
                            name="weights")
            B = tf.Variable(tf.zeros([hidden2_units]),
                            name="biases")
            hidden2 = tf.nn.relu(tf.matmul(hidden1_dropout, W) + B)

            hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

        # with tf.name_scope("hidden3"):
        #     hidden3_units = 200
        #     W = tf.Variable(tf.truncated_normal(shape=[hidden2_units, hidden3_units],
        #                                         stddev=1.0 / math.sqrt(float(input_size))),
        #                     name="weights")
        #     B = tf.Variable(tf.zeros([hidden3_units]),
        #                     name="biases")
        #     hidden3 = tf.nn.relu(tf.matmul(hidden2_dropout, W) + B)
        #
        #     hidden3_dropout = tf.nn.dropout(hidden3, keep_prob)

        with tf.name_scope("output_layer"):
            output_units = 2
            W = tf.Variable(tf.truncated_normal(shape=[hidden2_units, output_units],
                                                stddev=1.0 / math.sqrt(float(input_size))),
                            name="weights")
            B = tf.Variable(tf.zeros([output_units]),
                            name="biases")
            output_layer = tf.matmul(hidden2_dropout, W) + B

        return output_layer

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def train_network(self, network_pred, X_train, y_train, X_valid, y_valid, X_, y_, keep_prob, batch_size, num_epochs=100):
        y_ = tf.to_int64(y_)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_, logits=network_pred, name='xentropy')

        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        tf.summary.scalar('loss', loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            print("Start training")
            for epoch in range(num_epochs):

                start_time = time.time()
                train_err = 0
                train_batches = 0
                for batch in self.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                    _, c, pred = sess.run([optimizer, loss, network_pred], feed_dict={X_: batch[0], y_: batch[1], keep_prob: .7})
                    #print(sess.run(tf.nn.softmax(pred)))
                    #print(sess.run(tf.one_hot(batch[1], 2)))
                    train_err += c
                    train_batches += 1

                pred_train = sess.run(network_pred, feed_dict={X_: X_train, y_: y_train, keep_prob: 1.})
                predictions, targets = np.round(sess.run(tf.nn.softmax(pred_train))), sess.run(tf.one_hot(y_train, 2))
                train_acc = np.mean(predictions == targets)
                train_f1 = sklearn.metrics.f1_score(targets, predictions, average='macro')

                val_err = 0
                val_batches = 0
                for batch in self.iterate_minibatches(X_valid, y_valid, batch_size, shuffle=True):
                    c, pred = sess.run([loss, network_pred], feed_dict={X_: batch[0], y_: batch[1], keep_prob: 1.})
                    val_err += c
                    #val_acc += np.mean(np.round(sess.run(tf.nn.softmax(pred))) == sess.run(tf.one_hot(batch[1], 2)))
                    val_batches += 1

                pred_valid = sess.run(network_pred, feed_dict={X_: X_valid, y_: y_valid, keep_prob: 1.})
                predictions, targets = np.round(sess.run(tf.nn.softmax(pred_valid))), sess.run(tf.one_hot(y_valid, 2))
                val_acc = np.mean(predictions == targets)
                val_f1 = sklearn.metrics.f1_score(targets, predictions, average='macro')

                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  training accuracy: \t\t{:.2f} %".format(train_acc * 100))
                print("  validation accuracy: \t\t{:.2f} %".format(val_acc * 100))
                print("  training f1: \t\t{:.2f}".format(train_f1))
                print("  validation f1: \t\t{:.2f}".format(val_f1))
            print("Stop training")

            pred_valid = sess.run(network_pred, feed_dict={X_: X_valid, y_: y_valid, keep_prob: 1.})
            confidence_scores = np.amax(sess.run(tf.nn.softmax(pred_valid)), axis=1)
            predictions = np.round(sess.run(tf.nn.softmax(pred_valid)))
            return predictions, confidence_scores


    def main(self, batch_size, num_epochs):
        self.X_train, self.y_train, self.X_valid, self.y_valid, test_idx = self.data_loader.get_data()
        print(self.X_train)

        input_size = self.X_train.shape[1]
        X_ = tf.placeholder(tf.float32, shape=(None, input_size))
        y_ = tf.placeholder(tf.int32, shape=(None))
        keep_prob = tf.placeholder(tf.float32)

        network_pred = self.build_network(input_size=input_size, X_=X_, keep_prob=keep_prob)

        predictions, conf_scores = self.train_network(network_pred=network_pred,
                                         X_train=self.X_train,
                                         y_train=self.y_train,
                                         X_valid=self.X_valid,
                                         y_valid=self.y_valid,
                                         X_=X_,
                                         y_=y_,
                                         keep_prob=keep_prob,
                                         batch_size=batch_size,
                                         num_epochs=num_epochs)

        filename = "scorer/test.pred"
        self.write_predictions_to_file(predictions, conf_scores, test_idx, filename)

    def write_predictions_to_file(self, pred, conf_scores, validation_ids, filename):
        if os.path.exists(filename):
            self.clean_file(filename)
        line = "{} \t {} \t 0 \t {} \t {} \n"
        true_array = np.array([0., 1.])
        for i, elem in enumerate(validation_ids):
            self.write_line(filename,
                            line.format(elem['q_id'], elem['a_id'],
                                        conf_scores[i] if np.array_equal(pred[i],true_array) else (1-conf_scores[i]),
                                        "true" if np.array_equal(pred[i],true_array) else "false"))


    def write_line(self, filename, line):
        f = open(filename, 'a')
        f.write(line)
        f.close()

    def clean_file(self, filename):
        f = open(filename, 'w')
        f.close()





