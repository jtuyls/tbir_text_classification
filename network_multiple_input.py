

import math
import time
import sklearn.metrics
import numpy as np
import tensorflow as tf



from network import Network


class NetworkMI(Network):

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def build_network(self, input_size_Q, input_size_A, XQ_, XA_, keep_prob):

        # setup input layers
        input_units = 50
        with tf.name_scope("input1"):
            W = tf.Variable(tf.truncated_normal(shape=[input_size_Q, input_units],
                                                stddev=1.0 / math.sqrt(float(input_size_Q))),
                            name="weights")
            B = tf.Variable(tf.zeros([input_units]),
                            name="biases")
            input1 = tf.nn.relu(tf.matmul(XQ_, W) + B)

            input_1_dropout = tf.nn.dropout(input1, keep_prob) # (*, input_units)

        with tf.name_scope("input2"):
            W = tf.Variable(tf.truncated_normal(shape=[input_size_A, input_units],
                                                stddev=1.0 / math.sqrt(float(input_size_A))),
                            name="weights")
            B = tf.Variable(tf.zeros([input_units]),
                            name="biases")
            input2 = tf.nn.relu(tf.matmul(XA_, W) + B)

            input_2_dropout = tf.nn.dropout(input2, keep_prob) # (*, input_units)

        with tf.name_scope("merge"):
            merge = tf.concat([input_1_dropout, input_2_dropout], 1)

        with tf.name_scope("hidden2"):
            hidden2_units = 600
            W = tf.Variable(tf.truncated_normal(shape=[input_units*2, hidden2_units],
                                                stddev=1.0 / math.sqrt(float(input_size_Q) + float(input_size_A))),
                            name="weights")
            B = tf.Variable(tf.zeros([hidden2_units]),
                            name="biases")
            hidden2 = tf.nn.relu(tf.matmul(merge, W) + B)

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
                                                stddev=1.0 / math.sqrt(float(input_size_Q) + float(input_size_A))),
                            name="weights")
            B = tf.Variable(tf.zeros([output_units]),
                            name="biases")
            output_layer = tf.matmul(hidden2_dropout, W) + B

        return output_layer

    def iterate_minibatches(self, inputs, inputs2, targets, batchsize, shuffle):
        assert len(inputs) == len(targets) == len(inputs2)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], inputs2[excerpt], targets[excerpt]

    def train_network(self, network_pred, X_train_Q, X_train_A, y_train, X_valid_Q, X_valid_A, y_valid, XQ_, XA_, y_,
                      keep_prob, batch_size, num_epochs=100):
        y_ = tf.to_int64(y_)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_, logits=network_pred, name='xentropy')

        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        tf.summary.scalar('loss', loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

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
                for batch in self.iterate_minibatches(X_train_Q, X_train_A, y_train, batch_size, shuffle=True):
                    _, c, pred = sess.run([optimizer, loss, network_pred], feed_dict={XQ_: batch[0], XA_: batch[1], y_: batch[2], keep_prob: .9})
                    #print(sess.run(tf.nn.softmax(pred)))
                    #print(sess.run(tf.one_hot(batch[1], 2)))
                    train_err += c
                    train_batches += 1

                pred_train = sess.run(network_pred, feed_dict={XQ_: X_train_Q, XA_: X_train_A, y_: y_train, keep_prob: 1.})
                predictions, targets = np.round(sess.run(tf.nn.softmax(pred_train))), sess.run(tf.one_hot(y_train, 2))
                train_acc = np.mean(predictions == targets)
                train_f1 = sklearn.metrics.f1_score(targets, predictions, average='macro')

                val_err = 0
                val_batches = 0
                for batch in self.iterate_minibatches(X_valid_Q, X_valid_A, y_valid, batch_size, shuffle=True):
                    c, pred = sess.run([loss, network_pred], feed_dict={XQ_: batch[0], XA_: batch[1], y_: batch[2], keep_prob: 1.})
                    val_err += c
                    #val_acc += np.mean(np.round(sess.run(tf.nn.softmax(pred))) == sess.run(tf.one_hot(batch[1], 2)))
                    val_batches += 1

                pred_valid = sess.run(network_pred, feed_dict={XQ_: X_valid_Q, XA_: X_valid_A, y_: y_valid, keep_prob: 1.})
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

            pred_valid = sess.run(network_pred, feed_dict={XQ_: X_valid_Q, XA_: X_valid_A, y_: y_valid, keep_prob: 1.})
            confidence_scores = np.amax(sess.run(tf.nn.softmax(pred_valid)), axis=1)
            predictions = np.round(sess.run(tf.nn.softmax(pred_valid)))
            return predictions, confidence_scores


    def main(self, batch_size, num_epochs, validation_split=0.05):
        self.X_train_Q, self.X_train_A, self.y_train, self.X_test_Q, \
            self.X_test_A, self.y_test, _ = self.data_loader.get_data_separate_sentences()
        print(self.X_train_Q)

        # Validation
        #validation_split_nb = int((1-validation_split)*len(self.X_train_Q))
        #self.self.X_train_Q = self.self.X_train_Q[:validation_split_nb]
        #self.self.X_train_A = self.self.X_train_Q[:validation_split_nb]

        input_size_Q = self.X_train_Q.shape[1]
        input_size_A = self.X_train_A.shape[1]
        XQ_ = tf.placeholder(tf.float32, shape=(None, input_size_Q))
        XA_ = tf.placeholder(tf.float32, shape=(None, input_size_A))
        y_ = tf.placeholder(tf.int32, shape=(None))
        keep_prob = tf.placeholder(tf.float32)

        network_pred = self.build_network(input_size_Q=input_size_Q, input_size_A=input_size_A, XQ_=XQ_, XA_=XA_, keep_prob=keep_prob)

        predictions, conf_scores = self.train_network(network_pred=network_pred,
                                                      X_train_Q=self.X_train_Q,
                                                      X_train_A=self.X_train_A,
                                                      y_train=self.y_train,
                                                      X_valid_Q=self.X_test_Q,
                                                      X_valid_A=self.X_test_A,
                                                      y_valid=self.y_test,
                                                      XQ_=XQ_,
                                                      XA_=XA_,
                                                      y_=y_,
                                                      keep_prob=keep_prob,
                                                      batch_size=batch_size,
                                                      num_epochs=num_epochs)
        filename = "scorer/test.pred"
        validation_ids = self.data_loader.get_validation_ids()
        self.write_predictions_to_file(predictions, conf_scores, validation_ids, filename)




