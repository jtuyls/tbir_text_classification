


import math
import time
import sklearn.metrics
import numpy as np
import tensorflow as tf

from sklearn.model_selection import ShuffleSplit


from main.ffnn_network import Network


class FFNNRankingNetwork(Network):

    def __init__(self, data_loader_pairwise):
        self.data_loader = data_loader_pairwise
        self.model = None

    def build_network(self, input_size_Q, input_size_A, XQ_, XA1_, XA2_, keep_prob_, input_units=50):

        # setup input layers

        with tf.name_scope("input1"):
            W = tf.Variable(tf.truncated_normal(shape=[input_size_Q, input_units],
                                                stddev=1.0 / math.sqrt(float(input_size_Q))),
                            name="weights")
            B = tf.Variable(tf.zeros([input_units]),
                            name="biases")
            input1 = tf.nn.relu(tf.matmul(XQ_, W) + B)

            input_1_dropout = tf.nn.dropout(input1, keep_prob_) # (*, input_units)

        with tf.name_scope("input2"):
            W = tf.Variable(tf.truncated_normal(shape=[input_size_A, input_units],
                                                stddev=1.0 / math.sqrt(float(input_size_A))),
                            name="weights")
            B = tf.Variable(tf.zeros([input_units]),
                            name="biases")
            input2 = tf.nn.relu(tf.matmul(XA1_, W) + B)

            input_2_dropout = tf.nn.dropout(input2, keep_prob_) # (*, input_units)

        with tf.name_scope("input3"):
            W = tf.Variable(tf.truncated_normal(shape=[input_size_A, input_units],
                                                stddev=1.0 / math.sqrt(float(input_size_A))),
                            name="weights")
            B = tf.Variable(tf.zeros([input_units]),
                            name="biases")
            input3 = tf.nn.relu(tf.matmul(XA2_, W) + B)

            input_3_dropout = tf.nn.dropout(input2, keep_prob_)  # (*, input_units)

        with tf.name_scope("merge"):
            merge = tf.concat([input_1_dropout, input_2_dropout, input_3_dropout], 1)

        with tf.name_scope("hidden2"):
            hidden2_units = 600
            W = tf.Variable(tf.truncated_normal(shape=[input_units*3, hidden2_units],
                                                stddev=1.0 / math.sqrt(float(input_size_Q) + float(input_size_A))),
                            name="weights")
            B = tf.Variable(tf.zeros([hidden2_units]),
                            name="biases")
            hidden2 = tf.nn.relu(tf.matmul(merge, W) + B)

            hidden2_dropout = tf.nn.dropout(hidden2, keep_prob_)

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

            #output_layer = tf.nn.sigmoid(output_layer)

        return output_layer

    def iterate_minibatches(self, inputs, inputs2, inputs3, targets, batchsize, shuffle):
        assert len(inputs) == len(targets) == len(inputs2) == len(inputs3)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], inputs2[excerpt], inputs3[excerpt], targets[excerpt]

    def get_optimizer(self, optimizer_name, lr):
        if optimizer_name == "adam":
            return tf.train.AdamOptimizer(learning_rate=lr)
        elif optimizer_name == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif optimizer_name == "momentum":
            return tf.train.MomentumOptimizer(learning_rate=lr)
        elif optimizer_name == "rmsprop":
            return tf.train.RMSPropOptimizer(learning_rate=lr)
        else:
            raise ValueError("No such optimizer")

    def get_loss(self, loss_name, y_, logits):
        if loss_name == "cross_entropy":
            return tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        elif loss_name == "hinge":
            return tf.losses.hinge_loss(labels=y_, logits=logits)
        else:
            raise ValueError("No such optimizer")

    def train_network(self, network_pred,
                      X_train_Q, X_train_A_1, X_train_A_2, y_train,
                      X_valid_Q, X_valid_A_1, X_valid_A_2, y_valid,
                      XQ_, XA1_, XA2_, y_, keep_prob_,
                      loss="cross_entropy", optimizer_name="sgd", learning_rate=0.0001,
                      batch_size=32, dropout=0.1, num_epochs=100):
        #y_ = tf.to_int64(y_)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #   labels=y_, logits=network_pred, name='xentropy')

        loss_function = self.get_loss(loss, y_, network_pred)

        loss = tf.reduce_mean(loss_function)

        #loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        tf.summary.scalar('loss', loss)

        optimizer = self.get_optimizer(optimizer_name, learning_rate).minimize(loss)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            y_train = sess.run(tf.one_hot(y_train, 2))
            y_valid = sess.run(tf.one_hot(y_valid, 2))

            print("Start training")
            for epoch in range(num_epochs):

                start_time = time.time()
                train_err = 0
                train_batches = 0
                for batch in self.iterate_minibatches(X_train_Q, X_train_A_1, X_train_A_2, y_train, batch_size, shuffle=True):
                    _, c, pred = sess.run([optimizer, loss, network_pred], feed_dict={XQ_: batch[0], XA1_: batch[1], XA2_: batch[2], y_: batch[3], keep_prob_: 1-dropout})
                    #print(sess.run(tf.nn.softmax(pred)))
                    #print(sess.run(tf.one_hot(batch[1], 2)))
                    train_err += c
                    train_batches += 1

                pred_train = sess.run(network_pred, feed_dict={XQ_: X_train_Q, XA1_: X_train_A_1, XA2_: X_train_A_2, y_: y_train, keep_prob_: 1.})
                #predictions, targets = np.round(sess.run(tf.nn.softmax(pred_train))), sess.run(tf.one_hot(y_train, 2))
                predictions, targets = np.round(sess.run(tf.nn.sigmoid(pred_train))), y_train
                train_acc = np.mean(predictions == targets)
                train_f1 = sklearn.metrics.f1_score(targets, predictions, average='macro')

                val_err = 0
                val_batches = 0
                for batch in self.iterate_minibatches(X_valid_Q, X_valid_A_1, X_valid_A_2, y_valid, batch_size, shuffle=True):
                    c, pred = sess.run([loss, network_pred], feed_dict={XQ_: batch[0], XA1_: batch[1], XA2_: batch[2], y_: batch[3], keep_prob_: 1.})
                    val_err += c
                    #val_acc += np.mean(np.round(sess.run(tf.nn.softmax(pred))) == sess.run(tf.one_hot(batch[1], 2)))
                    val_batches += 1

                pred_valid = sess.run(network_pred, feed_dict={XQ_: X_valid_Q, XA1_: X_valid_A_1, XA1_: X_valid_A_2, y_: y_valid, keep_prob_: 1.})
                #predictions, targets = np.round(sess.run(tf.nn.softmax(pred_valid))), sess.run(tf.one_hot(y_valid, 2))
                predictions, targets = np.round(sess.run(tf.nn.sigmoid(pred_valid))), y_valid
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

            return network_pred


    def make_predictions(self, XQ_, XA1_, XA2_, keep_prob_,
                         X_test_Q, X_test_A_1, X_test_A_2):
        if self.model == None:
            raise ValueError("Model is not yet trained")

        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            pred_test = sess.run(self.model, feed_dict={XQ_: X_test_Q, XA1_: X_test_A_1, XA2_: X_test_A_2, keep_prob_: 1.})
            confidence_scores = np.amax(sess.run(tf.nn.softmax(pred_test)), axis=1)
            predictions = np.round(sess.run(tf.nn.softmax(pred_test)))

            return predictions, confidence_scores



    def main(self, batch_size, num_epochs, dropout=0.1, validation_split=0.20,
             optimizer_name="sgd", learning_rate=0.0001, loss="cross_entropy",
             prediction_filename="scorer/ffnn_ranking.pred",
             test=False, save_data_after_loading=True):
        if test:
            X_train_Q, X_train_A_1, X_train_A_2, y_train, \
                X_test_Q, X_test_A_1, X_test_A_2, \
                y_test, train_idx, test_idx, test_idx_org, _ = self.data_loader.get_data_separate_sentences_test(save=save_data_after_loading)
        else:
            X_train_Q, X_train_A_1, X_train_A_2, y_train, \
                X_test_Q, X_test_A_1, X_test_A_2, \
                y_test, train_idx, test_idx, test_idx_org, _ = self.data_loader.get_data_separate_sentences(save=save_data_after_loading)
        print("Length training dataset: {}".format(X_train_Q.shape[0]))
        print("Length test dataset: {}".format(X_test_Q.shape[0]))

        # reshape y_train
        #y_train = np.reshape(y_train, (len(y_train), 1))
        #y_test = np.reshape(y_test, (len(y_test), 1))

        # Shuffled split of training data in training set and validation set
        rs = ShuffleSplit(test_size=validation_split)
        for train_idx, valid_idx in rs.split(X_train_Q):
            pass
        print(len(train_idx), len(valid_idx))
        self.X_train_Q = X_train_Q[train_idx]
        self.X_train_A_1 = X_train_A_1[train_idx]
        self.X_train_A_2 = X_train_A_2[train_idx]
        self.y_train = y_train[train_idx]

        self.X_valid_Q = X_train_Q[valid_idx]
        self.X_valid_A_1 = X_train_A_1[valid_idx]
        self.X_valid_A_2 = X_train_A_2[valid_idx]
        self.y_valid = y_train[valid_idx]

        self.X_test_Q = X_test_Q
        self.X_test_A_1 = X_test_A_1
        self.X_test_A_2 = X_test_A_2

        # Define input placeholders
        input_size_Q = self.X_train_Q.shape[1]
        input_size_A = self.X_train_A_1.shape[1]
        XQ_ = tf.placeholder(tf.float32, shape=(None, input_size_Q))
        XA1_ = tf.placeholder(tf.float32, shape=(None, input_size_A))
        XA2_ = tf.placeholder(tf.float32, shape=(None, input_size_A))
        y_ = tf.placeholder(tf.int32, shape=(None, 2))
        keep_prob_ = tf.placeholder(tf.float32)

        network_pred = self.build_network(input_size_Q=input_size_Q, input_size_A=input_size_A, XQ_=XQ_, XA1_=XA1_, XA2_=XA2_, keep_prob_=keep_prob_)

        self.model = self.train_network(network_pred=network_pred,
                                        X_train_Q=self.X_train_Q,
                                        X_train_A_1=self.X_train_A_1,
                                        X_train_A_2=self.X_train_A_2,
                                        y_train=self.y_train,
                                        X_valid_Q=self.X_valid_Q,
                                        X_valid_A_1=self.X_valid_A_1,
                                        X_valid_A_2=self.X_valid_A_2,
                                        y_valid=self.y_valid,
                                        XQ_=XQ_,
                                        XA1_=XA1_,
                                        XA2_=XA2_,
                                        y_=y_,
                                        keep_prob_=keep_prob_,
                                        loss=loss,
                                        optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        dropout=dropout,
                                        num_epochs=num_epochs)

        predictions, conf_scores = self.make_predictions(XQ_=XQ_,
                                                         XA1_=XA1_,
                                                         XA2_=XA2_,
                                                         keep_prob_=keep_prob_,
                                                         X_test_Q=self.X_test_Q,
                                                         X_test_A_1=self.X_test_A_1,
                                                         X_test_A_2=self.X_test_A_2)

        # Calculate the confidence scores using the rank
        conf_scores = self.rank_data(predictions, test_idx, test_idx_org)
        predictions_for_writing = np.array([[0., 1.] for _ in range(len(conf_scores))])
        self.write_predictions_to_file(predictions_for_writing, conf_scores, test_idx_org, prediction_filename)

    def rank_data(self, predictions, test_idx, test_idx_org):
        # rank the each (question answer_1) pair according to the number of times it beats the a (question answer_2) pair
        # Rank 9 means that answer 1 is better than all other answers
        # Rank 0 means that answer 1 is worse than all other answers
        true_array = np.array([0., 1.])
        conf_scores = []
        for i, index in enumerate(test_idx_org):
            q_id = index['q_id']
            a_id = index['a_id']
            answers_idx_list = [test_idx.index(item) for item in test_idx if
                                ((item['q_id'] == index['q_id']) and (item['a_id'] == index['a_id']))]
            rank = len([predictions[ind] for ind in answers_idx_list if np.array_equal(predictions[ind],true_array)])
            #print("Q_id: {}, A_id: {}, rank: {}".format(q_id, a_id, rank))
            conf_scores.append(1/(1+rank))

        return conf_scores



