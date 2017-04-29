
import tensorflow as tf
import numpy as np

from sklearn.decomposition import PCA

from network_multiple_input import NetworkMI


class NetworkPCA(NetworkMI):

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def train_pca_preprocessor(self, data_train, n_components):

        # Initialize and fit PCA
        self.pca = PCA(n_components=n_components)

        print("Fit PCA")
        self.pca.fit(data_train)
        print("Done fitting PCA")

    def transform_to_correct_format(self, data, q_idx, a_idx):
        print("Format transformation")
        X_q = []
        X_a = []
        for i, q_id_name in enumerate(q_idx):
            q_x = data[i]
            for j, a_id_name in enumerate(a_idx):
                if (a_id_name.split("_")[0] + "_" + a_id_name.split("_")[1]) == q_id_name:
                    a_x = data[len(q_idx) + j]
                    X_q.append(q_x)
                    X_a.append(a_x)
        print("End format transformation")
        return np.array(X_q), np.array(X_a)

    def main(self, batch_size, num_epochs, validation_split=0.05, test=False):
        if test == False:
            data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid,\
                self.y_train, self.y_test = self.data_loader.get_data_for_pca()
        else:
            data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid = self.data_loader.get_data_for_pca_test()


        # Fit preprocessor
        self.train_pca_preprocessor(data_train, 20)
        X_t = self.pca.transform(data_train)
        X_v = self.pca.transform(data_valid)

        # transform data to question and answer format of neural network
        self.X_train_Q, self.X_train_A = self.transform_to_correct_format(X_t, q_idx_train, a_idx_train)
        self.X_test_Q, self.X_test_A = self.transform_to_correct_format(X_v, q_idx_valid, a_idx_valid)

        print(self.X_train_Q.shape, self.X_train_A.shape)
        print(self.X_test_Q.shape, self.X_test_A.shape)


        input_size_Q = self.X_train_Q.shape[1]
        input_size_A = self.X_train_A.shape[1]
        XQ_ = tf.placeholder(tf.float32, shape=(None, input_size_Q))
        XA_ = tf.placeholder(tf.float32, shape=(None, input_size_A))
        y_ = tf.placeholder(tf.int32, shape=(None))
        keep_prob = tf.placeholder(tf.float32)

        network_pred = self.build_network(input_size_Q=input_size_Q, input_size_A=input_size_A, XQ_=XQ_, XA_=XA_,
                                          keep_prob=keep_prob)

        predictions, conf_scores = self.train_network(network_pred=network_pred,
                                                      X_train_Q=self.X_train_Q,
                                                      X_train_A=self.X_train_A,
                                                      y_train=self.y_train,
                                                      X_valid_Q=[],
                                                      X_valid_A=[],
                                                      y_valid=[],
                                                      XQ_=XQ_,
                                                      XA_=XA_,
                                                      y_=y_,
                                                      keep_prob=keep_prob,
                                                      batch_size=batch_size,
                                                      num_epochs=num_epochs)
        filename = "scorer/test_sup_pca.pred"
        validation_ids = self.data_loader.get_validation_ids()
        self.write_predictions_to_file(predictions, conf_scores, validation_ids, filename)