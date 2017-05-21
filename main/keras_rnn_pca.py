
# The code in this file is based on the blog post http://smerity.com/articles/2015/keras_qa.html
#   and github repository https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py


# The code in this file is based on the blog post http://smerity.com/articles/2015/keras_qa.html
#   and github repository https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

from __future__ import print_function

import numpy as np

import keras


from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.utils.np_utils import to_categorical

from sklearn.decomposition import PCA

from main.data_loader import DataLoader

from main.output_file_writer import write_predictions_to_file

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 30
# print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
#                                                            EMBED_HIDDEN_SIZE,
#                                                            SENT_HIDDEN_SIZE,
#                                                            QUERY_HIDDEN_SIZE))

class KerasRNNPCA(object):

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def build_network(self, input_shape_Q, input_shape_A, vocab_size,
                      embed_hidden_size=50, rnn_size=100, dropout=0.3):
        print('Build model...')
        print('input_shape_Q: {}'.format(input_shape_Q))
        print('input_shape_A: {}'.format(input_shape_A))
        question = layers.Input(shape=(input_shape_Q,), dtype='int32')
        encoded_question = layers.Embedding(vocab_size, embed_hidden_size)(question)
        encoded_question = layers.Dropout(dropout)(encoded_question)
        # encoded_question = RNN(RNN_SIZE)(encoded_question)
        # encoded_question = layers.RepeatVector(RNN_SIZE)(encoded_question)
        print(encoded_question)

        answer = layers.Input(shape=(input_shape_A,), dtype='int32')
        encoded_answer = layers.Embedding(vocab_size, embed_hidden_size)(answer)
        encoded_answer = layers.Dropout(dropout)(encoded_answer)
        encoded_answer = RNN(embed_hidden_size)(encoded_answer)
        encoded_answer = layers.RepeatVector(input_shape_Q)(encoded_answer)
        print(encoded_answer)

        merged = layers.add([encoded_question, encoded_answer])
        merged = RNN(rnn_size)(merged)
        print(merged)
        merged = layers.Dropout(dropout)(merged)
        print(merged)
        preds = layers.Dense(2, activation='softmax')(merged)

        return preds, question, answer

    def get_optimizer(self, optimizer_name, lr):
        if optimizer_name == "adam":
            return keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif optimizer_name == "sgd":
            return keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        elif optimizer_name == "rmsprop":
            return keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            raise ValueError("No such optimizer")

    def train_pca_preprocessor(data_train, n_components):

        # Initialize and fit PCA
        pca = PCA(n_components=n_components)

        print("Fit PCA")
        pca.fit(data_train)
        print("Done fitting PCA")

        return pca

    def transform_to_correct_format(data, q_idx, a_idx):
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

    def train_network(self, preds, question, answer, X_train_Q, X_train_A, y_train,
                      batch_size, num_epochs, optimizer_name="adam", learning_rate=0.0001, validation_split=0.2):

        optimizer = self.get_optimizer(optimizer_name=optimizer_name, lr=learning_rate)

        model = Model([question, answer], preds)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print('Training')
        model.fit([X_train_Q, X_train_A], y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_split=validation_split)

        return model

    def main(self, embed_hidden_size, rnn_size, optimizer_name="adam", learning_rate=0.0001,
             dropout=0.3, batch_size=32, num_epochs=40, test=False,
             prediction_filename="scorer/keras_rnn_pca.pred"):

        data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid, \
            y_train, y_test, test_idx = self.data_loader.get_data_for_pca()

        # Fit preprocessor
        pca = train_pca_preprocessor(data_train, 20)
        X_t = pca.transform(data_train)
        X_v = pca.transform(data_valid)

        # transform data to question and answer format of neural network
        X_train_Q, X_train_A = transform_to_correct_format(X_t, q_idx_train, a_idx_train)
        X_test_Q, X_test_A = transform_to_correct_format(X_v, q_idx_valid, a_idx_valid)

        print(X_train_Q.shape, X_train_A.shape)
        print(X_test_Q.shape, X_test_A.shape)

        print(y_train)
        y_train = to_categorical(y_train)
        print(y_train)
        y_test = to_categorical(y_test)

        print('X_train_Q.shape = {}'.format(X_train_Q.shape))
        print('X_train_A.shape = {}'.format(X_train_A.shape))
        print('y_train.shape = {}'.format(y_train.shape))
        print('X_test_Q = {}'.format(X_test_Q.shape))
        print('X_test_A = {}'.format(X_test_A.shape))
        print('vocabulary size: {}'.format(vocab_size))

        input_shape_Q = X_train_Q.shape[1]
        input_shape_A = X_train_A.shape[1]

        preds, question, answer = self.build_network(input_shape_Q=input_shape_Q,
                                                     input_shape_A=input_shape_A,
                                                     vocab_size=vocab_size,
                                                     embed_hidden_size=embed_hidden_size,
                                                     rnn_size=rnn_size,
                                                     dropout=dropout,)

        self.model = self.train_network(preds=preds,
                                        question=question,
                                        answer=answer,
                                        X_train_Q=X_train_Q,
                                        X_train_A=X_train_A,
                                        y_train=y_train,
                                        optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        num_epochs=num_epochs)

        # PREDICTIONS
        # loss, acc = model.evaluate([X_valid_Q, X_valid_A], y_valid,
        #                           batch_size=BATCH_SIZE)
        # print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
        pred_valid = model.predict([X_test_Q, X_test_A], batch_size=batch_size)

        confidence_scores = np.amax(pred_valid, axis=1)
        predictions = np.round(pred_valid)

        print(pred_valid)
        print(confidence_scores)
        print(predictions)

        write_predictions_to_file(predictions, confidence_scores, test_idx, prediction_filename)





if __name__ == "__main__":
    d = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
               'data/test_input.xml')
    keras_rnn = KerasRNN(data_loader=d)
    keras_rnn.main(embed_hidden_size=50, rnn_size=100, batch_size=40)





# ORIGINAL FROM BLOG POST
# question = layers.Input(shape=(input_shape_Q,), dtype='int32')
# encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
# encoded_question = layers.Dropout(0.3)(encoded_question)
#
# answer = layers.Input(shape=(input_shape_A,), dtype='int32')
# encoded_answer = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(answer)
# encoded_answer = layers.Dropout(0.3)(encoded_answer)
# encoded_answer = RNN(EMBED_HIDDEN_SIZE)(encoded_answer)
# encoded_answer = layers.RepeatVector(input_shape_Q)(encoded_answer)
#
# merged = layers.add([encoded_question, encoded_answer])
# merged = RNN(EMBED_HIDDEN_SIZE)(merged)
# merged = layers.Dropout(0.3)(merged)
# preds = layers.Dense(2, activation='softmax')(merged)
