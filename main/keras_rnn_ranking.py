


# The code in this file is based on the blog post http://smerity.com/articles/2015/keras_qa.html
#   and github repository https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

from __future__ import print_function

import numpy as np

import keras

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

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

class KerasRNNRanking(object):

    def __init__(self, data_loader_pairwise):
        self.data_loader = data_loader_pairwise

    def build_network(self, label_size, input_shape_Q, input_shape_A, vocab_size,
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

        answer_1 = layers.Input(shape=(input_shape_A,), dtype='int32')
        encoded_answer_1 = layers.Embedding(vocab_size, embed_hidden_size)(answer_1)
        encoded_answer_1 = layers.Dropout(dropout)(encoded_answer_1)
        encoded_answer_1 = RNN(embed_hidden_size)(encoded_answer_1)
        encoded_answer_1 = layers.RepeatVector(input_shape_Q)(encoded_answer_1)
        print(encoded_answer_1)

        answer_2 = layers.Input(shape=(input_shape_A,), dtype='int32')
        encoded_answer_2 = layers.Embedding(vocab_size, embed_hidden_size)(answer_2)
        encoded_answer_2 = layers.Dropout(dropout)(encoded_answer_2)
        encoded_answer_2 = RNN(embed_hidden_size)(encoded_answer_2)
        encoded_answer_2 = layers.RepeatVector(input_shape_Q)(encoded_answer_2)
        print(encoded_answer_2)

        merged = layers.add([encoded_question, encoded_answer_1, encoded_answer_2])
        merged = RNN(rnn_size)(merged)
        print(merged)
        merged = layers.Dropout(dropout)(merged)
        print(merged)
        preds = layers.Dense(label_size, activation='softmax')(merged)

        return preds, question, answer_1, answer_2

    def get_optimizer(self, optimizer_name, lr):
        if optimizer_name == "adam":
            return keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif optimizer_name == "sgd":
            return keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        elif optimizer_name == "rmsprop":
            return keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            raise ValueError("No such optimizer")

    def train_network(self, preds, question, answer_1, answer_2, X_train_Q, X_train_A_1, X_train_A_2, y_train,
                      batch_size, num_epochs, loss_name='categorical_crossentropy', optimizer_name="adam", learning_rate=0.0001, validation_split=0.1):

        optimizer = self.get_optimizer(optimizer_name=optimizer_name, lr=learning_rate)

        model = Model([question, answer_1, answer_2], preds)
        model.compile(optimizer=optimizer,
                      loss=loss_name,
                      metrics=['accuracy'])

        print('Training')
        model.fit([X_train_Q, X_train_A_1, X_train_A_2], y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_split=validation_split)

        return model

    def main(self, embed_hidden_size, rnn_size, loss_name="categorical_crossentropy", optimizer_name="adam",
             learning_rate=0.0001, dropout=0.3, validation_split=0.1, batch_size=32, num_epochs=40, test=False,
             prediction_filename="scorer/keras_rnn.pred", save_data_after_loading=True):
        if test:
            X_train_Q, X_train_A_1, X_train_A_2, y_train, \
            X_test_Q, X_test_A_1, X_test_A_2, \
            y_test, train_idx, test_idx, test_idx_org, vocab_size = self.data_loader.get_data_separate_sentences_test(
                save=save_data_after_loading)
        else:
            X_train_Q, X_train_A_1, X_train_A_2, y_train, \
            X_test_Q, X_test_A_1, X_test_A_2, \
            y_test, train_idx, test_idx, test_idx_org, vocab_size = self.data_loader.get_data_separate_sentences(
                save=save_data_after_loading)

        #print(train_idx[:100])
        #print(y_train[:100])
        y_train = to_categorical(y_train)
        label_size = y_train.shape[1]
        #print(y_train)
        print('vocabulary size: {}'.format(vocab_size))

        input_shape_Q = X_train_Q.shape[1]
        input_shape_A = X_train_A_1.shape[1]

        preds, question, answer_1, answer_2 = self.build_network(label_size=label_size,
                                                                 input_shape_Q=input_shape_Q,
                                                                 input_shape_A=input_shape_A,
                                                                 vocab_size=vocab_size,
                                                                 embed_hidden_size=embed_hidden_size,
                                                                 rnn_size=rnn_size,
                                                                 dropout=dropout)

        self.model = self.train_network(preds=preds,
                                        question=question,
                                        answer_1=answer_1,
                                        answer_2=answer_2,
                                        X_train_Q=X_train_Q,
                                        X_train_A_1=X_train_A_1,
                                        X_train_A_2=X_train_A_2,
                                        y_train=y_train,
                                        loss_name=loss_name,
                                        optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        validation_split=validation_split,
                                        batch_size=batch_size,
                                        num_epochs=num_epochs)

        # PREDICTIONS
        # loss, acc = model.evaluate([X_valid_Q, X_valid_A], y_valid,
        #                           batch_size=BATCH_SIZE)
        # print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
        pred_valid = self.model.predict([X_test_Q, X_test_A_1, X_test_A_2], batch_size=batch_size)

        confidence_scores = np.amax(pred_valid, axis=1)
        predictions = np.round(pred_valid)

        print(pred_valid)
        print(confidence_scores)
        print(predictions)

        # Calculate the confidence scores using the rank
        conf_scores = self.rank_data(predictions, test_idx, test_idx_org)
        self.write_predictions_to_file(conf_scores, test_idx_org, prediction_filename)

    def rank_data(self, predictions, test_idx, test_idx_org):
        # rank the each (question answer_1) pair according to the number of times it beats the a (question answer_2) pair
        # Rank 18 means that answer 1 is better than all other answers
        # Rank 0 means that answer 1 is worse than all other answers
        print("Rank data")
        better = np.array([0., 0., 1.])
        even = np.array([0., 1., 0.])
        conf_scores = []
        for i, index in enumerate(test_idx_org):
            q_id = index['q_id']
            a_id = index['a_id']
            answers_idx_list = [test_idx.index(item) for item in test_idx if
                                ((item['q_id'] == index['q_id']) and (item['a_id'] == index['a_id']))]
            better_rank = len(
                [predictions[ind] for ind in answers_idx_list if np.array_equal(predictions[ind], better)])
            even_rank = len([predictions[ind] for ind in answers_idx_list if np.array_equal(predictions[ind], even)])
            # print("Q_id: {}, A_id: {}, rank: {}".format(q_id, a_id, rank))
            rank = better_rank * 2 + even_rank
            conf_scores.append(rank / 18.0)

        print("Done ranking data")
        return conf_scores

    def write_predictions_to_file(self, conf_scores, validation_ids, filename):

        def write_line(filename, line):
            f = open(filename, 'a')
            f.write(line)
            f.close()

        def clean_file(filename):
            f = open(filename, 'w')
            f.close()

        if os.path.exists(filename):
            clean_file(filename)
        line = "{} \t {} \t 0 \t {} \t {} \n"
        for i, elem in enumerate(validation_ids):
            write_line(filename, line.format(elem['q_id'], elem['a_id'], conf_scores[i], "true"))



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











