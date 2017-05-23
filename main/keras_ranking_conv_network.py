




# The code in this file is based on the blog post http://smerity.com/articles/2015/keras_qa.html
#   and github repository https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
#   and blog post: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

from __future__ import print_function

import os
import numpy as np

import keras

from keras.utils.data_utils import get_file
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

GLOVE_DIR = 'word_embeddings/glove.6B/'
MAX_SEQUENCE_LENGTH = 250
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class KerasRankingConv(object):

    def __init__(self, data_loader_pairwise):
        self.data_loader = data_loader_pairwise

    def build_network(self, label_size, input_shape_Q, input_shape_A, word_index, embedding_matrix,
                      dropout=0.3):
        print('Build model...')
        print('input_shape_Q: {}'.format(input_shape_Q))
        print('input_shape_A: {}'.format(input_shape_A))
        question = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        encoded_question = layers.Embedding(len(word_index) + 1,
                                            EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)(question)
        #encoded_question = layers.Dropout(dropout)(encoded_question)
        # encoded_question = RNN(RNN_SIZE)(encoded_question)
        # encoded_question = layers.RepeatVector(RNN_SIZE)(encoded_question)
        print(encoded_question)

        answer_1 = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        encoded_answer_1 = layers.Embedding(len(word_index) + 1,
                                            EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)(answer_1)
        #encoded_answer_1 = layers.Dropout(dropout)(encoded_answer_1)
        #encoded_answer_1 = RNN(embed_hidden_size)(encoded_answer_1)
        #encoded_answer_1 = layers.RepeatVector(input_shape_Q)(encoded_answer_1)
        print(encoded_answer_1)

        answer_2 = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        encoded_answer_2 = layers.Embedding(len(word_index) + 1,
                                            EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)(answer_2)
        #encoded_answer_2 = layers.Dropout(dropout)(encoded_answer_2)
        #encoded_answer_2 = RNN(embed_hidden_size)(encoded_answer_2)
        #encoded_answer_2 = layers.RepeatVector(input_shape_Q)(encoded_answer_2)
        print(encoded_answer_2)

        merged = layers.add([encoded_question, encoded_answer_1, encoded_answer_2])
        print(merged)
        merged = Conv1D(128, 5, activation='relu')(merged)
        print(merged)
        merged = layers.Dropout(dropout)(merged)
        print(merged)
        merged = MaxPooling1D(5)(merged)
        print(merged)
        merged = Conv1D(128, 5, activation='relu')(merged)
        print(merged)
        merged = layers.Dropout(dropout)(merged)
        print(merged)
        merged = MaxPooling1D(35)(merged)
        print(merged)
        merged = Flatten()(merged)
        print(merged)
        merged = layers.Dense(128, activation='relu')(merged)
        print(merged)
        preds = layers.Dense(label_size, activation='softmax')(merged)
        print(preds)

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

    def main(self, loss_name="categorical_crossentropy", optimizer_name="adam",
             learning_rate=0.0001, dropout=0.3, validation_split=0.1, batch_size=32, num_epochs=40, test=False,
             prediction_filename="scorer/keras_rnn.pred", save_data_after_loading=True):
        if test:
            X_train_Q, X_train_A_1, X_train_A_2, y_train, \
            X_test_Q, X_test_A_1, X_test_A_2, \
            y_test, train_idx, test_idx, test_idx_org, word_index = self.data_loader.get_data_separate_sentences_test(
                save=save_data_after_loading)
        else:
            X_train_Q, X_train_A_1, X_train_A_2, y_train, \
            X_test_Q, X_test_A_1, X_test_A_2, \
            y_test, train_idx, test_idx, test_idx_org, word_index = self.data_loader.get_data_separate_sentences(
                save=save_data_after_loading)

        #print(train_idx[:100])
        print(y_train[:10])
        y_train = np.asarray(to_categorical(y_train))
        print(y_train[:10])
        print('vocabulary size: {}'.format(len(word_index)))

        # Retrieve glove embedding matrix
        embeddings_index = {}
        f = open('word_embeddings/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        # Compute embedding matrix
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector


        input_shape_Q = X_train_Q.shape[1]
        input_shape_A = X_train_A_1.shape[1]

        preds, question, answer_1, answer_2 = self.build_network(input_shape_Q=input_shape_Q,
                                                                 input_shape_A=input_shape_A,
                                                                 word_index=word_index,
                                                                 embedding_matrix=embedding_matrix,
                                                                 dropout=dropout,
                                                                 label_size=y_train.shape[1])

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
        predictions = np.argmax(pred_valid, axis=1)

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
        better = 2
        even = 1
        conf_scores = []
        for i, index in enumerate(test_idx_org):
            q_id = index['q_id']
            a_id = index['a_id']
            answers_idx_list = [test_idx.index(item) for item in test_idx if
                                ((item['q_id'] == index['q_id']) and (item['a_id'] == index['a_id']))]
            better_rank = len([predictions[ind] for ind in answers_idx_list if predictions[ind] == better])
            even_rank = len([predictions[ind] for ind in answers_idx_list if predictions[ind] == even])
            # print("Q_id: {}, A_id: {}, rank: {}".format(q_id, a_id, rank))
            rank = better_rank*2 + even_rank
            conf_scores.append(rank/18.0)

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
            write_line(filename,
                       line.format(elem['q_id'], elem['a_id'], conf_scores[i], "true"))









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











