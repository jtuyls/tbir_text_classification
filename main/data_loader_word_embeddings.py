
import os.path
import re
import itertools
import json

import numpy as np
import xml.etree.ElementTree as ET

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from main.data_loader import DataLoader

GLOVE_DIR = 'word_embeddings/glove.6B/'
MAX_SEQUENCE_LENGTH = 250
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class DataLoaderWordEmbeddings(DataLoader):

    def __init__(self, data_path_train_1, data_path_train_2, data_path_validation, data_path_test=None):
        super(DataLoaderWordEmbeddings, self).__init__(data_path_train_1, data_path_train_2, data_path_validation, data_path_test)

    def get_data_separate_sentences(self, save=True):
        print("Load data pairwise")

        # Try loading from disk
        if os.path.isfile('data/word_train_data.npz') and os.path.isfile('data/word_test_data.npz')\
                and os.path.isfile('data/word_info_data.npz') and os.path.isfile('data/word_index.json'):
            print("Load data from cached files")
            file_train = np.load('data/word_train_data.npz')
            X_train_Q = file_train['X_train_Q']
            X_train_A_1 = file_train['X_train_A_1']
            X_train_A_2 = file_train['X_train_A_2']
            y_train = file_train['y_train']

            file_test = np.load('data/word_test_data.npz')
            X_test_Q = file_test['X_test_Q']
            X_test_A_1 = file_test['X_test_A_1']
            X_test_A_2 = file_test['X_test_A_2']
            y_test = file_test['y_test']

            file_info = np.load('data/word_info_data.npz')
            idx_train = list(file_info['idx_train'])
            idx_test = list(file_info['idx_test'])
            idx_test_original = list(file_info['idx_test_original'])
            with open('data/word_index.json', 'r') as fp:
                line = fp.readline()
                word_index = json.loads(line)
        else:
            print("Compute data and save to files")
            Q_sentences_train, A_sentences_train, \
                labels_train, idx_train = self._read_raw_xml_data_separate_sentences(self.data_path_train_1)
            Q_sentences_train_2, A_sentences_train_2, \
                labels_train_2, idx_train_2 = self._read_raw_xml_data_separate_sentences(self.data_path_train_2)
            Q_sentences_test, A_sentences_test, \
                labels_test, idx_test_original = self._read_raw_xml_data_separate_sentences(self.data_path_validation)

            Q_sentences_train = Q_sentences_train + Q_sentences_train_2
            A_sentences_train = A_sentences_train + A_sentences_train_2
            labels_train = labels_train + labels_train_2
            idx_train = idx_train + idx_train_2

            Q_sentences_train = Q_sentences_train
            A_sentences_train = A_sentences_train
            labels_train = labels_train
            idx_train = idx_train

            all_sentences = Q_sentences_train + A_sentences_train + Q_sentences_test + A_sentences_test
            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(all_sentences)
            Q_sentences_train_sequences = tokenizer.texts_to_sequences(Q_sentences_train)
            A_sentences_train_sequences = tokenizer.texts_to_sequences(A_sentences_train)
            Q_sentences_test_sequences = tokenizer.texts_to_sequences(Q_sentences_test)
            A_sentences_test_sequences = tokenizer.texts_to_sequences(A_sentences_test)

            word_index = tokenizer.word_index
            print('Found %s unique tokens.' % len(word_index))

            Q_sentences_train_padded = pad_sequences(Q_sentences_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            A_sentences_train_padded = pad_sequences(A_sentences_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            Q_sentences_test_padded = pad_sequences(Q_sentences_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            A_sentences_test_padded = pad_sequences(A_sentences_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

            #labels_train = to_categorical(np.asarray(labels_train))
            print('Shape of Q data tensor:', Q_sentences_train_padded.shape)
            print('Shape of A data tensor:', A_sentences_train_padded.shape)
            print('Shape of Q test tensor:', Q_sentences_test_padded.shape)
            print('Shape of A test tensor:', A_sentences_test_padded.shape)
            print('Shape of label tensor:', len(labels_train))
            print('Shape of idx tensor:', len(idx_train))

            # Transform to pairwise data
            print("Transform to pairwise data")
            X_train_Q, X_train_A_1, X_train_A_2, y_train, idx_train = \
                self._compute_pairwise_data(Q_sentences_train_padded, A_sentences_train_padded, labels_train,
                                            idx_train)

            print(len([True for x in y_train if x == 2]), len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))

            #y_train = to_categorical(np.asarray(y_train))
            #print('Shape of label tensor:', y_train.shape)

            X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_test = \
                self._compute_pairwise_data_test(Q_sentences_test_padded, A_sentences_test_padded, idx_test_original)

            print(len([True for x in y_test if x == 2]), len([True for x in y_test if x == 1]), len([True for x in y_test if x == 0]))

            # Save info to file
            if save == True:
                np.savez('data/word_train_data', X_train_Q=X_train_Q, X_train_A_1=X_train_A_1, X_train_A_2=X_train_A_2,
                         y_train=y_train)
                np.savez('data/word_test_data.npz', X_test_Q=X_test_Q, X_test_A_1=X_test_A_1,
                         X_test_A_2=X_test_A_2,
                         y_test=y_test)
                np.savez('data/word_info_data.npz', idx_train=idx_train, idx_test=idx_test,
                         idx_test_original=idx_test_original)
                with open('data/word_index.json', 'w') as fp:
                    json.dump(word_index, fp)

        print('X_train_Q shape: {}'.format(X_train_Q.shape))
        print('X_train_A_1 shape: {}'.format(X_train_A_1.shape))
        print('X_train_A_2 shape: {}'.format(X_train_A_2.shape))
        print('y_train shape: {}'.format(y_train.shape))
        print('idx train length: {}'.format(len(idx_train)))

        print('X_test_Q shape: {}'.format(X_test_Q.shape))
        print('X_test_A_1 shape: {}'.format(X_test_A_1.shape))
        print('X_test_A_2 shape: {}'.format(X_test_A_2.shape))
        print('idx test length: {}'.format(len(idx_test)))
        print('idx test original length: {}'.format(len(idx_test_original)))

        print("Done loading data pairwise")
        return X_train_Q, X_train_A_1, X_train_A_2, y_train, X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_train, idx_test, idx_test_original, word_index



    def get_data_separate_sentences_test(self, save=True):
        print("Load data pairwise")

        # Try loading from disk
        if os.path.isfile('data/word_train_data_test.npz') and os.path.isfile('data/word_test_data_test.npz') \
                and os.path.isfile('data/word_info_data_test.npz') and os.path.isfile('data/word_index_test.json'):
            print("Load data from cached files")
            file_train = np.load('data/word_train_data_test.npz')
            X_train_Q = file_train['X_train_Q']
            X_train_A_1 = file_train['X_train_A_1']
            X_train_A_2 = file_train['X_train_A_2']
            y_train = file_train['y_train']

            file_test = np.load('data/word_test_data_test.npz')
            X_test_Q = file_test['X_test_Q']
            X_test_A_1 = file_test['X_test_A_1']
            X_test_A_2 = file_test['X_test_A_2']
            y_test = file_test['y_test']

            file_info = np.load('data/word_info_data_test.npz')
            idx_train = list(file_info['idx_train'])
            idx_test = list(file_info['idx_test'])
            idx_test_original = list(file_info['idx_test_original'])
            with open('data/word_index_test.json', 'r') as fp:
                line = fp.readline()
                word_index = json.loads(line)
        else:
            print("Compute data and save to files")
            Q_sentences_train, A_sentences_train, \
            labels_train, idx_train = self._read_raw_xml_data_separate_sentences(self.data_path_train_1)
            Q_sentences_train_2, A_sentences_train_2, \
            labels_train_2, idx_train_2 = self._read_raw_xml_data_separate_sentences(self.data_path_train_2)
            Q_sentences_valid, A_sentences_valid, \
            labels_valid, idx_valid_original = self._read_raw_xml_data_separate_sentences(self.data_path_validation)
            Q_sentences_test, A_sentences_test, \
            labels_test, idx_test_original = self._read_raw_xml_data_separate_sentences(self.data_path_test)

            Q_sentences_train = Q_sentences_train + Q_sentences_train_2 + Q_sentences_valid
            A_sentences_train = A_sentences_train + A_sentences_train_2 + A_sentences_valid
            labels_train = labels_train + labels_train_2 + labels_valid
            idx_train = idx_train + idx_train_2 + idx_valid_original

            Q_sentences_train = Q_sentences_train
            A_sentences_train = A_sentences_train
            labels_train = labels_train
            idx_train = idx_train

            all_sentences = Q_sentences_train + A_sentences_train + Q_sentences_test + A_sentences_test
            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(all_sentences)
            Q_sentences_train_sequences = tokenizer.texts_to_sequences(Q_sentences_train)
            A_sentences_train_sequences = tokenizer.texts_to_sequences(A_sentences_train)
            Q_sentences_test_sequences = tokenizer.texts_to_sequences(Q_sentences_test)
            A_sentences_test_sequences = tokenizer.texts_to_sequences(A_sentences_test)

            word_index = tokenizer.word_index
            print('Found %s unique tokens.' % len(word_index))

            Q_sentences_train_padded = pad_sequences(Q_sentences_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            A_sentences_train_padded = pad_sequences(A_sentences_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            Q_sentences_test_padded = pad_sequences(Q_sentences_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
            A_sentences_test_padded = pad_sequences(A_sentences_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

            # labels_train = to_categorical(np.asarray(labels_train))
            print('Shape of Q data tensor:', Q_sentences_train_padded.shape)
            print('Shape of A data tensor:', A_sentences_train_padded.shape)
            print('Shape of Q test tensor:', Q_sentences_test_padded.shape)
            print('Shape of A test tensor:', A_sentences_test_padded.shape)
            print('Shape of label tensor:', len(labels_train))
            print('Shape of idx tensor:', len(idx_train))

            # Transform to pairwise data
            print("Transform to pairwise data")
            X_train_Q, X_train_A_1, X_train_A_2, y_train, idx_train = \
                self._compute_pairwise_data(Q_sentences_train_padded, A_sentences_train_padded, labels_train,
                                            idx_train)

            print(len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))

            # y_train = to_categorical(np.asarray(y_train))
            # print('Shape of label tensor:', y_train.shape)

            X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_test = \
                self._compute_pairwise_data_test(Q_sentences_test_padded, A_sentences_test_padded, idx_test_original)

            print(len([True for x in y_test if x == 1]), len([True for x in y_test if x == 0]))

            # Save info to file
            if save == True:
                np.savez('data/word_train_data_test', X_train_Q=X_train_Q, X_train_A_1=X_train_A_1, X_train_A_2=X_train_A_2,
                         y_train=y_train)
                np.savez('data/word_test_data_test.npz', X_test_Q=X_test_Q, X_test_A_1=X_test_A_1,
                         X_test_A_2=X_test_A_2,
                         y_test=y_test)
                np.savez('data/word_info_data_test.npz', idx_train=idx_train, idx_test=idx_test,
                         idx_test_original=idx_test_original)
                with open('data/word_index_test.json', 'w') as fp:
                    json.dump(word_index, fp)

        print('X_train_Q shape: {}'.format(X_train_Q.shape))
        print('X_train_A_1 shape: {}'.format(X_train_A_1.shape))
        print('X_train_A_2 shape: {}'.format(X_train_A_2.shape))
        print('y_train shape: {}'.format(y_train.shape))
        print('idx train length: {}'.format(len(idx_train)))

        print('X_test_Q shape: {}'.format(X_test_Q.shape))
        print('X_test_A_1 shape: {}'.format(X_test_A_1.shape))
        print('X_test_A_2 shape: {}'.format(X_test_A_2.shape))
        print('idx test length: {}'.format(len(idx_test)))
        print('idx test original length: {}'.format(len(idx_test_original)))

        print("Done loading data pairwise")
        return X_train_Q, X_train_A_1, X_train_A_2, y_train, X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_train, idx_test, idx_test_original, word_index


    #### HELPER FUNCTIONS ####

    def _compute_pairwise_data_test(self, X_Q, X_A, idx):
        X_Q_new = []
        X_A_1_new = []
        X_A_2_new = []
        labels_new = []
        idx_new = []
        for i, index in enumerate(idx):
            #print(i, index)
            c_idx = [idx.index(item) for item in idx if item['q_id'] == index['q_id']]
            for c_id in c_idx:
                if idx[c_id]['a_id'] != index['a_id']:
                    X_Q_new.append(X_Q[i])
                    X_A_1_new.append(X_A[i])
                    X_A_2_new.append(X_A[c_id])
                    new_index = dict(index)
                    new_index['a_id_2'] = idx[c_id]['a_id']
                    idx_new.append(new_index)
        return np.array(X_Q_new), np.array(X_A_1_new), np.array(X_A_2_new), [], idx_new

    def _compute_pairwise_data(self, X_Q, X_A, labels, idx):
        X_Q_new = []
        X_A_1_new = []
        X_A_2_new = []
        labels_new = []
        idx_new = []
        for i, index in enumerate(idx):
            # print(i, index)
            c_idx = [idx.index(item) for item in idx if item['q_id'] == index['q_id']]
            for c_id in c_idx:
                if labels[i] > labels[c_id]:
                    X_Q_new.append(X_Q[i])
                    X_A_1_new.append(X_A[i])
                    X_A_2_new.append(X_A[c_id])
                    labels_new.append(2)
                    new_index = dict(index)
                    new_index['a_id_2'] = idx[c_id]['a_id']
                    new_index['label'] = 2
                    idx_new.append(new_index)
                if labels[i] < labels[c_id]:
                    X_Q_new.append(X_Q[i])
                    X_A_1_new.append(X_A[i])
                    X_A_2_new.append(X_A[c_id])
                    labels_new.append(0)
                    new_index = dict(index)
                    new_index['a_id_2'] = idx[c_id]['a_id']
                    new_index['label'] = 0
                    idx_new.append(new_index)
                if labels[i] == labels[c_id]:
                    X_Q_new.append(X_Q[i])
                    X_A_1_new.append(X_A[i])
                    X_A_2_new.append(X_A[c_id])
                    labels_new.append(1)
                    new_index = dict(index)
                    new_index['a_id_2'] = idx[c_id]['a_id']
                    new_index['label'] = 1
                    idx_new.append(new_index)
        return np.array(X_Q_new), np.array(X_A_1_new), np.array(X_A_2_new), np.array(labels_new), idx_new

    def _read_raw_xml_data_separate_sentences(self, data_path):
        """
        Read the raw xml data

        :param data_path: the datapath to xml file with questions and answers

        :return:
        List: question sentences
        List: answer sentences
        List: labels
        List: indexes of the question answer pairs
        """
        label_to_int = {
            'good': 2,
            'potentiallyuseful': 1,
            'bad': 0,
            '?': -1,
        }
        idx = []
        Q_sentences = []
        A_sentences = []
        labels = []
        tree = ET.parse(data_path)
        xml = tree.getroot()
        for thread in xml:
            question_info = dict()
            question_sentence = ""
            for rel in thread:
                if rel.tag == "RelQuestion":
                    question_info['q_id'] = rel.attrib['RELQ_ID']
                    question_sentence = question_sentence + "_" + rel.attrib['RELQ_CATEGORY'] +  "_" + str(rel[0].text) + "_" + str(rel[1].text)
                elif rel.tag == "RelComment":
                    question_answer_info = dict(question_info)
                    question_answer_info['a_id'] = rel.attrib['RELC_ID']
                    answer_sentence = str(rel[0].text)
                    Q_sentences.append(question_sentence)
                    A_sentences.append(answer_sentence)
                    answer_label = label_to_int[rel.attrib['RELC_RELEVANCE2RELQ'].lower()]
                    labels.append(answer_label)
                    idx.append(question_answer_info)
            #Q_sentences.append(question_sentence)
            #A_sentences.append(answer_sentences)
            #labels.append(answer_labels)

        return Q_sentences, A_sentences, labels, idx