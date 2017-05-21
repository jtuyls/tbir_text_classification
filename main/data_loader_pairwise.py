
import os.path
import re
import itertools

import numpy as np
import xml.etree.ElementTree as ET

from main.data_loader import DataLoader

class PairwiseDataLoader(DataLoader):

    def __init__(self, data_path_train_1, data_path_train_2, data_path_validation, data_path_test=None):
        super(PairwiseDataLoader, self).__init__(data_path_train_1, data_path_train_2, data_path_validation, data_path_test)

    def get_data_separate_sentences(self, save=True):
        print("Load data pairwise")

        # Try loading from disk
        if os.path.isfile('data/train_data.npz') and os.path.isfile('data/test_data.npz')\
                and os.path.isfile('data/info_data.npz'):
            print("Load data from cached files")
            file_train = np.load('data/train_data.npz')
            X_train_Q = file_train['X_train_Q']
            X_train_A_1 = file_train['X_train_A_1']
            X_train_A_2 = file_train['X_train_A_2']
            y_train = file_train['y_train']

            file_test = np.load('data/test_data.npz')
            X_test_Q = file_test['X_test_Q']
            X_test_A_1 = file_test['X_test_A_1']
            X_test_A_2 = file_test['X_test_A_2']
            y_test = file_test['y_test']

            file_info = np.load('data/info_data.npz')
            idx_train = list(file_info['idx_train'])
            idx_test = list(file_info['idx_test'])
            idx_test_original = list(file_info['idx_test_original'])
            vocabulary_size = int(file_info['vocabulary_size'])
        else:
            print("Compute data and save to files")
            Q_sentences_train, A_sentences_train, \
                labels_train, idx_train = self._read_raw_xml_data_separate_sentences(self.data_path_train_1)
            Q_sentences_train_2, A_sentences_train_2, \
                labels_train_2, idx_train_2 = self._read_raw_xml_data_separate_sentences(self.data_path_train_2)
            Q_sentences_test, A_sentences_test, \
                labels_test, idx_test_original = self._read_raw_xml_data_separate_sentences(self.data_path_validation)

            #print("compute pairwise data")
            #X_Q_new, X_A_1_new, X_A_2_new, labels_new = self._compute_pairwise_data(Q_sentences_train[:20], A_sentences_train[:20], labels_train[:20], idx_train[:20])
            #print([item for item in zip(X_Q_new, X_A_1_new, X_A_2_new)])

            Q_padded_sentences = self._pad_sentences(Q_sentences_train + Q_sentences_train_2 + Q_sentences_test)
            Q_sentences_train_padded = Q_padded_sentences[:len(Q_sentences_train) + len(Q_sentences_train_2)]
            Q_sentences_test_padded = Q_padded_sentences[len(Q_sentences_train) + len(Q_sentences_train_2):]

            A_padded_sentences = self._pad_sentences(A_sentences_train + A_sentences_train_2 + A_sentences_test)
            A_sentences_train_padded = A_padded_sentences[:len(A_sentences_train) + len(A_sentences_train_2)]
            A_sentences_test_padded = A_padded_sentences[len(A_sentences_train) + len(A_sentences_train_2):]

            all_sentences = Q_sentences_train_padded + A_sentences_train_padded + Q_sentences_test_padded + A_sentences_test_padded
            vocabulary, vocabulary_inv = self._build_vocab(all_sentences)
            vocabulary_size = len(vocabulary_inv)

            # transform to vocabulary
            Q_sentences_train_voc = [[vocabulary[word] for word in sentence] for sentence in Q_sentences_train_padded]
            A_sentences_train_voc = [[vocabulary[word] for word in sentence] for sentence in A_sentences_train_padded]

            Q_sentences_test_voc = [[vocabulary[word] for word in sentence] for sentence in Q_sentences_test_padded]
            A_sentences_test_voc = [[vocabulary[word] for word in sentence] for sentence in A_sentences_test_padded]

            # Transform to pairwise data
            print("Transform to pairwise data")
            X_train_Q, X_train_A_1, X_train_A_2, y_train, idx_train = \
                self._compute_pairwise_data(Q_sentences_train_voc, A_sentences_train_voc, labels_train + labels_train_2,
                                            idx_train + idx_train_2)

            print(len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))

            X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_test = \
                self._compute_pairwise_data_test(Q_sentences_test_voc, A_sentences_test_voc, idx_test_original)

            print(len([True for x in y_test if x == 1]), len([True for x in y_test if x == 0]))

            # Save info to file
            if save == True:
                np.savez('data/train_data', X_train_Q=X_train_Q, X_train_A_1=X_train_A_1, X_train_A_2=X_train_A_2,
                         y_train=y_train)
                np.savez('data/test_data.npz', X_test_Q=X_test_Q, X_test_A_1=X_test_A_1,
                         X_test_A_2=X_test_A_2,
                         y_test=y_test)
                np.savez('data/info_data.npz', idx_train=idx_train, idx_test=idx_test,
                         idx_test_original=idx_test_original, vocabulary_size=vocabulary_size)

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
        return X_train_Q, X_train_A_1, X_train_A_2, y_train, X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_train, idx_test, idx_test_original, vocabulary_size

    def get_data_separate_sentences_test(self, save=True):
        print("Load data pairwise")

        # Try loading from disk
        if os.path.isfile('data/train_data_test.npz') and os.path.isfile('data/test_data_test.npz') \
                and os.path.isfile('data/info_data_test.npz'):
            print("Load data from cached files")
            file_train = np.load('data/train_data.npz')
            X_train_Q = file_train['X_train_Q']
            X_train_A_1 = file_train['X_train_A_1']
            X_train_A_2 = file_train['X_train_A_2']
            y_train = file_train['y_train']

            file_test = np.load('data/test_data.npz')
            X_test_Q = file_test['X_test_Q']
            X_test_A_1 = file_test['X_test_A_1']
            X_test_A_2 = file_test['X_test_A_2']
            y_test = file_test['y_test']

            file_info = np.load('data/info_data.npz')
            idx_train = list(file_info['idx_train'])
            idx_test = list(file_info['idx_test'])
            idx_test_original = list(file_info['idx_test_original'])
            vocabulary_size = int(file_info['vocabulary_size'])
        else:
            print("Compute data and save to files")
            Q_sentences_train, A_sentences_train, \
            labels_train, idx_train = self._read_raw_xml_data_separate_sentences(self.data_path_train_1)
            Q_sentences_train_2, A_sentences_train_2, \
            labels_train_2, idx_train_2 = self._read_raw_xml_data_separate_sentences(self.data_path_train_2)
            Q_sentences_valid, A_sentences_valid, \
            labels_valid, idx_valid = self._read_raw_xml_data_separate_sentences(self.data_path_validation)
            Q_sentences_test, A_sentences_test, \
            labels_test, idx_test_original = self._read_raw_xml_data_separate_sentences(self.data_path_test)

            # print("compute pairwise data")
            # X_Q_new, X_A_1_new, X_A_2_new, labels_new = self._compute_pairwise_data(Q_sentences_train[:20], A_sentences_train[:20], labels_train[:20], idx_train[:20])
            # print([item for item in zip(X_Q_new, X_A_1_new, X_A_2_new)])

            Q_padded_sentences = self._pad_sentences(
                Q_sentences_train + Q_sentences_train_2 + Q_sentences_valid + Q_sentences_test)
            Q_sentences_train_padded = Q_padded_sentences[
                                       :(len(Q_sentences_train) + len(Q_sentences_train_2) + len(Q_sentences_valid))]
            Q_sentences_test_padded = Q_padded_sentences[
                                      (len(Q_sentences_train) + len(Q_sentences_train_2) + len(Q_sentences_valid)):]

            A_padded_sentences = self._pad_sentences(
                A_sentences_train + A_sentences_train_2 + A_sentences_valid + A_sentences_test)
            A_sentences_train_padded = A_padded_sentences[
                                       :(len(A_sentences_train) + len(A_sentences_train_2) + len(A_sentences_valid))]
            A_sentences_test_padded = A_padded_sentences[
                                      (len(A_sentences_train) + len(A_sentences_train_2) + len(A_sentences_valid)):]

            all_sentences = Q_sentences_train_padded + A_sentences_train_padded + Q_sentences_test_padded + A_sentences_test_padded
            vocabulary, vocabulary_inv = self._build_vocab(all_sentences)
            vocabulary_size = len(vocabulary_inv)

            # transform to vocabulary
            Q_sentences_train_voc = [[vocabulary[word] for word in sentence] for sentence in Q_sentences_train_padded]
            A_sentences_train_voc = [[vocabulary[word] for word in sentence] for sentence in A_sentences_train_padded]

            Q_sentences_test_voc = [[vocabulary[word] for word in sentence] for sentence in Q_sentences_test_padded]
            A_sentences_test_voc = [[vocabulary[word] for word in sentence] for sentence in A_sentences_test_padded]

            # Transform to pairwise data
            print("Transform to pairwise data")
            X_train_Q, X_train_A_1, X_train_A_2, y_train, idx_train = \
                self._compute_pairwise_data(Q_sentences_train_voc, A_sentences_train_voc, labels_train + labels_train_2 + labels_valid,
                                            idx_train + idx_train_2 + idx_valid)

            print(len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))

            X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_test = \
                self._compute_pairwise_data_test(Q_sentences_test_voc, A_sentences_test_voc, idx_test_original)

            print(len([True for x in y_test if x == 1]), len([True for x in y_test if x == 0]))

            # Save info to file
            if save == True:
                np.savez('data/train_data_test', X_train_Q=X_train_Q, X_train_A_1=X_train_A_1, X_train_A_2=X_train_A_2,
                         y_train=y_train)
                np.savez('data/test_data_test.npz', X_test_Q=X_test_Q, X_test_A_1=X_test_A_1,
                         X_test_A_2=X_test_A_2,
                         y_test=y_test)
                np.savez('data/info_data_test.npz', idx_train=idx_train, idx_test=idx_test,
                         idx_test_original=idx_test_original, vocabulary_size=vocabulary_size)

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
        return X_train_Q, X_train_A_1, X_train_A_2, y_train, X_test_Q, X_test_A_1, X_test_A_2, y_test, idx_train, idx_test, idx_test_original, vocabulary_size

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
                    labels_new.append(1)
                    new_index = dict(index)
                    new_index['a_id_2'] = idx[c_id]['a_id']
                    new_index['label'] = 1
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
            question_sentence = []
            for rel in thread:
                if rel.tag == "RelQuestion":
                    question_info['q_id'] = rel.attrib['RELQ_ID']
                    question_sentence.extend(self.clean_str(rel.attrib['RELQ_CATEGORY']).split(" "))
                    question_sentence.extend(self.clean_str(str(rel[0].text)).split(" "))
                    question_sentence.extend(self.clean_str(str(rel[1].text)).split(" "))
                elif rel.tag == "RelComment":
                    question_answer_info = dict(question_info)
                    question_answer_info['a_id'] = rel.attrib['RELC_ID']
                    answer_sentence = self.clean_str(str(rel[0].text)).split(" ")
                    Q_sentences.append(question_sentence)
                    A_sentences.append(answer_sentence)
                    answer_label = label_to_int[rel.attrib['RELC_RELEVANCE2RELQ'].lower()]
                    labels.append(answer_label)
                    idx.append(question_answer_info)
            #Q_sentences.append(question_sentence)
            #A_sentences.append(answer_sentences)
            #labels.append(answer_labels)

        return Q_sentences, A_sentences, labels, idx