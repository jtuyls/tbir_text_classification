import re
import itertools

import numpy as np
import xml.etree.ElementTree as ET

from collections import Counter




class DataLoader(object):

    def __init__(self, data_path_train_1,  data_path_train_2, data_path_validation, data_path_test=None):
        self.data_path_train_1 = data_path_train_1
        self.data_path_train_2 = data_path_train_2
        self.data_path_validation = data_path_validation
        self.data_path_test = data_path_test

    #### GET DATA FOR CLASSIFICATION ####

    def get_data(self):
        print("Load data")
        sentences_train_1, labels_train_1, _ = self._read_raw_xml_data(self.data_path_train_1)
        sentences_train_2, labels_train_2, _ = self._read_raw_xml_data(self.data_path_train_2)
        sentences_valid, labels_valid, validation_idx = self._read_raw_xml_data(self.data_path_validation)

        padded_sentences = self._pad_sentences(sentences_train_1 + sentences_train_2 + sentences_valid)
        padded_sentences_train = padded_sentences[:(len(sentences_train_1)+len(sentences_train_2))]
        padded_sentences_valid = padded_sentences[(len(sentences_train_1)+len(sentences_train_2)):]

        all_sentences = padded_sentences_train + padded_sentences_valid
        vocabulary, vocabulary_inv = self._build_vocab(all_sentences)

        X_train = np.array([[vocabulary[word] for word in sentence] for sentence in padded_sentences_train])
        y_train = np.array(labels_train_1 + labels_train_2)
        print(len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))

        X_valid = np.array([[vocabulary[word] for word in sentence] for sentence in padded_sentences_valid])
        y_valid = np.array(labels_valid)
        print(len([True for x in y_valid if x == 1]), len([True for x in y_valid if x == 0]))

        print("Done loading data")
        return X_train, y_train, X_valid, y_valid, validation_idx

    def get_data_separate_sentences(self):
        print("Load data")
        Q_sentences_train, A_sentences_train, labels_train, _ = self._read_raw_xml_data_separate_sentences(self.data_path_train_1)
        Q_sentences_train_2, A_sentences_train_2, labels_train_2, _ = self._read_raw_xml_data_separate_sentences(
            self.data_path_train_2)
        Q_sentences_valid, A_sentences_valid, labels_valid, validation_idx = self._read_raw_xml_data_separate_sentences(self.data_path_validation)

        Q_padded_sentences = self._pad_sentences(Q_sentences_train + Q_sentences_train_2 + Q_sentences_valid)
        Q_sentences_train_padded = Q_padded_sentences[:len(Q_sentences_train) + len(Q_sentences_train_2)]
        Q_sentences_valid_padded = Q_padded_sentences[len(Q_sentences_train) + len(Q_sentences_train_2):]

        A_padded_sentences = self._pad_sentences(A_sentences_train + A_sentences_train_2 + A_sentences_valid)
        A_sentences_train_padded = A_padded_sentences[:len(A_sentences_train) + len(A_sentences_train_2)]
        A_sentences_valid_padded = A_padded_sentences[len(A_sentences_train) + len(A_sentences_train_2):]

        all_sentences = Q_sentences_train_padded + A_sentences_train_padded + Q_sentences_valid_padded + A_sentences_valid_padded
        vocabulary, vocabulary_inv = self._build_vocab(all_sentences)
        vocabulary_size = len(vocabulary_inv)

        X_train_Q = np.array([[vocabulary[word] for word in sentence] for sentence in Q_sentences_train_padded])
        X_train_A = np.array([[vocabulary[word] for word in sentence] for sentence in A_sentences_train_padded])
        y_train = np.array(labels_train + labels_train_2)
        print(len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))
        #
        X_valid_Q = np.array([[vocabulary[word] for word in sentence] for sentence in Q_sentences_valid_padded])
        X_valid_A = np.array([[vocabulary[word] for word in sentence] for sentence in A_sentences_valid_padded])
        y_valid = np.array(labels_valid)
        print(len([True for x in y_valid if x == 1]), len([True for x in y_valid if x == 0]))

        print("Done loading data")
        return X_train_Q, X_train_A, y_train, X_valid_Q, X_valid_A, y_valid, validation_idx, vocabulary_size

    def get_data_separate_sentences_test(self):
        print("Load data")
        # Load training data
        Q_sentences_train, A_sentences_train, labels_train, _ = self._read_raw_xml_data_separate_sentences(
            self.data_path_train_1)
        Q_sentences_train_2, A_sentences_train_2, labels_train_2, _ = self._read_raw_xml_data_separate_sentences(
            self.data_path_train_2)

        # Load validation data
        Q_sentences_valid, A_sentences_valid, labels_valid, validation_idx = self._read_raw_xml_data_separate_sentences(
            self.data_path_validation)

        # Load test data
        Q_sentences_test, A_sentences_test, labels_test, test_idx = self._read_raw_xml_data_separate_sentences(
            self.data_path_test)

        Q_padded_sentences = self._pad_sentences(Q_sentences_train + Q_sentences_train_2 + Q_sentences_valid + Q_sentences_test)
        Q_sentences_train_padded = Q_padded_sentences[:(len(Q_sentences_train) + len(Q_sentences_train_2) + len(Q_sentences_valid))]
        Q_sentences_test_padded = Q_padded_sentences[(len(Q_sentences_train) + len(Q_sentences_train_2) + len(Q_sentences_valid)):]

        A_padded_sentences = self._pad_sentences(A_sentences_train + A_sentences_train_2 + A_sentences_valid + A_sentences_test)
        A_sentences_train_padded = A_padded_sentences[:(len(A_sentences_train) + len(A_sentences_train_2) + len(A_sentences_valid))]
        A_sentences_test_padded = A_padded_sentences[(len(A_sentences_train) + len(A_sentences_train_2) + len(A_sentences_valid)):]

        all_sentences = Q_sentences_train_padded + A_sentences_train_padded + Q_sentences_test_padded + A_sentences_test_padded
        vocabulary, vocabulary_inv = self._build_vocab(all_sentences)
        vocabulary_size = len(vocabulary_inv)

        X_train_Q = np.array([[vocabulary[word] for word in sentence] for sentence in Q_sentences_train_padded])
        X_train_A = np.array([[vocabulary[word] for word in sentence] for sentence in A_sentences_train_padded])
        y_train = np.array(labels_train + labels_train_2 + labels_valid)
        print(len([True for x in y_train if x == 1]), len([True for x in y_train if x == 0]))
        #
        X_test_Q = np.array([[vocabulary[word] for word in sentence] for sentence in Q_sentences_test_padded])
        X_test_A = np.array([[vocabulary[word] for word in sentence] for sentence in A_sentences_test_padded])

        print("Done loading data")
        return X_train_Q, X_train_A, y_train, X_test_Q, X_test_A, None, test_idx, vocabulary_size

    #### GET DATA FOR PCA ####
    def get_data_for_pca(self):
        print("Loading data")
        question_sentences_tuples, answer_sentences_tuples, answer_labels_train = self._read_raw_xml_data_general(self.data_path_train_1)
        question_sentences_tuples_2, answer_sentences_tuples_2, answer_labels_train_2 = self._read_raw_xml_data_general(
            self.data_path_train_2)
        question_sentences_tuples_valid, answer_sentences_tuples_valid, answer_labels_valid = self._read_raw_xml_data_general(self.data_path_validation)

        question_sentences_tuples = question_sentences_tuples + question_sentences_tuples_2
        answer_sentences_tuples = answer_sentences_tuples + answer_sentences_tuples_2
        answer_labels_train = answer_labels_train + answer_labels_train_2

        question_sentences = [q_tuple[1]['category'] + q_tuple[1]['subject'] + q_tuple[1]['question']
                              for q_tuple in question_sentences_tuples]
        q_idx = [q_tuple[0] for q_tuple in question_sentences_tuples]
        answer_sentences = [a_tuple[1]['answer'] for a_tuple in answer_sentences_tuples]
        a_idx = [a_tuple[0] for a_tuple in answer_sentences_tuples]
        question_sentences_valid = [q_tuple[1]['category'] + q_tuple[1]['subject'] + q_tuple[1]['question']
                              for q_tuple in question_sentences_tuples_valid]
        q_idx_valid = [q_tuple[0] for q_tuple in question_sentences_tuples_valid]
        answer_sentences_valid = [a_tuple[1]['answer'] for a_tuple in answer_sentences_tuples_valid]
        a_idx_valid = [a_tuple[0] for a_tuple in answer_sentences_tuples_valid]

        print(len(question_sentences), len(answer_sentences))
        sentences_train = question_sentences + answer_sentences
        sentences_valid = question_sentences_valid + answer_sentences_valid
        sentences = sentences_train + sentences_valid
        vocabulary, vocabulary_inv = self._build_vocab(sentences)
        vocabulary_size = len(vocabulary_inv)

        print(len(sentences), vocabulary_size)
        data_train = np.zeros((len(sentences_train), vocabulary_size))
        for i, sentence in enumerate(sentences_train):
            for word in sentence:
                data_train[i, vocabulary[word]] = 1


        data_valid = np.zeros((len(sentences_valid), vocabulary_size))
        for i, sentence in enumerate(sentences_valid):
            for word in sentence:
                data_valid[i, vocabulary[word]] = 1

        y_train = np.array(answer_labels_train)
        y_valid = np.array(answer_labels_valid)

        _, _, test_idx = self._read_raw_xml_data(self.data_path_validation)
        print("Done loading data")
        return data_train, data_valid, q_idx, a_idx, q_idx_valid, a_idx_valid, y_train, y_valid, test_idx

    def get_data_for_pca_test(self):
        print("Loading data")
        question_sentences_tuples, answer_sentences_tuples, answer_labels_train = self._read_raw_xml_data_general(
            self.data_path_train_1)
        question_sentences_tuples_2, answer_sentences_tuples_2, answer_labels_train_2 = self._read_raw_xml_data_general(
            self.data_path_train_2)
        question_sentences_tuples_valid, answer_sentences_tuples_valid, answer_labels_valid = self._read_raw_xml_data_general(
            self.data_path_validation)
        question_sentences_tuples_test, answer_sentences_tuples_test, answer_labels_test = self._read_raw_xml_data_general(
            self.data_path_test)

        question_sentences_tuples = question_sentences_tuples + question_sentences_tuples_2 + question_sentences_tuples_valid
        answer_sentences_tuples = answer_sentences_tuples + answer_sentences_tuples_2 + answer_sentences_tuples_valid
        answer_labels_train = answer_labels_train + answer_labels_train_2 + answer_labels_valid

        question_sentences = [q_tuple[1]['category'] + q_tuple[1]['subject'] + q_tuple[1]['question']
                              for q_tuple in question_sentences_tuples]
        q_idx = [q_tuple[0] for q_tuple in question_sentences_tuples]
        answer_sentences = [a_tuple[1]['answer'] for a_tuple in answer_sentences_tuples]
        a_idx = [a_tuple[0] for a_tuple in answer_sentences_tuples]
        question_sentences_test = [q_tuple[1]['category'] + q_tuple[1]['subject'] + q_tuple[1]['question']
                                    for q_tuple in question_sentences_tuples_test]
        q_idx_test = [q_tuple[0] for q_tuple in question_sentences_tuples_test]
        answer_sentences_test = [a_tuple[1]['answer'] for a_tuple in answer_sentences_tuples_test]
        a_idx_test = [a_tuple[0] for a_tuple in answer_sentences_tuples_test]

        print(len(question_sentences), len(answer_sentences))
        sentences_train = question_sentences + answer_sentences
        sentences_test = question_sentences_test + answer_sentences_test
        sentences = sentences_train + sentences_test
        vocabulary, vocabulary_inv = self._build_vocab(sentences)
        vocabulary_size = len(vocabulary_inv)

        print(len(sentences), vocabulary_size)
        data_train = np.zeros((len(sentences_train), vocabulary_size))
        for i, sentence in enumerate(sentences_train):
            for word in sentence:
                data_train[i, vocabulary[word]] = 1

        data_test = np.zeros((len(sentences_test), vocabulary_size))
        for i, sentence in enumerate(sentences_test):
            for word in sentence:
                data_test[i, vocabulary[word]] = 1

        y_train = np.array(answer_labels_train)
        y_valid = np.array(answer_labels_test)

        _, _, test_idx = self._read_raw_xml_data(self.data_path_test)
        print("Done loading data")
        return data_train, data_test, q_idx, a_idx, q_idx_test, a_idx_test, y_train, y_valid, test_idx

    #### HELPER FUNCTIONS ####

    def _read_raw_xml_data_general(self, data_path):
        question_sentences = []
        answer_sentences = []
        answer_labels = []
        tree = ET.parse(data_path)
        xml = tree.getroot()
        for thread in xml:
            question_id = None
            for rel in thread:
                if rel.tag == "RelQuestion":
                    question_id = rel.attrib['RELQ_ID']
                    category = self.clean_str(rel.attrib['RELQ_CATEGORY']).split(" ")
                    subject = self.clean_str(str(rel[0].text)).split(" ")
                    question = self.clean_str(str(rel[1].text)).split(" ")
                    question_info = {
                        'category': category,
                        'subject': subject,
                        'question': question
                    }
                    question_sentences.append((question_id, question_info))
                elif rel.tag == "RelComment":
                    answer_sentence = self.clean_str(str(rel[0].text)).split(" ")
                    answer_info = {
                        'q_id': question_id,
                        'answer': answer_sentence
                    }
                    answer_sentences.append((rel.attrib['RELC_ID'], answer_info))
                    answer_labels.append(1 if rel.attrib['RELC_RELEVANCE2RELQ'].lower() == 'good' else 0)

        return question_sentences, answer_sentences, answer_labels

    def _read_raw_xml_data(self, data_path):
        ids = []
        sentences = []
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
                    sentences.append(question_sentence + answer_sentence)
                    labels.append(1 if rel.attrib['RELC_RELEVANCE2RELQ'].lower() == 'good' else 0)
                    ids.append(question_answer_info)

        return sentences, labels, ids

    def _read_raw_xml_data_separate_sentences(self, data_path):
        ids = []
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
                    labels.append(1 if rel.attrib['RELC_RELEVANCE2RELQ'].lower() == 'good' else 0)
                    ids.append(question_answer_info)

        return Q_sentences, A_sentences, labels, ids

    def _pad_sentences(self, sentences, padding_word="</s>"):
        """
        Copied from http://mxnet.io/tutorials/nlp/cnn.html
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = max(len(x) for x in sentences)
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences

    def _build_vocab(self, sentences):
        """
        Copied from http://mxnet.io/tutorials/nlp/cnn.html
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def clean_str(self, string, TREC=False):
        """
        Copied from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip() if TREC else string.strip().lower()


