
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

from sklearn.decomposition import PCA

from data_loader import DataLoader

from output_file_writer import write_predictions_to_file


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

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 20
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

d = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml')

data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid, \
y_train, y_test = d.get_data_for_pca()

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
print('X_valid_Q = {}'.format(X_test_Q.shape))
print('X_valid_A = {}'.format(X_test_A.shape))
print('y_valid.shape = {}'.format(y_test.shape))
#print('vocabulary size: {}'.format(vocab_size))
print('Build model...')

input_shape_Q = X_train_Q.shape[1]
print('input_shape_Q: {}'.format(input_shape_Q))
input_shape_A = X_train_A.shape[1]
print('input_shape_A: {}'.format(input_shape_A))

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


# ADJUSTED NETWORK
EMBED_HIDDEN_SIZE = 50
RNN_SIZE = 50
BATCH_SIZE = 32

question = layers.Input(shape=(input_shape_Q,), dtype='int32')
#encoded_question = layers.Embedding(20, EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.RepeatVector(input_shape_Q)(question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
#encoded_question = RNN(RNN_SIZE)(encoded_question)
#encoded_question = layers.RepeatVector(RNN_SIZE)(encoded_question)
print(encoded_question)

answer = layers.Input(shape=(input_shape_A,), dtype='int32')
#encoded_answer = layers.Embedding(20, EMBED_HIDDEN_SIZE)(answer)
encoded_answer = layers.RepeatVector(input_shape_A)(answer)
encoded_answer = RNN(EMBED_HIDDEN_SIZE)(encoded_answer)
encoded_answer = layers.Dropout(0.3)(encoded_answer)
print(encoded_answer)

merged = layers.add([encoded_question, encoded_answer])
merged = RNN(RNN_SIZE)(merged)
print(merged)
merged = layers.Dropout(0.3)(merged)
print(merged)
preds = layers.Dense(2, activation='softmax')(merged)

#optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
#optimizer = keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = Model([question, answer], preds)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([X_train_Q, X_train_A], y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)

loss, acc = model.evaluate([X_test_Q, X_test_A], y_test,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

# PREDICTIONS
pred_valid = model.predict([X_test_Q, X_test_A], batch_size=BATCH_SIZE)

confidence_scores = np.amax(pred_valid, axis=1)
predictions = np.round(pred_valid)

print(pred_valid)
print(confidence_scores)
print(predictions)

validation_ids = d.get_validation_ids()

write_predictions_to_file(predictions, confidence_scores, validation_ids, "scorer/test.pred")
