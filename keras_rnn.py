
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

from data_loader import DataLoader

from output_file_writer import write_predictions_to_file


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

d = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml')

X_train_Q, X_train_A, y_train, X_valid_Q, X_valid_A, y_valid, vocab_size = d.get_data_separate_sentences()

print(y_train)
y_train = to_categorical(y_train)
print(y_train)
y_valid = to_categorical(y_valid)

print('X_train_Q.shape = {}'.format(X_train_Q.shape))
print('X_train_A.shape = {}'.format(X_train_A.shape))
print('y_train.shape = {}'.format(y_train.shape))
print('X_valid_Q = {}'.format(X_valid_Q.shape))
print('X_valid_A = {}'.format(X_valid_A.shape))
print('y_valid.shape = {}'.format(y_valid.shape))
print('vocabulary size: {}'.format(vocab_size))
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
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = layers.Dropout(0.3)(encoded_question)
#encoded_question = RNN(RNN_SIZE)(encoded_question)
#encoded_question = layers.RepeatVector(RNN_SIZE)(encoded_question)
print(encoded_question)

answer = layers.Input(shape=(input_shape_A,), dtype='int32')
encoded_answer = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(answer)
encoded_answer = layers.Dropout(0.3)(encoded_answer)
encoded_answer = RNN(EMBED_HIDDEN_SIZE)(encoded_answer)
encoded_answer = layers.RepeatVector(input_shape_Q)(encoded_answer)
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
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([X_train_Q, X_train_A], y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)

loss, acc = model.evaluate([X_valid_Q, X_valid_A], y_valid,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

# PREDICTIONS
pred_valid = model.predict([X_valid_Q, X_valid_A], batch_size=BATCH_SIZE)

confidence_scores = np.amax(pred_valid, axis=1)
predictions = np.round(pred_valid)

print(pred_valid)
print(confidence_scores)
print(predictions)

validation_ids = d.get_validation_ids()

write_predictions_to_file(predictions, confidence_scores, validation_ids, "scorer/test.pred")
