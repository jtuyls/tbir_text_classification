

import os
import numpy as np

def write_predictions_to_file(pred, conf_scores, validation_ids, filename):
    if os.path.exists(filename):
        clean_file(filename)
    line = "{} \t {} \t 0 \t {} \t {} \n"
    true_array = np.array([0., 1.])
    for i, elem in enumerate(validation_ids):
        write_line(filename,
                        line.format(elem['q_id'], elem['a_id'],
                                    conf_scores[i] if np.array_equal(pred[i], true_array) else (1 - conf_scores[i]),
                                    "true" if np.array_equal(pred[i], true_array) else "false"))


def write_predictions_to_file_unsupervised(conf_scores, validation_ids, filename):
    if os.path.exists(filename):
        clean_file(filename)
    line = "{} \t {} \t 0 \t {} \t {} \n"
    for i, elem in enumerate(validation_ids):
        write_line(filename,
                        line.format(elem['q_id'], elem['a_id'], conf_scores[i], "true"))


def write_line(filename, line):
    f = open(filename, 'a')
    f.write(line)
    f.close()


def clean_file(filename):
    f = open(filename, 'w')
    f.close()
