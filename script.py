
import os
import random

from data_loader import DataLoader

def write_predictions_to_file(pred, conf_scores, validation_ids, filename):
    if os.path.exists(filename):
        clean_file(filename)
    line = "{} \t {} \t 0 \t {} \t {} \n"
    for i, elem in enumerate(validation_ids):
        if i < len(conf_scores):
            write_line(filename, line.format(elem['q_id'], elem['a_id'], conf_scores[i] , pred[i]))
        else:
            value = random.uniform(0.0, 1.0)
            write_line(filename, line.format(elem['q_id'], elem['a_id'], value, "true" if value > 0.5 else "false"))


def write_line(filename, line):
    f = open(filename, 'a')
    f.write(line)
    f.close()


def clean_file(filename):
    f = open(filename, 'w')
    f.close()

d = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
               'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
               'data/test_input.xml')

test_ids = d.get_test_ids()



conf_scores = []
classes = []
with open('scorer/test.pred', 'r') as f:
    for line in f:
        conf_scores.append(float(line.split(" ")[6]))
        classes.append(line.split(" ")[8])

write_predictions_to_file(classes, conf_scores, test_ids, "scorer/supervised_pred_jorn_tuyls.text")


