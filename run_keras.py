
#### FILE FOR RUNNING ON GPU MACHINE WITHOUT TENSORFLOW ####

from main.keras_rnn import KerasRNN
from main.keras_rnn_ranking import KerasRNNRanking
from main.keras_rnn_pca import KerasRNNPCA

scenario = 6.0

data_loader = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                         'data/test_input.xml')
data_loader_pairwise = PairwiseDataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                                          'data/test_input.xml')

# 6. Run keras recursive neural network
# Run on test dataset
if scenario == 6.0:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=1, test=True,
                            prediction_filename="scorer/scenario_6_0.pred")
# Run on validation (dev) dataset
if scenario == 6.1:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=50,
                            prediction_filename="scorer/scenario_6_1.pred")
if scenario == 6.2:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=20,
                           prediction_filename="scorer/scenario_6_2.pred")