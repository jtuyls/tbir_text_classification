#### FILE FOR RUNNING ON GPU MACHINE WITHOUT TENSORFLOW ####
from main.data_loader import DataLoader
from main.data_loader_pairwise import PairwiseDataLoader

from main.keras_rnn import KerasRNN
from main.keras_rnn_ranking import KerasRNNRanking
from main.keras_rnn_pca import KerasRNNPCA

scenario = 6.7

data_loader = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                         'data/test_input.xml')
data_loader_pairwise = PairwiseDataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                                          'data/test_input.xml')

# 4. Run keras recursive neural network for classification
# Run on test dataset
if scenario == 4.0:
    keras_rnn = KerasRNN(data_loader=data_loader)
    keras_rnn.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=40, test=True,
                   prediction_filename="scorer/scenario_4_0.pred")
# Run on validation (dev) dataset
if scenario == 4.1:
    keras_rnn = KerasRNN(data_loader=data_loader)
    keras_rnn.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=1,
                   prediction_filename="scorer/scenario_4_1.pred")


# 5. Run keras rnn with PCA preprocessor
if scenario == 5.0:
    keras_rnn_pca = KerasRNNPCA(data_loader=data_loader)
    keras_rnn_pca.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=1,
                       prediction_filename="scorer/scenario_5_0.pred")


# 6. Run keras recursive neural network
# Run on test dataset
if scenario == 6.0:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=1, test=True,
                            prediction_filename="scorer/scenario_6_0.pred")
# Run on validation (dev) dataset
if scenario == 6.1:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=1,
                           prediction_filename="scorer/scenario_6_1.pred",
                           validation_split=0.1, save_data_after_loading=False)
if scenario == 6.2:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=20,
                           prediction_filename="scorer/scenario_6_2.pred",
                           validation_split=0.1, save_data_after_loading=False)
if scenario == 6.3:
        keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
        keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=32, num_epochs=15,
                               prediction_filename="scorer/scenario_6_3.pred",
                               validation_split=0.1, save_data_after_loading=False)
if scenario == 6.4:
            keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
            keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=32, num_epochs=10,
                                   prediction_filename="scorer/scenario_6_4.pred",
                                   validation_split=0.1, save_data_after_loading=False)
if scenario == 6.5:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=32, num_epochs=20,
                           prediction_filename="scorer/scenario_6_5.pred",
                           validation_split=0.1, save_data_after_loading=False)
if scenario == 6.6:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=64, num_epochs=40,
                           prediction_filename="scorer/scenario_6_6.pred",
                           validation_split=0.1, save_data_after_loading=False)
if scenario == 6.7:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=64, num_epochs=10,
                           prediction_filename="scorer/scenario_6_7.pred",
                           validation_split=0.1, save_data_after_loading=False)
