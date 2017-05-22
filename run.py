
from main.data_loader import DataLoader
from main.data_loader_pairwise import PairwiseDataLoader
from main.ffnn_network_multiple_input import NetworkMI
from main.ffnn_ranking_network import FFNNRankingNetwork
from main.ffnn_ranking_network_sigmoid import FFNNRankingNetworkSigmoid
from main.ffnn_network_pca import NetworkPCA

from main.keras_rnn import KerasRNN
from main.keras_rnn_ranking import KerasRNNRanking
from main.keras_rnn_pca import KerasRNNPCA

scenario = 3.2

data_loader = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                         'data/test_input.xml')
data_loader_pairwise = PairwiseDataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                                          'data/test_input.xml')

# 1. Run feedforward neural network with multiple input for classification
# Run on test dataset
if scenario == 1.0:
    network = NetworkMI(data_loader=data_loader)
    network.main(batch_size=32, num_epochs=5, test=True)
# Run on validation (dev) dataset
if scenario == 1.1:
    network = NetworkMI(data_loader=data_loader)
    network.main(batch_size=32, num_epochs=50, dropout=0.3, test=False)


# 2. Run feedforward neural network with PCA preprocessor
if scenario == 2.0:
    network = NetworkPCA(data_loader=data_loader)
    network.main(batch_size=32, num_epochs=50)


# 3. Run feedforward neural network with multiple input for learning to rank
# Run network with softmax outputon test dataset
if scenario == 3.0:
    network = FFNNRankingNetwork(data_loader_pairwise=data_loader_pairwise)
    network.main(batch_size=32, num_epochs=1, prediction_filename="scorer/scenario_3_0.pred",
                 test=True, save_data_after_loading=True)
# Run network with softmax output on validation (dev) dataset
if scenario == 3.1:
    network = FFNNRankingNetwork(data_loader_pairwise=data_loader_pairwise)
    network.main(batch_size=32, num_epochs=1, dropout=0.1, loss="cross_entropy", prediction_filename="scorer/scenario_3_1.pred")
# Run network with sigmoid output on validation (dev) dataset
if scenario == 3.2:
    network = FFNNRankingNetworkSigmoid(data_loader_pairwise=data_loader_pairwise)
    network.main(batch_size=32, num_epochs=50, dropout=0.1,
                 loss="cross_entropy", optimizer_name="sgd", learning_rate=0.0001,
                 prediction_filename="scorer/scenario_3_2.pred")
if scenario == 3.3:
    network = FFNNRankingNetworkSigmoid(data_loader_pairwise=data_loader_pairwise)
    network.main(batch_size=32, num_epochs=50, dropout=0.1,
                 loss="hinge", optimizer_name="sgd", learning_rate=0.001,
                 prediction_filename="scorer/scenario_3_3.pred")


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
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=50,
                            prediction_filename="scorer/scenario_6_1.pred")
if scenario == 6.2:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=20,
                           prediction_filename="scorer/scenario_6_2.pred")






