#### FILE FOR RUNNING ON GPU MACHINE WITHOUT TENSORFLOW ####
from main.data_loader import DataLoader
from main.data_loader_pairwise import PairwiseDataLoader
from main.data_loader_word_embeddings import DataLoaderWordEmbeddings
from main.keras_rnn import KerasRNN
from main.keras_rnn_ranking import KerasRNNRanking
from main.keras_rnn_pca import KerasRNNPCA
from main.keras_ranking_conv_network import KerasRankingConv

scenario = 7.2

data_loader = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                         'data/test_input.xml')
data_loader_pairwise = PairwiseDataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                                          'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                                          'data/test_input.xml')
data_loader_word_embeddings = DataLoaderWordEmbeddings('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
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
    keras_rnn.main(embed_hidden_size=50, rnn_size=100, batch_size=32, num_epochs=40,
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
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=128, num_epochs=20, test=True,
                           prediction_filename="scorer/scenario_6_0.pred",
                           save_data_after_loading=False)
if scenario == 6.01:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=128, num_epochs=20, test=True,
                           prediction_filename="scorer/scenario_6_0_0.pred",
                           save_data_after_loading=False)
if scenario == 6.02:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=128, num_epochs=10, test=True,
                           prediction_filename="scorer/scenario_6_0_2.pred",
                           save_data_after_loading=False)
# Run on validation (dev) dataset
if scenario == 6.1:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=20, rnn_size=50, batch_size=128, num_epochs=20,
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
            keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=256, num_epochs=20,
                                   prediction_filename="scorer/scenario_6_4.pred",
                                   validation_split=0.1, save_data_after_loading=False)
if scenario == 6.5:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=32, num_epochs=20,
                           prediction_filename="scorer/scenario_6_5.pred",
                           validation_split=0.1, save_data_after_loading=False)
if scenario == 6.6:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=128, num_epochs=20,
                           prediction_filename="scorer/scenario_6_6.pred",
                           validation_split=0.1, save_data_after_loading=False)
if scenario == 6.7:
    keras_rnn_ranking = KerasRNNRanking(data_loader_pairwise=data_loader_pairwise)
    keras_rnn_ranking.main(embed_hidden_size=50, rnn_size=50, batch_size=64, num_epochs=10,
                           prediction_filename="scorer/scenario_6_7.pred",
                           validation_split=0.1, save_data_after_loading=False)

# 7. Run conv neural network with word embedding
# Run on test dataset
if scenario == 7.0:
    keras_ranking_cnn = KerasRankingConv(data_loader_word_embeddings)
    keras_ranking_cnn.main(batch_size=128, num_epochs=20,
                           dropout=0.1, learning_rate=0.0001,
                           prediction_filename="scorer/scenario_7_0.pred", test=True,
                           save_data_after_loading=False)
if scenario == 7.1:
    keras_ranking_cnn = KerasRankingConv(data_loader_word_embeddings)
    keras_ranking_cnn.main(batch_size=128, num_epochs=10,
                           dropout=0.1, learning_rate=0.0001,
                           prediction_filename="scorer/scenario_7_1.pred",
                           save_data_after_loading=False)

if scenario == 7.2:
    keras_ranking_cnn = KerasRankingConv(data_loader_word_embeddings)
    keras_ranking_cnn.main(batch_size=128, num_epochs=20,
                           dropout=0.2, learning_rate=0.0001,
                           prediction_filename="scorer/scenario_7_2.pred",
                           save_data_after_loading=False)
    

