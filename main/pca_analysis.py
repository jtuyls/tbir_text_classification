
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

from sklearn.decomposition import PCA

from output_file_writer import write_predictions_to_file_unsupervised


class PCA_analysis(object):

    def __init__(self, data_loader):
        self.data_loader = data_loader


    def run(self, n_components, downsampling=None, test=False):
        if test == False:
            data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid, _, _ = self.data_loader.get_data_for_pca()
        else:
            data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid, _, _ = self.data_loader.get_data_for_pca_test()
        print(data_train)

        if downsampling:
            data_train = data_train[:downsampling]

        # Initialize and fit PCA
        pca = PCA(n_components=n_components)
        print("Fit PCA")
        X_r = pca.fit(data_train).transform(data_train)
        #X_r = data_train
        print("Done fitting PCA")

        #self.visualize_pca(X_r=X_r, idx=q_idx_train+a_idx_train)


        X_v = pca.transform(data_valid)
        #X_v = data_valid
        #print(data_valid.shape)
        #print(q_idx_valid)
        #print(a_idx_valid)

        # calculate confidence scores as norms
        conf_scores = []
        idx = []
        for i, q_id_name in enumerate(q_idx_valid):
            q_x = X_v[i]
            for j, a_id_name in enumerate(a_idx_valid):
                if (a_id_name.split("_")[0] + "_" + a_id_name.split("_")[1]) == q_id_name:
                    idx.append({'q_id': q_id_name, 'a_id': a_id_name})
                    a_x = X_v[len(q_idx_valid)+j]
                    norm = np.linalg.norm(q_x - a_x, 2)
                    conf_scores.append(1/norm)

        # Normalize confidence scores between 0 and 1 (divide by the largest element)
        max_conf_score = max(conf_scores)
        conf_scores = [conf_score/max_conf_score for conf_score in conf_scores]

        write_predictions_to_file_unsupervised(conf_scores, idx, 'scorer/test_pca.pred')

        return X_r, q_idx_train+a_idx_train, X_v, q_idx_valid + a_idx_valid

    def visualize_pca(self, X_r, idx):
        plt.figure("pca analysis", figsize=(16, 16))

        visualization_idx = ["Q1_R1", "Q2_R20"]
        colors = cm.rainbow(np.linspace(0, 1, len(visualization_idx)))
        lw = 2

        for i, id_name in enumerate(idx):
            id_q_name = id_name.split("_")[0] + "_" + id_name.split("_")[1]
            if id_q_name in visualization_idx:
                plt.scatter(X_r[i, 0], X_r[i, 1], color=colors[visualization_idx.index(id_q_name)], alpha=.8, lw=lw)
                plt.annotate(id_name, (X_r[i, 0], X_r[i, 1]))

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA analysis')
        plt.savefig("figures/pca_analysis.png", bbox_inches='tight')
        plt.show()

    def visualize_pca_questions(self, X_r, idx):
        plt.figure("pca analysis", figsize=(16, 16))

        visualization_idx = ["Q1", "Q2"]
        colors = cm.rainbow(np.linspace(0, 1, len(visualization_idx)))
        lw = 2

        for i, id_name in enumerate(idx):
            id_q_name = id_name.split("_")[0]
            if id_q_name in visualization_idx:
                plt.scatter(X_r[i, 0], X_r[i, 1], color=colors[visualization_idx.index(id_q_name)], alpha=.8, lw=lw)
                plt.annotate(id_name, (X_r[i, 0], X_r[i, 1]))

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA analysis')
        plt.savefig("figures/pca_analysis_questions.png", bbox_inches='tight')
        plt.show()







