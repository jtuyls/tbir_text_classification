
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

from sklearn.decomposition import PCA

from output_file_writer import write_predictions_to_file_unsupervised


class PCA_analysis(object):

    def __init__(self, data_loader):
        self.data_loader = data_loader

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


    def run(self):
        data_train, data_valid, q_idx_train, a_idx_train, q_idx_valid, a_idx_valid = self.data_loader.get_data_for_pca()
        print(data_train)

        pca = PCA(n_components=2)
        print("Fit PCA")
        X_r = pca.fit(data_train).transform(data_train)
        print("Done fitting PCA")

        #self.visualize_pca(X_r=X_r, idx=q_idx_train+a_idx_train)

        X_v = pca.transform(data_valid)

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
                    conf_scores.append(norm)

        #print(idx)

        write_predictions_to_file_unsupervised(conf_scores, idx, 'scorer/test_pca.pred')







