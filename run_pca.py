

from main.pca_analysis import PCA_analysis
from main.data_loader import DataLoader
data_loader = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml',
                         'data/test_input.xml')
pca_a = PCA_analysis(data_loader)

scenario = 3

# Run PCA for visualization
if scenario == 1:
    X_r, idx_r, X_v, idx_v = pca_a.run(n_components=2, downsampling=None)
    pca_a.visualize_pca_questions(X_r=X_r[:100], idx=idx_r[:100])

# Run PCA with higher number of components
if scenario == 2:
    X_r, idx_r, X_v, idx_v = pca_a.run(n_components=16, downsampling=None)

# TEST SCENARIO
if scenario == 3:
    X_r, idx_r, X_v, idx_v = pca_a.run(n_components=18, downsampling=None, test=True)
