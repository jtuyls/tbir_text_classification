
from data_loader import DataLoader
from network_multiple_input import NetworkMI
from network_pca import NetworkPCA

scenario = 2

data_loader = DataLoader('data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml',
                         'data/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml')

if scenario == 1:
    network = NetworkMI(data_loader=data_loader)
    network.main(batch_size=32, num_epochs=50)

if scenario == 2:
    network = NetworkPCA(data_loader=data_loader)
    network.main(batch_size=32, num_epochs=50)




