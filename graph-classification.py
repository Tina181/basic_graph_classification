# Install required packages.
import os
import torch
import logging
import datetime
import argparse
from torch_geometric.loader import DataLoader
from model.GCN import GCN

# code reference : https://blog.csdn.net/dream_of_grass/article/details/135566707

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PROTEINS', help='PROTEINS, DD, MUTAG, IMDB-BINARY, IMDB-MULTI, COLLAB')

# parse parameters
args = parser.parse_args()

import torch
from torch_geometric.datasets import TUDataset



# define basic parameters
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = './logs'
dataset_name = args.dataset
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=os.path.join(log_dir, f'{dataset_name}.log'), filemode='w')
print(f'logging to {os.path.join(log_dir, f"{dataset_name}.log")}')
dataset = TUDataset(root='data/TUDataset', name=dataset_name)
# - root (str) – Root directory where the dataset should be saved.（保存的路径）
# - name (str) – The name of the dataset.（名字）

# STEP 1: Load and preprocess the dataset NOTE
# show basic info about the dataset
print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

logging.info('====================show dataset=======================')
logging.info(f'dataset - Dataset: {dataset}:')
logging.info(f'len(dataset) - Number of graphs: {len(dataset)}')
logging.info(f'dataset.num_features - Number of features: {dataset.num_features}')
logging.info(f'dataset.num_classes - Number of classes: {dataset.num_classes}')

# Gather some statistics about the first graph.
data = dataset[0]  # Get the first graph object.
print()
print(data)
print('=============================================================')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
logging.info('====================show one graph============================')
logging.info(f'dataset[0].num_nodes - Number of nodes: {data.num_nodes}')
logging.info(f'dataset[0].num_edges - Number of edges: {data.num_edges}')
logging.info(f'dataset[0].num_edges / dataset[0].num_nodes - Average node degree: {data.num_edges / data.num_nodes:.2f}')
logging.info(f'dataset[0].has_isolated_nodes() - Has isolated nodes: {data.has_isolated_nodes()}')
logging.info(f'dataset[0].has_self_loops() - Has self-loops: {data.has_self_loops()}')
logging.info(f'dataset[0].is_undirected() - Is undirected: {data.is_undirected()}')


# split the dataset
torch.manual_seed(12345)
dataset = dataset.shuffle()
total_size =  len(dataset)
train_size = int(total_size * 0.7)
test_size = total_size*0.3
train_dataset = dataset[:train_size]   # 70% for training
test_dataset = dataset[-test_size:]   # 30% for testing
logging.info('====================show split dataset=====================')
logging.info(f'len(train_dataset) - Number of training graphs: {len(train_dataset)}')
logging.info(f'len(test_dataset) - Number of test graphs: {len(test_dataset)}')

# create data loaders: turning many small graphs into a big graph (get mini-batch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
data = next(iter(train_loader))
logging.info('====================show data loader======================')
logging.info(f'train_loader.dataset - train_loader.dataset: {train_loader.dataset}')
logging.info(f'data= next(iter(train_loader)) - data in dataloader: \n {data}')
logging.info(f'len(train_loader) - batch num for train_loader: {len(train_loader)}')
logging.info(f'len(test_loader) - batch num for test_loader: {len(test_loader)}')

# STEP 2: define the model, loss function, and optimizer NOTE
model = GCN(dataset)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
logging.info('====================show model, loss, optim================')
logging.info(f'model: {model}')
logging.info(f'loss function: {criterion}')
logging.info(f'optimizer: {optimizer}')

# STEP 3: define the train process NOTE
def train(train_loader):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# STEP 4: define the test process NOTE
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc = correct / len(loader.dataset)
    return acc

# STEP 5: main
def main():
    for epoch in range(1, 5):
        train_loss = train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    logging.info('==============training and testing ================')
    logging.info(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()
    
    

        