import os
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
import warnings
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from Protein_Feature import *
from graph_transformer_layer import *
warnings.simplefilter('ignore', BiopythonWarning)
import random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


train_fa = 'train.txt'
test_fa = 'test.txt'
train_fp = 'Train_PDB'
test_fp = 'Test_PDB'

num_layers = 2
num_heads = 8
hidden_channels = 128
out_channels = 64
dropout = 0.2
layer_norm = True
batch_norm = True


temperature = 0.2
weight_seqstr = 0.6
weight_label = 0.4
projection_dim = 128


batch_size = 32
epochs = 200
learning_rate = 1e-4
seed = 42


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        sequence_id = ''
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id:
                    sequences[sequence_id] = sequence
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line
        if sequence_id:
            sequences[sequence_id] = sequence
    return sequences

def create_data_object(edge_index, edge_features, node_features, sequence_features, label):
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)
    bert_features = torch.tensor(sequence_features, dtype=torch.float).unsqueeze(0)
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.bert_features = bert_features
    return data




def create_data_objects(sequence_dict, folder_path, distance_threshold=8.0):
    data_list = []
    for sequence_id, sequence in sequence_dict.items():
        pdb_file = os.path.join(folder_path, sequence_id + '.pdb')
        edge_index, edge_features, node_features, sequence_features = process_pdb_and_sequence(pdb_file, sequence, distance_threshold)
        label = 1 if ('DCT' in sequence_id or 'pos' in sequence_id) else 0
        data = create_data_object(edge_index, edge_features, node_features, sequence_features, label)
        data_list.append(data)
    return data_list


def train_one_epoch(model, loader, optim, loss_cls, loss_seq, loss_lab, device):
    model.train()
    total = 0
    for data in loader:
        data = data.to(device)
        optim.zero_grad()
        logits, x1, bert_f, z = model(data)
        y = data.y.view(-1).to(device)
        l_cls = loss_cls(logits, y)
        l_seq = loss_seq(x1, bert_f)
        l_lab = loss_lab(z.unsqueeze(1), y)
        loss  = l_cls + weight_seqstr*l_seq + weight_label*l_lab
        loss.backward()
        optim.step()
        total += loss.item()
    return total/len(loader)

def evaluate(model, loader, device):
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits, *_ = model(data)
            preds = logits.argmax(dim=1)
            corr += (preds == data.y.view(-1).to(device)).sum().item()
            tot  += data.y.size(0)
    return corr/tot



if __name__ == '__main__':

    train_dict = read_fasta(train_fa)
    test_dict  = read_fasta(test_fa)
    train_list = create_data_objects(train_dict, train_fp)
    test_list  = create_data_objects(test_dict,  test_fp)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_list,  batch_size=batch_size, shuffle=False)


    sample = train_list[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    bert_dim = 480

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = GraphTransformer(
        in_channels     = node_dim,
        hidden_channels = hidden_channels,
        out_channels    = out_channels,
        num_layers      = num_layers,
        num_heads       = num_heads,
        edge_dim        = edge_dim,
        bert_dim        = bert_dim,
        projection_dim  = projection_dim,
        dropout         = dropout,
        layer_norm      = layer_norm,
        batch_norm      = batch_norm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_cls  = nn.CrossEntropyLoss()
    loss_seq  = ContrastiveLoss(temperature)
    loss_lab  = SupervisedContrastiveLoss(temperature)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer,
                                  loss_cls, loss_seq, loss_lab, device)
        te_acc  = evaluate(model, test_loader, device)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch:03d}/{epochs} | TrainLoss: {tr_loss:.4f} | TestAcc: {te_acc:.4f} | BestAcc: {best_acc:.4f}")

    print(f"{best_acc:.4f}")
