

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score
)
from Protein_Feature import process_pdb_and_sequence
from graph_transformer_layer import GraphTransformer


model_path   = "best_model.pth"



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


def read_fasta(path):
    seqs = {}
    with open(path, 'r') as f:
        header, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    seqs[header] = ''.join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            seqs[header] = ''.join(seq)
    return seqs


def create_data_object(edge_index, edge_features, node_features, seq_features, label):
    import torch
    from torch_geometric.data import Data
    x = torch.tensor(node_features, dtype=torch.float)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ea = torch.tensor(edge_features, dtype=torch.float)
    if ea.dim() == 1:
        ea = ea.unsqueeze(1)
    bert_f = torch.tensor(seq_features, dtype=torch.float).unsqueeze(0)
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
    data.bert_features = bert_f
    return data

def create_data_list(fasta_dict, pdb_folder, threshold=8.0):
    lst = []
    for hid, seq in fasta_dict.items():
        pdb_file = os.path.join(pdb_folder, f"{hid}.pdb")
        ei, ef, nf, sf = process_pdb_and_sequence(pdb_file, seq, threshold)
        lbl = 1 if ('DCT' in hid or 'Pos' in hid) else 0
        lst.append(create_data_object(ei, ef, nf, sf, lbl))
    return lst


if __name__ == "__main__":

    test_dict = read_fasta(test_fa)
    test_list = create_data_list(test_dict, test_fp)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)


    sample   = test_list[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    bert_dim = sample.bert_features.shape[-1]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits, *_ = model(data)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels= data.y.view(-1).cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())


    acc  = accuracy_score(y_true, y_pred)
    sen  = recall_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe  = tn / (tn + fp)
    mcc  = matthews_corrcoef(y_true, y_pred)
    try:
        auc  = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    print(f"Accuracy      (ACC): {acc:.4f}")
    print(f"Sensitivity   (SEN): {sen:.4f}")
    print(f"Specificity   (SPE): {spe:.4f}")
    print(f"Matthews Corr (MCC): {mcc:.4f}")
    print(f"AUC               : {auc:.4f}")
