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
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
import torch.nn as nn

warnings.simplefilter('ignore', BiopythonWarning)



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


train_sequences = read_fasta('train.txt')
test_sequences = read_fasta('test.txt')



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
        edge_index, edge_features, node_features, sequence_features = process_pdb_and_sequence(pdb_file, sequence,
                                                                                               distance_threshold)
        label = 1 if 'DCT' in sequence_id else 0
        data = create_data_object(edge_index, edge_features, node_features, sequence_features, label)
        data_list.append(data)
    return data_list


train_folder = 'train'
test_folder = 'test'

train_data_list = create_data_objects(train_sequences, train_folder)
test_data_list = create_data_objects(test_sequences, test_folder)

train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)


if len(train_data_list) > 0:
    sample_data = train_data_list[0]
    node_features_shape = sample_data.x.shape[1]
    edge_features_shape = sample_data.edge_attr.shape[1]
    bert_feature_dim = sample_data.bert_features.shape[1]
else:
    raise ValueError("The training set is empty, so the model cannot be defined.")

model = GraphTransformer(in_channels=node_features_shape, hidden_channels=128, out_channels=64, num_layers=2,
                         num_heads=4, edge_dim=edge_features_shape, bert_feature_dim=bert_feature_dim, dropout=0.3,
                         layer_norm=True, batch_norm=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
clf_loss_fn = nn.CrossEntropyLoss()
contrastive_loss_fn = ContrastiveLoss(temperature=0.1)

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


def calculate_metrics(all_labels, all_preds, all_probs=None):
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(all_labels, all_preds)


    if all_probs is not None:
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = None

    return accuracy, sensitivity, specificity, mcc, auc
def evaluate(model, loader, clf_loss_fn, contrastive_loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            output, x1, bert_features = model(data)

            clf_loss = clf_loss_fn(output, data.y)
            contrastive_loss = contrastive_loss_fn(x1, bert_features)
            loss = clf_loss + contrastive_loss
            total_loss += loss.item()

            preds = output.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(output, dim=1).cpu().numpy()  # 获取类别概率
            labels = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)


    accuracy, sensitivity, specificity, mcc, auc = calculate_metrics(all_labels, all_preds, all_probs)
    return total_loss / len(loader), accuracy, sensitivity, specificity, mcc, auc



model.load_state_dict(torch.load('best_model_with_contrastive5.pth'))
model.eval()


test_loss, test_accuracy, test_sensitivity, test_specificity, test_mcc, test_auc = evaluate(
    model, test_loader, clf_loss_fn, contrastive_loss_fn)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Sensitivity: {test_sensitivity}')
print(f'Test Specificity: {test_specificity}')
print(f'Test MCC: {test_mcc}')
print(f'Test AUC: {test_auc}')

results = {
    'Test Loss': [round(test_loss, 4)],
    'Test Accuracy': [round(test_accuracy, 4)],
    'Test Sensitivity': [round(test_sensitivity, 4)],
    'Test Specificity': [round(test_specificity, 4)],
    'Test MCC': [round(test_mcc, 4)],
    'Test AUC': [round(test_auc, 4)] if test_auc is not None else [None]
}


df_results = pd.DataFrame(results)
# df_results.to_csv('Results.csv', index=False)
