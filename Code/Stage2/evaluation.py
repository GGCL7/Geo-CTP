
import os
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
import warnings
from Protein_Feature import *
from graph_transformer_layer import *
warnings.simplefilter('ignore', BiopythonWarning)
from loss_function import *
from metrics import *

def load_labels(label_file):
    labels_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            seq_id = parts[0]
            labels = torch.tensor(list(map(float, parts[1].split(','))))
            labels_dict[seq_id] = labels
    return labels_dict


label_file_path = 'multi_label_sequences.txt'
labels_dict = load_labels(label_file_path)


def read_sequences_from_directory(directory):
    sequences = {}
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                sequence_id = ''
                sequence = ''
                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        if sequence_id:
                            if sequence_id not in sequences:
                                sequences[sequence_id] = {
                                    'sequence': sequence,
                                    'labels': labels_dict.get(sequence_id, torch.zeros(6))
                                }
                        sequence_id = line[1:]
                        sequence = ''
                    else:
                        sequence += line
                if sequence_id:
                    if sequence_id not in sequences:
                        sequences[sequence_id] = {
                            'sequence': sequence,
                            'labels': labels_dict.get(sequence_id, torch.zeros(6))
                        }
        else:
            print(f"File {file_path} not found. Skipping...")
    return sequences


def create_data_object(edge_index, edge_features, node_features, sequence_features, labels):
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)
    bert_features = torch.tensor(sequence_features, dtype=torch.float).unsqueeze(0)
    y = labels.clone().detach().float()
    if y.dim() == 1:
        y = y.unsqueeze(0)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.bert_features = bert_features
    return data


def create_data_objects(sequence_dict, pdb_folder, distance_threshold=8.0):
    data_list = []
    for sequence_id, content in sequence_dict.items():
        pdb_file = os.path.join(pdb_folder, sequence_id + '.pdb')
        if os.path.exists(pdb_file):
            edge_index, edge_features, node_features, sequence_features = process_pdb_and_sequence(pdb_file, content['sequence'], distance_threshold)
            labels = content['labels']
            data = create_data_object(edge_index, edge_features, node_features, sequence_features, labels)
            data_list.append(data)
        else:
            print(f"PDB file {pdb_file} not found, skipping...")
    return data_list


train_dir = 'train'
test_dir = 'test'
pdb_folder = 'pos_pdb'


train_sequences = read_sequences_from_directory(train_dir)
test_sequences = read_sequences_from_directory(test_dir)


train_data_list = create_data_objects(train_sequences, pdb_folder)
test_data_list = create_data_objects(test_sequences, pdb_folder)


train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)


if len(train_data_list) > 0:
    sample_data = train_data_list[0]
    node_features_shape = sample_data.x.shape[1]
    edge_features_shape = sample_data.edge_attr.shape[1]
    bert_feature_dim = sample_data.bert_features.shape[1]
else:
    raise ValueError("训练集为空，无法定义模型。")

# 定义模型
model = GraphTransformer(
    in_channels=node_features_shape,
    hidden_channels=128,
    out_channels=64,
    num_layers=2,
    num_heads=4,
    edge_dim=edge_features_shape,
    bert_feature_dim=bert_feature_dim,
    num_classes=6,
    dropout=0.3,
    layer_norm=True,
    batch_norm=True
)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
clf_loss_fn = BCEFocalLoss(gamma=2)
contrastive_loss_fn = ContrastiveLoss(temperature=0.1)

from sklearn.metrics import f1_score
def train(model, loader, optimizer, clf_loss_fn, contrastive_loss_fn):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for data in loader:
        optimizer.zero_grad()
        output, x1, bert_features = model(data)

        clf_loss = clf_loss_fn(output, data.y)
        contrastive_loss = contrastive_loss_fn(x1, bert_features)
        loss = clf_loss + contrastive_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.sigmoid(output).detach().cpu().numpy()
        labels = data.y.cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds > 0.5, average='macro')

    return avg_loss, f1, all_preds, all_labels



def evaluate(model, loader, clf_loss_fn, contrastive_loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            output, x1, bert_features = model(data)

            clf_loss = clf_loss_fn(output, data.y)
            contrastive_loss = contrastive_loss_fn(x1, bert_features)
            loss = clf_loss + contrastive_loss
            total_loss += loss.item()

            preds = torch.sigmoid(output).detach().cpu().numpy()
            labels = data.y.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds > 0.5, average='macro')

    return avg_loss, f1, all_preds, all_labels



model.load_state_dict(torch.load('best_model_with_absolute_true.pth'))
test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader, clf_loss_fn, contrastive_loss_fn)



evaluation_results = evaluate_metrics(test_preds, test_labels)
print("Evaluation Results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")


results_dict = {
    'Test Loss': [round(test_loss, 4)],
    'Test F1 Score': [round(test_f1, 4)],
}


for metric, value in evaluation_results.items():
    results_dict[metric] = [round(value, 4)]
import pandas as pd

df_results = pd.DataFrame(results_dict)


df_results.to_csv('results.csv', index=False)

