import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Protein_Feature import get_available_device, process_pdb_and_sequence
from graph_transformer_layer import ContrastiveLoss, GraphTransformer, SupervisedContrastiveLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 binary training for Geo-CTP.")
    parser.add_argument("--train-fa", required=True, help="Path to train fasta/txt file.")
    parser.add_argument("--test-fa", required=True, help="Path to test fasta/txt file.")
    parser.add_argument("--pdb-root", required=True, help="Directory containing residue-level PDB files.")
    parser.add_argument("--esm-model-path", required=True, help="Path or HF id for the ESM model used for sequence embeddings.")
    parser.add_argument("--esm-device", default=None, help="Device for ESM embedding extraction, e.g. cuda, mps, cpu.")
    parser.add_argument("--train-device", default=None, help="Device for GeoCTP training, e.g. cuda, mps, cpu.")
    parser.add_argument("--distance-threshold", type=float, default=8.0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--weight-seqstr", type=float, default=0.6)
    parser.add_argument("--weight-label", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="best_model.pth")
    parser.add_argument("--disable-layer-norm", action="store_true")
    parser.add_argument("--batch-norm", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_fasta(file_path):
    sequences = {}
    with open(file_path, "r") as handle:
        sequence_id = ""
        sequence = ""
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id:
                    sequences[sequence_id] = sequence
                sequence_id = line[1:]
                sequence = ""
            else:
                sequence += line
        if sequence_id:
            sequences[sequence_id] = sequence
    return sequences


def create_data_object(edge_index, edge_features, node_features, sequence_features, label):
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)
    bert_features = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.bert_features = bert_features
    return data


def create_data_objects(sequence_dict, pdb_root, args, split_name):
    data_list = []
    total_sequences = len(sequence_dict)

    for index, (sequence_id, sequence) in enumerate(sequence_dict.items(), start=1):
        pdb_file = os.path.join(pdb_root, f"{sequence_id}.pdb")
        if not os.path.exists(pdb_file):
            continue

        edge_index, edge_features, node_features, sequence_features = process_pdb_and_sequence(
            pdb_file,
            sequence,
            distance_threshold=args.distance_threshold,
            esm_model_path=args.esm_model_path,
            esm_device=args.esm_device,
        )
        label = 1 if ("DCT" in sequence_id or "pos" in sequence_id) else 0
        data_list.append(create_data_object(edge_index, edge_features, node_features, sequence_features, label))

        if index % 100 == 0 or index == total_sequences:
            print(f"[{split_name}] preprocessing {index}/{total_sequences}")

    return data_list


def get_training_device(requested_device):
    if requested_device is not None:
        return torch.device(requested_device)
    return get_available_device()


def train_one_epoch(model, loader, optimizer, loss_cls, loss_seq, loss_lab, args, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        logits = outputs["logits"]
        structure_proj = outputs["structure_proj"]
        sequence_proj = outputs["sequence_proj"]
        fused_proj = outputs["fused_proj"]
        labels = data.y.view(-1)

        classification_loss = loss_cls(logits, labels)
        seq_struct_loss = loss_seq(structure_proj, sequence_proj)
        label_loss = loss_lab(fused_proj.unsqueeze(1), labels)

        loss = classification_loss + args.weight_seqstr * seq_struct_loss + args.weight_label * label_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)["logits"]
            preds = logits.argmax(dim=1)
            labels = data.y.view(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    train_sequences = read_fasta(args.train_fa)
    test_sequences = read_fasta(args.test_fa)

    train_data = create_data_objects(train_sequences, args.pdb_root, args, "train")
    test_data = create_data_objects(test_sequences, args.pdb_root, args, "test")
    print(f"Loaded train graphs: {len(train_data)} | test graphs: {len(test_data)}")

    if not train_data:
        raise ValueError("No train graphs were created. Check your fasta and PDB paths.")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    sample = train_data[0]
    device = get_training_device(args.train_device)

    model = GraphTransformer(
        in_channels=sample.x.size(1),
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        edge_dim=sample.edge_attr.size(1),
        bert_dim=sample.bert_features.shape[-1],
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        layer_norm=not args.disable_layer_norm,
        batch_norm=args.batch_norm,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_cls = nn.CrossEntropyLoss()
    loss_seq = ContrastiveLoss(args.temperature)
    loss_lab = SupervisedContrastiveLoss(args.temperature)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_cls, loss_seq, loss_lab, args, device)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f} | TestAcc: {test_acc:.4f} | BestAcc: {best_acc:.4f}"
        )


if __name__ == "__main__":
    main()
