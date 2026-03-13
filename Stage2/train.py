import argparse
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Protein_Feature import get_available_device, process_pdb_and_sequence
from graph_transformer_layer import ContrastiveLoss, GraphTransformer
from loss_function import BCEFocalLoss
from metrics import evaluate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 multi-label training for Geo-CTP.")
    parser.add_argument("--train-dir", required=True, help="Directory containing training fasta files.")
    parser.add_argument("--test-dir", required=True, help="Directory containing test fasta files.")
    parser.add_argument("--pdb-root", required=True, help="Directory containing PDB files.")
    parser.add_argument("--label-file", required=True, help="Tab-separated sequence labels file.")
    parser.add_argument("--esm-model-path", required=True, help="Path or HF id for the ESM model.")
    parser.add_argument("--esm-device", default=None, help="Device for ESM embedding extraction.")
    parser.add_argument("--train-device", default=None, help="Device for model training.")
    parser.add_argument("--distance-threshold", type=float, default=8.0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="best_model_with_absolute_true.pth")
    parser.add_argument("--disable-layer-norm", action="store_true")
    parser.add_argument("--batch-norm", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_training_device(requested_device):
    if requested_device is not None:
        return torch.device(requested_device)
    return get_available_device()


def load_labels(label_file):
    labels_dict = {}
    with open(label_file, "r") as handle:
        for line in handle:
            seq_id, raw_labels = line.strip().split("\t")
            labels_dict[seq_id] = torch.tensor(list(map(float, raw_labels.split(","))), dtype=torch.float32)
    return labels_dict


def read_sequences_from_directory(directory, labels_dict):
    sequences = {}
    for file_name in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r") as handle:
            sequence_id = ""
            sequence = ""
            for line in handle:
                line = line.strip()
                if line.startswith(">"):
                    if sequence_id and sequence_id not in sequences:
                        sequences[sequence_id] = {
                            "sequence": sequence,
                            "labels": labels_dict.get(sequence_id, torch.zeros(6, dtype=torch.float32)),
                        }
                    sequence_id = line[1:]
                    sequence = ""
                else:
                    sequence += line
            if sequence_id and sequence_id not in sequences:
                sequences[sequence_id] = {
                    "sequence": sequence,
                    "labels": labels_dict.get(sequence_id, torch.zeros(6, dtype=torch.float32)),
                }
    return sequences


def create_data_object(edge_index, edge_features, node_features, sequence_features, labels):
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)

    y = labels.clone().detach().float()
    if y.dim() == 1:
        y = y.unsqueeze(0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.bert_features = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0)
    return data


def create_data_objects(sequence_dict, pdb_root, args, split_name):
    data_list = []
    total_sequences = len(sequence_dict)

    for index, (sequence_id, content) in enumerate(sequence_dict.items(), start=1):
        pdb_file = os.path.join(pdb_root, f"{sequence_id}.pdb")
        if not os.path.exists(pdb_file):
            continue

        edge_index, edge_features, node_features, sequence_features = process_pdb_and_sequence(
            pdb_file,
            content["sequence"],
            distance_threshold=args.distance_threshold,
            esm_model_path=args.esm_model_path,
            esm_device=args.esm_device,
        )
        data_list.append(create_data_object(edge_index, edge_features, node_features, sequence_features, content["labels"]))

        if index % 100 == 0 or index == total_sequences:
            print(f"[{split_name}] preprocessing {index}/{total_sequences}")

    return data_list


def run_epoch(model, loader, optimizer, clf_loss_fn, contrastive_loss_fn, device, contrastive_weight, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(data)
            logits = outputs["logits"]
            structure_proj = outputs["structure_proj"]
            sequence_proj = outputs["sequence_proj"]

            clf_loss = clf_loss_fn(logits, data.y)
            contrastive_loss = contrastive_loss_fn(structure_proj, sequence_proj)
            loss = clf_loss + contrastive_weight * contrastive_loss

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(data.y.detach().cpu().numpy())

    average_loss = total_loss / max(len(loader), 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    macro_f1 = f1_score(all_labels, all_preds > 0.5, average="macro")
    return average_loss, macro_f1, all_preds, all_labels


def main():
    args = parse_args()
    set_seed(args.seed)

    labels_dict = load_labels(args.label_file)
    train_sequences = read_sequences_from_directory(args.train_dir, labels_dict)
    test_sequences = read_sequences_from_directory(args.test_dir, labels_dict)

    train_data = create_data_objects(train_sequences, args.pdb_root, args, "train")
    test_data = create_data_objects(test_sequences, args.pdb_root, args, "test")
    print(f"Loaded train graphs: {len(train_data)} | test graphs: {len(test_data)}")

    if not train_data:
        raise ValueError("No train graphs were created. Check your input paths.")

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
        bert_feature_dim=sample.bert_features.shape[-1],
        num_classes=sample.y.size(-1),
        dropout=args.dropout,
        layer_norm=not args.disable_layer_norm,
        batch_norm=args.batch_norm,
        projection_dim=args.projection_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    clf_loss_fn = BCEFocalLoss(gamma=args.focal_gamma)
    contrastive_loss_fn = ContrastiveLoss(temperature=args.contrastive_temperature)

    best_absolute_true = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            clf_loss_fn,
            contrastive_loss_fn,
            device,
            args.contrastive_weight,
            training=True,
        )
        test_loss, test_f1, test_preds, test_labels = run_epoch(
            model,
            test_loader,
            optimizer,
            clf_loss_fn,
            contrastive_loss_fn,
            device,
            args.contrastive_weight,
            training=False,
        )

        evaluation_results = evaluate_metrics(test_preds, test_labels)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrainLoss: {train_loss:.4f} | TrainF1: {train_f1:.4f} | "
            f"TestLoss: {test_loss:.4f} | TestF1: {test_f1:.4f}"
        )
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")

        if evaluation_results["absolute_true"] > best_absolute_true:
            best_absolute_true = evaluation_results["absolute_true"]
            torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
