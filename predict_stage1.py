import argparse
import warnings
import torch
from torch_geometric.data import Data
from Code.Stage1.Protein_Feature import process_pdb_and_sequence
from Code.Stage1.graph_transformer_layer import GraphTransformer
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)


def predict_ctp(model, sequence, pdb_file, distance_threshold=8.0):

    edge_index, edge_features, node_features, sequence_features = process_pdb_and_sequence(
        pdb_file, sequence, distance_threshold
    )


    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)
    bert_features = torch.tensor(sequence_features, dtype=torch.float).unsqueeze(0)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.bert_features = bert_features


    model.eval()
    with torch.no_grad():
        output, _, _ = model(data)
        probs = torch.softmax(output, dim=1).cpu().numpy()
        pred_class = output.argmax(dim=1).item()

    return pred_class, probs


def main():

    parser = argparse.ArgumentParser(description="Predict if a sequence is CTP")
    parser.add_argument("-i", "--input_sequence", required=True, help="Input peptide sequence")
    parser.add_argument("-p", "--pdb_file", required=True, help="Path to the corresponding PDB file")
    args = parser.parse_args()


    model = GraphTransformer(
        in_channels=28,
        hidden_channels=128,
        out_channels=64,
        num_layers=2,
        num_heads=4,
        edge_dim=1,
        bert_feature_dim=480,
        dropout=0.3,
        layer_norm=True,
        batch_norm=True
    )
    model.load_state_dict(torch.load('model_stage1.pth'))


    predicted_class, probabilities = predict_ctp(model, args.input_sequence, args.pdb_file)


    print(f"Predicted Class: {'CTP' if predicted_class == 1 else 'Non-CTP'}")



if __name__ == "__main__":
    main()



