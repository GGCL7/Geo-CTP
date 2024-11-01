
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one

import torch
from transformers import AutoTokenizer, AutoModel

zscale = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30],
    'C': [0.84, -1.67, 3.71, 0.18, -2.65],
    'D': [3.98, 0.93, 1.93, -2.46, 0.75],
    'E': [3.11, 0.26, -0.11, -0.34, -0.25],
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62],
    'G': [2.05, -4.06, 0.36, -0.82, -0.38],
    'H': [2.47, 1.95, 0.26, 3.90, 0.09],
    'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
    'K': [2.29, 0.89, -2.49, 1.49, 0.31],
    'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98],
    'N': [3.05, 1.62, 1.04, -1.15, 1.61],
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],
    'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
    'R': [3.52, 2.50, -3.50, 1.99, -0.17],
    'S': [2.39, -1.07, 1.15, -1.39, 0.67],
    'T': [0.75, -2.18, -1.12, -1.46, -0.40],
    'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59],
    'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
    '-': [0.00, 0.00, 0.00, 0.00, 0.00]
}

kyte_doolittle_hydrophobicity = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    '-': 0.0
}

isoelectric_point = {
    'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48,
    'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98,
    'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76,
    'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.66,
    '-': 0.00
}

molecular_weight = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2,
    '-': 0.0
}

blosum62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
    '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}



def create_node_features(residues):
    node_features = []
    for residue in residues:
        aa = three_to_one(residue.get_resname())
        z_scale_features = zscale.get(aa, zscale['-'])
        hydrophobicity_feature = [kyte_doolittle_hydrophobicity.get(aa, 0.0)]
        isoelectric_point_feature = [isoelectric_point.get(aa, 0.00)]
        molecular_weight_feature = [molecular_weight.get(aa, 0.0)]
        blosum62_features = blosum62.get(aa, blosum62['-'])
        features = z_scale_features + hydrophobicity_feature + isoelectric_point_feature + molecular_weight_feature + blosum62_features
        node_features.append(features)

    return np.array(node_features)



def parse_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    chain = list(model.get_chains())[0]
    residues = [residue for residue in chain if is_aa(residue, standard=True)]
    return residues

def get_ca_coordinates(residues):
    ca_coords = []
    for residue in residues:
        ca = residue['CA']
        ca_coords.append(ca.get_coord())
    return np.array(ca_coords)

def calculate_distance_matrix(coords):
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    return dist_matrix

def create_adjacency_matrix(dist_matrix, threshold=8.0):
    adjacency_matrix = (dist_matrix < threshold).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)  # No self-loops
    return adjacency_matrix

def create_edge_features(dist_matrix, adjacency_matrix):
    edge_index = np.array(np.nonzero(adjacency_matrix))
    edge_features = dist_matrix[edge_index[0], edge_index[1]]
    # print("Edge Index:", edge_index.shape)
    # print("Edge Features:", edge_features.shape)
    return edge_index, edge_features



def generate_features_protein_bert(sequences, tokenizer, model):
    features = []
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors='pt')["input_ids"]
        with torch.no_grad():
            hidden_states = model(inputs)[0]
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        features.append(embedding_mean)
    features_np = np.vstack(features)
    # print(f"Generated protein BERT features with shape: {features_np.shape}")
    return features_np



tokenizer = AutoTokenizer.from_pretrained("ESM_Pre_model", trust_remote_code=True)
model = AutoModel.from_pretrained("ESM_Pre_model", trust_remote_code=True)
def process_pdb_and_sequence(pdb_file, sequence, distance_threshold=8.0):
    residues = parse_pdb(pdb_file)
    ca_coords = get_ca_coordinates(residues)
    dist_matrix = calculate_distance_matrix(ca_coords)
    adjacency_matrix = create_adjacency_matrix(dist_matrix, distance_threshold)
    edge_index, edge_features = create_edge_features(dist_matrix, adjacency_matrix)
    node_features = create_node_features(residues)
    sequence_features = generate_features_protein_bert([sequence], tokenizer, model)[0]
    return edge_index, edge_features, node_features, sequence_features
