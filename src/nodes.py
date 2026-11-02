import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score
import networkx as nx
from pyvis.network import Network
import os


def build_bipartite_graph_and_sync(transactions_df: pd.DataFrame, neo4j_credentials: dict) -> HeteroData:
    # 1. Initialize the Heterogeneous Graph
    data = HeteroData()
    
    # 2. Extract detailed originator and destination mappings (entity extraction)
    accounts = transactions_df['nameOrig'].unique()
    merchants = transactions_df['nameDest'].unique()
    
    # 3. Map string IDs to continuous zero-indexed integers (Required for PyTorch tensors)
    acc_map = {name: i for i, name in enumerate(accounts)}
    merch_map = {name: i for i, name in enumerate(merchants)}
    
    # 4. Create Node Feature Tensors 
    # (Generating dummy 64-dimensional initial features vector matrix for GCNlayer input)
    data['account'].x = torch.randn(len(accounts), 64) 
    data['merchant'].x = torch.randn(len(merchants), 64)
    
    # 5. Construct the Bipartite Edge Index Matrix [2, E] and define edge connection, pair souce w target
    src = [acc_map[acc] for acc in transactions_df['nameOrig']]
    dst = [merch_map[mer] for mer in transactions_df['nameDest']]
    data['account', 'transacts_with', 'merchant'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 6. Open connection to Neo4j AuraDB
    driver = GraphDatabase.driver(
        neo4j_credentials['uri'], 
        auth=(neo4j_credentials['username'], neo4j_credentials['password'])
    )
    
    # Convert dataframe to a list of dictionaries for bulk insertion
    records = transactions_df[['nameOrig', 'nameDest', 'amount']].to_dict('records')
    
    # 7. Execute Cypher using UNWIND for bulk performance, MERGE to avoid duplicates and to avoid network congestion/latency
    cypher_query = """
    UNWIND $batch AS row
    MERGE (a:Account {id: row.nameOrig})
    MERGE (m:Merchant {id: row.nameDest})
    MERGE (a)-[r:TRANSACTED_WITH]->(m)
    SET r.amount = row.amount
    """
    
    driver.execute_query(cypher_query, batch=records, database_="neo4j")
    #close driver connection to avoid memory leaks and connection pool exhaustion
    driver.close()
    
    return data

def extract_temporal_windows(transactions_df: pd.DataFrame) -> torch.Tensor:
    """Extracts sequential transaction windows per account."""
    accounts = transactions_df['nameOrig'].unique()
    batch_size = len(accounts)
    num_features = 4  # amount, frequency, channel type etc.
    seq_len = 10      # looking at the last 10 transactions per account
    
    # For architectural flow, we generate a tensor matching the required CNN input shape.
    # Expected Combined Input Tensor Shape: [B, total_channels, seq_len]
    temporal_seq = torch.randn(batch_size, num_features, seq_len)
    return temporal_seq

# the model hehehehehe
class HybridGCFDFraudDetector(nn.Module):
    """Fuses GCN neighborhood aggregation with CNN temporal extraction."""
    def __init__(self, in_features, gcn_hidden, gnn_out, cnn_channels, seq_len):
        super().__init__()
        
        # 1. Graph Convolutional Layers (for Topology Extraction)
        self.conv1 = SAGEConv(in_features, gcn_hidden)
        self.conv2 = SAGEConv(gcn_hidden, gnn_out)
        
        # 2. Convolutional Neural Network Layers (Sequential/Spatial Filtering)
        total_channels = gnn_out + 4  
        self.conv1d = nn.Conv1d(in_channels=total_channels, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        flattened_dim = cnn_channels * (seq_len // 2)
        
        # 3. Classification Head
        self.fc1 = nn.Linear(flattened_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    #define how the inputs will flow in backpropagation and forward passes 
    def forward(self, x, edge_index, temporal_seq):
        # Pass through GCN layers to get topological embeddings
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        
        node_embeddings = self.conv2(h, edge_index) # Shape: [N, gnn_out]
        
        # Note:  align the GNN embeddings with the temporal sequence tensor andReshape node_embeddings to broadcast across the sequence length
        node_embeddings_expanded = node_embeddings.unsqueeze(2).expand(-1, -1, temporal_seq.size(2))
        combined_input = torch.cat([temporal_seq, node_embeddings_expanded], dim=1) 
        
        # Pass through 1D-CNN
        c_out = F.relu(self.conv1d(combined_input)) 
        c_out = self.pool(c_out)                    
        
        # Flatten and classify
        flat = c_out.view(c_out.size(0), -1)
        out = F.relu(self.fc1(flat))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return torch.sigmoid(logits)


def train_gcn_cnn_model(data, temporal_seq: torch.Tensor):
    """Initializes the model and runs a forward pass."""
    x = data['account'].x
    
    # We use a standard homogenous edge index for the SAGEConv layer demo
    # In a full heterogeneous setup, HeteroConv is used.
    edge_index = data['account', 'transacts_with', 'merchant'].edge_index
    
    # Initialize the Hybrid model
    model = HybridGCFDFraudDetector(
        in_features=64, 
        gcn_hidden=128, 
        gnn_out=64, 
        cnn_channels=32, 
        seq_len=10
    )
    
    # Execute a forward pass to generate fraud probabilities
    fraud_probabilities = model(x, edge_index, temporal_seq)
    
    return fraud_probabilities


def evaluate_model_performance(data, temporal_seq: torch.Tensor, fraud_probabilities: torch.Tensor, risk_params: dict) -> dict:
    """
    Evaluates topological retrieval, ROC-AUC, and Embedded Anomaly F1-Score 
    based on dynamic risk appetites.
    """
    threshold = risk_params['decision_threshold']
    
    # 1. Convert output probabilities into binary predictions using the dynamic threshold
    prob_arrays = fraud_probabilities.detach().cpu().numpy()
    preds = (prob_arrays > threshold).astype(int)
    
    # (Generating dummy ground-truth targets for the pipeline flow)
    targets = (torch.rand_like(fraud_probabilities) > 0.5).long().cpu().numpy()
    
    # 2. Compute standard and dynamic risk metrics
    f1_embedded = f1_score(targets, preds, zero_division=0)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    
    # ROC-AUC evaluates the model's performance independent of the threshold
    roc_auc = roc_auc_score(targets, prob_arrays)
    
    # 3. Graph Topology Analysis (Cycles & Chains)
    edge_index = data['account', 'transacts_with', 'merchant'].edge_index
    edges = edge_index.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edges)
    
    graph_cycles = len(nx.cycle_basis(G))
    
    metrics = {
        "roc_auc": float(roc_auc),
        "embedded_anomaly_f1": float(f1_embedded),
        "precision": float(precision),
        "recall": float(recall),
        "detected_graph_cycles": int(graph_cycles),
        "applied_threshold": float(threshold)
    }
    
    print("\n================ AML RISK EVALUATION ===============")
    print(f"Risk Threshold Applied: {threshold}")
    print(f"ROC-AUC (Threshold Independent): {roc_auc:.4f}")
    print(f"Embedded Anomaly F1-Score: {f1_embedded:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"Detected Topology Cycles: {graph_cycles}")
    print("====================================================\n")
    
    return metrics

def run_louvain_community_detection(data, fraud_probabilities: torch.Tensor) -> dict:
    """
    Partitions the network into fraud syndicates and renders a PyVis dashboard.
    """
    # 1. Extract the edge topology from the PyTorch Geometric tensor
    edge_index = data['account', 'transacts_with', 'merchant'].edge_index
    edges = edge_index.t().tolist()
    
    # 2. Build the NetworkX Graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # 3. Run Louvain Community Detection
    # Identifies dense clusters of accounts indicating laundering syndicates
    communities = nx.community.louvain_communities(G)
    
    # Map each node to its community ID
    partition = {}
    for cluster_id, node_set in enumerate(communities):
        for node in node_set:
            partition[node] = cluster_id
            
    # 4. Generate PyVis Interactive UI
    # Create a dark-mode dashboard for risk analysts
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes colored by their Louvain community cluster
    for node, comm_id in partition.items():
        net.add_node(int(node), group=comm_id, title=f"Entity ID: {node} | Syndicate: {comm_id}")
        
    for src, dst in edges:
        net.add_edge(int(src), int(dst))
        
    # Ensure the model output directory exists
    os.makedirs("data/07_model_output", exist_ok=True)
    
    # Export the dashboard
    output_path = "data/07_model_output/fraud_rings_dashboard.html"
    net.write_html(output_path)
    
    return partition