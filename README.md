Hybrid GCFD: Graph Convolutional Fraud Detector 🕵️‍♂️📊

An enterprise-grade, hybrid deep learning pipeline built with Kedro and PyTorch Geometric to detect complex money-laundering syndicates and corporate financial fraud.

📖 Executive Summary

Context: Modern financial ecosystems require high-velocity transaction processing without exposing the network to sophisticated money laundering syndicates.


Complication: Traditional rules-based engines and tabular ML models (like XGBoost) evaluate transactions in isolation. Sophisticated actors easily bypass these systems using multi-party layering, shell companies, and circular payment schemes, resulting in catastrophic false positive rates.
Resolution: This project deploys a Hybrid Graph Convolutional Fraud Detector (GCFD). By treating transactions as a bipartite network rather than isolated events, the system aggregates multi-hop neighborhood risk topologies (GCN) and fuses them with temporal sliding-window features (1D-CNN).
Impact:

Operational Actionability: Working off standard academic baselines (e.g., Cai et al., 2026) that output isolated probability scores, this pipeline passes latent embeddings through Louvain Community Detection to automatically cluster nodes into discrete, actionable money-laundering syndicates via an interactive PyVis UI.

Dynamic Risk Appetite: Evaluates networks using the Embedded Anomaly F1-Score (EAF1) and ROC-AUC, allowing compliance teams to dynamically slide decision thresholds based on real-world regulatory risk appetites.

🏗️ Architecture & Pipeline (Kedro DAG)

This project leveraged QuantumBlack's enterprise standard for Kedro pipelines, decoupling data management, cloud graph persistence, and deep learning tensor flows.

build_graph_node: Ingests raw CSV data and pushes the bipartite topology to a cloud-hosted Neo4j AuraDB instance. Outputs PyG HeteroData tensors.

extract_temporal_node: Extracts sliding temporal windows (amount, frequency) to a 3D sequence tensor for CNN consumption.

train_hybrid_model_node: Fuses GCN (SAGEConv) structural embeddings with 1D-CNN spatial-temporal extractions to predict localized fraud probabilities.

louvain_clustering_node: Partitions network topology to group flagged accounts into syndicates, exporting an interactive fraud_rings_dashboard.html.

evaluation_node: Computes Threshold-Independent ROC-AUC, Embedded Anomaly F1-Scores, and Topological Cycle structural retrievals.

🛠️ Tech Stack

Orchestration: Kedro

Deep Learning: PyTorch, PyTorch Geometric (PyG)

Graph Database: Neo4j AuraDB (Cypher)

Graph Analytics: NetworkX, Louvain Modularity

Visualization: PyVis

🚀 How to Run Locally (based off my experience)

Note: This pipeline is optimized to run safely on machines with standard memory (8GB RAM) by utilizing micro-sampling.

1. Clone the repository and activate your environment:

git clone https://github.com/YOUR_USERNAME/fraud-ring-detector.git
cd fraud-ring-detector
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\Activate.ps1


2. Install dependencies:

use pip install reading the requirements.txt file


3. Configure Credentials & Parameters:

Add your model hyperparameters to conf/base/parameters.yml.

Create a conf/local/credentials.yml file (this is git-ignored for safety) and add your Neo4j AuraDB connection details:

neo4j_credentials:
  uri: "neo4j+s://YOUR_INSTANCE.databases.neo4j.io"
  username: "neo4j"
  password: "YOUR_PASSWORD"


4. Prepare the Data:
Place your raw dataset in data/01_raw/. ( To prevent RAM crashes, I sliced the dataset to a 50,000-row sample)

python filter_data.py


5. Execute the Pipeline:

kedro run


📊 Viewing the Results

Once the pipeline successfully finishes, navigate to data/07_model_output/ and open fraud_rings_dashboard.html in your web browser to explore the interactive Louvain community clusters.