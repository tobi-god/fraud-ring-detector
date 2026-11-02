from kedro.pipeline import Pipeline, node
from .nodes import (build_bipartite_graph_and_sync, extract_temporal_windows, train_gcn_cnn_model, run_louvain_community_detection, evaluate_model_performance) 


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_bipartite_graph_and_sync,
                inputs=["raw_transactions_dataset", "params:neo4j_credentials"],
                outputs="pyg_bipartite_tensor",
                name="build_graph_node",
            ),
            node(
                func=extract_temporal_windows,
                inputs="raw_transactions_dataset",
                outputs="temporal_sequence_tensor",
                name="extract_temporal_node",
            ),
            node(
                func=train_gcn_cnn_model,
                inputs=["pyg_bipartite_tensor", "temporal_sequence_tensor"],
                outputs="fraud_probabilities",
                name="train_hybrid_model_node",
            ),
            node(
                func=run_louvain_community_detection,
                inputs=["pyg_bipartite_tensor", "fraud_probabilities"],
                outputs="community_partitions",
                name="louvain_clustering_node",
            ),
            node(
                func=evaluate_model_performance,
                inputs=["pyg_bipartite_tensor", "temporal_sequence_tensor", "fraud_probabilities","params:aml_risk_appetite"],
                outputs="evaluation_metrics",
                name="evaluation_node",
            )
        ]
    )