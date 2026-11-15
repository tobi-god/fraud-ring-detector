from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (build_features, detect_syndicates, evaluate_models, evaluate_patterns, prepare_dataset, sync_to_neo4j,
                    train_baselines, train_hybrid)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(prepare_dataset, ["raw_transactions", "params:data"], "model_dataset", name="prepare_dataset_node"),
        node(build_features, ["model_dataset", "params:features"], "model_features", name="build_features_node"),
        node(train_baselines, ["model_dataset", "model_features", "params:data"], "baseline_scores",
             name="train_baselines_node"),
        node(train_hybrid, ["model_dataset", "model_features", "params:model"], "hybrid_scores",
             name="train_hybrid_model_node"),
        node(evaluate_models, ["model_dataset", "baseline_scores", "hybrid_scores", "params:evaluation"],
             ["evaluation_metrics", "threshold_sweep"], name="evaluation_node"),
        node(evaluate_patterns, ["model_dataset", "baseline_scores", "hybrid_scores", "evaluation_metrics", "params:evaluation"],
             "pattern_recall", name="pattern_evaluation_node"),
        node(detect_syndicates, ["model_dataset", "baseline_scores", "hybrid_scores", "evaluation_metrics", "params:syndicates"],
             ["syndicate_summary", "account_roles", "fraud_rings_dashboard"], name="syndicate_analysis_node"),
        node(sync_to_neo4j, ["model_dataset", "hybrid_scores", "evaluation_metrics", "params:neo4j"],
             "neo4j_sync_status", name="neo4j_sync_node"),
    ])
