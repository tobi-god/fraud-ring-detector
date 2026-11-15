from kedro.pipeline import Pipeline
from fraud_ring_detector.pipelines.fraud_detection import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipeline = create_pipeline()
    return {"__default__": pipeline, "fraud_detection": pipeline}
