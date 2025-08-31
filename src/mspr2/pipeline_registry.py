from kedro.pipeline import Pipeline
from mspr2.pipelines import etl_pipeline, modeling_pipeline# importe ton pipeline
from mspr2.pipelines.prediction import pipeline as prediction_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "etl_pipeline": etl_pipeline.create_pipeline(),  # nom que tu donnes en CLI
        "modeling_pipeline": modeling_pipeline.create_pipeline(),
        "prediction_pipeline": prediction_pipeline.create_pipeline(),
        "__default__": etl_pipeline.create_pipeline()
    }
