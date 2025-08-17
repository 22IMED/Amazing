from kedro.pipeline import Pipeline
from mspr2.pipelines import etl_pipeline  # importe ton pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "etl_pipeline": etl_pipeline.create_pipeline(),  # nom que tu donnes en CLI
        "__default__": etl_pipeline.create_pipeline()
    }
