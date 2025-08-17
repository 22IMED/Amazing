# # src/project_name/pipelines/rfm/pipeline.py
# from kedro.pipeline import Pipeline, node, pipeline
# from .modeling_nodes import prepare_rfm, kmeans_rfm_evaluation

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#             func=prepare_rfm,
#             inputs="event_metrics",
#             outputs="rfm_prepared",
#             name="prepare_rfm_node"
#         ),
#         node(
#             func=kmeans_rfm_evaluation,
#             inputs="rfm_prepared",
#             outputs="rfm_kmeans_results",
#             name="kmeans_rfm_evaluation_node"
#         )
#     ])

# src/project_name/pipelines/rfm/pipeline.py
from kedro.pipeline import Pipeline, node
from .modeling_nodes import prepare_rfm, kmeans_rfm_evaluation

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_rfm,
                inputs="event_metrics",    # Nom du dataset dans catalog.yml
                outputs="rfm_df",
                name="prepare_rfm_node"
            ),
            node(
                func=kmeans_rfm_evaluation,
                inputs="rfm_df",
                outputs="kmeans_results",
                name="kmeans_rfm_node"
            ),
        ]
    )
