from kedro.pipeline import Pipeline, node
from .nodes import predict_new_clients

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=predict_new_clients,
                inputs=dict(
                    test_events="test_events",
                    kmeans_model_path="params:kmeans_model_path"  # <-- ici
                ),
                outputs="calssifiesd_clients",
            )
        ]
    )
