from kedro.pipeline import Pipeline, node
from .nodes import predict_new_clients

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=predict_new_clients,
                inputs=["test_events", "params:kmeans_model_path"],
                outputs="calssifiesd_clients",
                name="predict_new_clients_node",
            )
        ]
    )
