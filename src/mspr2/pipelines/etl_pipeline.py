from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_events, filter_events, user_event_counts, user_spent_stats, user_diversity_stats, build_final_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_events,
            inputs="events",
            outputs="cleaned_events",
            name="clean_events_node",
        ),
        node(
            func=filter_events,
            inputs="cleaned_events",
            outputs="filtered_events",
            name="filter_events_node",
        ),
        node(
            func=user_event_counts,
            inputs="filtered_events",
            outputs="user_event_counts",
            name="user_event_counts_node",
        ),
        node(
            func=user_spent_stats,
            inputs="filtered_events",
            outputs="user_spent_stats",
            name="user_spent_stats_node",
        ),
        node(
            func=user_diversity_stats,
            inputs="filtered_events",
            outputs="user_diversity_stats",
            name="user_diversity_stats_node",
        ),
        node(
            func=build_final_metrics,
            inputs=["user_event_counts", "user_spent_stats", "user_diversity_stats"],
            outputs="final_event_metrics",
            name="build_final_metrics_node",
        ),
    ])
