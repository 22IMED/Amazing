from snowflake.snowpark import functions as F
from snowflake.snowpark import DataFrame as SnowparkDataFrame
import pandas as pd
import joblib
from snowflake.snowpark.session import Session
from sklearn.preprocessing import StandardScaler

from ..nodes import (
    clean_events,
    filter_events,
    user_event_counts,
    user_spent_stats,
    user_diversity_stats,
    build_final_metrics,
)
from ..modeling_nodes import (
    prepare_rfm,
)
import numpy as np



def predict_new_clients(test_events: SnowparkDataFrame, kmeans_model_path: str) -> SnowparkDataFrame:
    """
    Node pour prédire la classe des nouveaux clients à partir de test_events.
    """

      # 1. Nettoyage et filtrage
    df_cleaned = clean_events(test_events)
    df_cleaned.show(5)
    df_filtered = filter_events(df_cleaned)
    df_cleaned.show(5)
    # Étapes RFM
    counts = user_event_counts(df_filtered)
    counts.show(5)
    spent = user_spent_stats(test_events)
    spent.show(5)
    diversity = user_diversity_stats(test_events)
    diversity.show(5)
    final_metrics = build_final_metrics(counts, spent, diversity)
    final_metrics.show(5)
    

    # Récupérer le DataFrame Pandas pour KMeans

    df_pd = final_metrics.to_pandas()
    df_pd.columns = df_pd.columns.str.replace('"', '')
    print(df_pd.columns.tolist())
    df_pd["LAST_PURCHASE_DATE"] = pd.to_datetime(
    df_pd["LAST_PURCHASE_DATE"], errors="coerce"
    )

    # Garder uniquement la date jj-mm-aaaa
    df_pd["LAST_PURCHASE_DATE"] = df_pd["LAST_PURCHASE_DATE"].dt.date
    rfm_df = prepare_rfm(df_pd)
    # Préparer features pour KMeans
    features = ["recency", "frequency", "monetary",]
    print('rfm_df',rfm_df)
    X = rfm_df[features].fillna(0).astype(np.float32)
    print("x",X)
    X_sample = X 
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X_sample)
    X_scaled = (X - X_sample.mean()) / X_sample.std()
    X_scaled = (X_sample - X_sample.mean()) / X_sample.std()
    X_scaled = X_scaled.fillna(0)

    # Charger modèle KMeans
    kmeans = joblib.load(kmeans_model_path)
    print(X_scaled)
    labels = kmeans.predict(X_scaled)
    print("labels",labels)
    # Ajouter la classe au DataFrame
    df_pd["CLIENT_CLASS"] = labels

    # 5. Convertir Pandas → Snowpark
    session = test_events.session  # récupérer la session Snowpark existante
    result_df = session.create_dataframe(
        df_pd[["USER_ID", "CLIENT_CLASS"]].values.tolist(),
        schema=["USER_ID", "CLIENT_CLASS"]
    )

    return result_df

