# # src/project_name/pipelines/rfm/nodes.py
# from snowflake.snowpark import functions as F
# from snowflake.snowpark import DataFrame as SnowparkDataFrame
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import pandas as pd

# def prepare_rfm(user_metrics: SnowparkDataFrame) -> SnowparkDataFrame:
#     """Prépare le dataframe RFM avancé, remplace les NULL de LAST_PURCHASE_DATE."""
#     max_date = user_metrics.agg(F.max("LAST_PURCHASE_DATE")).collect()[0][0]

#     rfm_df = (
#         user_metrics
#         .with_column(
#             "recency",
#             F.datediff(
#                 "day",
#                 F.coalesce(F.col("LAST_PURCHASE_DATE"), F.to_date(F.lit("1900-01-01"))),
#                 F.to_date(F.lit(max_date))
#             )
#         )
#         .with_column("frequency", F.col("PURCHASE"))
#         .with_column("monetary", F.col("TOTAL_SPENT"))
#         .with_column("avg_purchase_price", F.col("AVG_PURCHASE_PRICE"))
#         .with_column("view", F.col("VIEW"))
#         .with_column("cart", F.col("CART"))
#         .with_column("conversion_rate", F.col("CONVERSION_RATE"))
#         .with_column("unique_brands", F.col("UNIQUE_BRANDS"))
#         .with_column("unique_categories", F.col("UNIQUE_CATEGORIES"))
#         .select(
#             "USER_ID",
#             "recency",
#             "frequency",
#             "monetary",
#             "avg_purchase_price",
#             "view",
#             "cart",
#             "conversion_rate",
#             "unique_brands",
#             "unique_categories"
#         )
#     )
#     return rfm_df

# def kmeans_rfm_evaluation(rfm_df: SnowparkDataFrame, min_k=2, max_k=10):
#     """Applique KMeans sur le RFM avancé, calcule silhouette et méthode du coude."""
#     df = rfm_df.to_pandas()
#     features = [
#         "recency", "frequency", "monetary",
#         "avg_purchase_price", "view", "cart",
#         "conversion_rate", "unique_brands", "unique_categories"
#     ]
#     X = df[features].fillna(0)
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     inertias, silhouettes, kmeans_models = [], [], {}
    
#     for k in range(min_k, max_k+1):
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(X_scaled)
#         inertias.append(kmeans.inertia_)
#         sil_score = silhouette_score(X_scaled, labels) if k > 1 else None
#         silhouettes.append(sil_score)
#         kmeans_models[k] = kmeans
#         print(f"K={k}, Inertia={kmeans.inertia_}, Silhouette={sil_score}")
    
#     # Visualisation
#     fig, ax1 = plt.subplots(figsize=(10,5))
#     ax1.set_xlabel("k (nombre de clusters)")
#     ax1.set_ylabel("Inertia (WCSS)", color='tab:blue')
#     ax1.plot(range(min_k, max_k+1), inertias, marker='o', color='tab:blue')
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("Silhouette Score", color='tab:red')
#     ax2.plot(range(min_k, max_k+1), silhouettes, marker='x', color='tab:red')
#     plt.title("Méthode du Coude et Silhouette pour RFM Clustering")
#     plt.show()
    
#     return {"inertia": inertias, "silhouette": silhouettes, "kmeans_models": kmeans_models}



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import seaborn as sns

def prepare_rfm(user_metrics: pd.DataFrame) -> pd.DataFrame:
    """Prépare le DataFrame RFM avancé, remplace les NULL de LAST_PURCHASE_DATE."""
    # Convertir en datetime et forcer les erreurs à NaT
    user_metrics["LAST_PURCHASE_DATE"] = pd.to_datetime(
        user_metrics["LAST_PURCHASE_DATE"], errors="coerce"
    )
    
    # Remplacer les NaT par une date ancienne si nécessaire
    max_date = user_metrics["LAST_PURCHASE_DATE"].max()
    
    rfm_df = user_metrics.copy()
    rfm_df["recency"] = (max_date - user_metrics["LAST_PURCHASE_DATE"]).dt.days
    rfm_df["frequency"] = user_metrics["PURCHASE"]
    rfm_df["monetary"] = user_metrics["TOTAL_SPENT"]
    rfm_df["avg_purchase_price"] = user_metrics["AVG_PURCHASE_PRICE"]
    rfm_df["view"] = user_metrics["VIEW"]
    rfm_df["cart"] = user_metrics["CART"]
    rfm_df["conversion_rate"] = user_metrics["CONVERSION_RATE"]
    rfm_df["unique_brands"] = user_metrics["UNIQUE_BRANDS"]
    rfm_df["unique_categories"] = user_metrics["UNIQUE_CATEGORIES"]
    
    return rfm_df[
        [
            "USER_ID", "recency", "frequency", "monetary", "avg_purchase_price",
            "view", "cart", "conversion_rate", "unique_brands", "unique_categories"
        ]
    ]


def kmeans_rfm_evaluation(rfm_df: pd.DataFrame, min_k=2, max_k=10, output_dir="outputs"):
    """
    Applique KMeans sur le RFM avancé, calcule silhouette, méthode du coude et analyse les clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    features = ["recency", "frequency", "monetary"]
    # X = rfm_df[features].fillna(0).astype(np.float32)
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)


    sample_size = 100000 # Taille d'échantillon pour scaler et silhouette 
    X = rfm_df[features].fillna(0).astype(np.float32) 
    # Échantillonnage pour scaler et silhouette si dataset très grand 
    if len(X) > sample_size: 
      X_sample = X.sample(sample_size, random_state=42) 
    else: 
      X_sample = X 
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X_sample)
    X_scaled = (X - X_sample.mean()) / X_sample.std()
    X_scaled = (X_sample - X_sample.mean()) / X_sample.std()


    
    inertias, silhouettes, kmeans_models = [], [], {}
    
    for k in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, labels) if k > 1 else None
        silhouettes.append(sil_score)
        kmeans_models[k] = kmeans
        print(f"K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")
        
        # Ajouter labels au DataFrame et sauvegarder
        cluster_df = X_scaled.copy()
        cluster_df["cluster"] = labels
        cluster_df.to_csv(os.path.join(output_dir, f"rfm_clusters_k{k}.csv"), index=False)
        
        # Analyse des clusters
        summary = cluster_df.groupby("cluster")[features].agg(["mean","median","count"])
        print(f"\nRésumé clusters K={k}:\n", summary)
        summary.to_csv(os.path.join(output_dir, f"cluster_summary_k{k}.csv"))
        
        # Heatmap des moyennes
        plt.figure(figsize=(8,5))
        sns.heatmap(summary.xs("mean", axis=1, level=1), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Heatmap des moyennes par cluster K={k}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_clusters_k{k}.png"))
        plt.close()
        
        # Sauvegarde du modèle
        joblib.dump(kmeans, os.path.join(output_dir, f"kmeans_model_k{k}.joblib"))
    
    # Méthode du coude et silhouette
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.set_xlabel("k (nombre de clusters)")
    ax1.set_ylabel("Inertia (WCSS)", color='tab:blue')
    ax1.plot(range(min_k, max_k+1), inertias, marker='o', color='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color='tab:red')
    ax2.plot(range(min_k, max_k+1), silhouettes, marker='x', color='tab:red')
    
    plt.title("Méthode du Coude et Silhouette pour RFM Clustering")
    plt.savefig(os.path.join(output_dir, "elbow_silhouette.png"))
    plt.show()
    
    return {"inertia": inertias, "silhouette": silhouettes, "kmeans_models": kmeans_models}


# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import numpy as np

# def prepare_rfm(user_metrics: pd.DataFrame) -> pd.DataFrame:
#     """Prépare le DataFrame RFM avancé, remplace les NULL de LAST_PURCHASE_DATE."""
#     # Convertir en datetime et forcer les erreurs à NaT
#     user_metrics["LAST_PURCHASE_DATE"] = pd.to_datetime(
#         user_metrics["LAST_PURCHASE_DATE"], errors="coerce"
#     )
    
#     # Remplacer les NaT par une date ancienne si nécessaire
#     max_date = user_metrics["LAST_PURCHASE_DATE"].max()
    
#     rfm_df = user_metrics.copy()
#     rfm_df["recency"] = (max_date - user_metrics["LAST_PURCHASE_DATE"]).dt.days
#     rfm_df["frequency"] = user_metrics["PURCHASE"]
#     rfm_df["monetary"] = user_metrics["TOTAL_SPENT"]
#     rfm_df["avg_purchase_price"] = user_metrics["AVG_PURCHASE_PRICE"]
#     rfm_df["view"] = user_metrics["VIEW"]
#     rfm_df["cart"] = user_metrics["CART"]
#     rfm_df["conversion_rate"] = user_metrics["CONVERSION_RATE"]
#     rfm_df["unique_brands"] = user_metrics["UNIQUE_BRANDS"]
#     rfm_df["unique_categories"] = user_metrics["UNIQUE_CATEGORIES"]
    
#     return rfm_df[
#         [
#             "USER_ID", "recency", "frequency", "monetary", "avg_purchase_price",
#             "view", "cart", "conversion_rate", "unique_brands", "unique_categories"
#         ]
#     ]


# def kmeans_rfm_evaluation(rfm_df: pd.DataFrame, min_k=2, max_k=10):
#     """
#     Applique KMeans sur le RFM avancé, calcule silhouette et méthode du coude.
#     """
#     features = [
#         "recency", "frequency", "monetary",
#     ]
#     # X = rfm_df[features].fillna(0)
    
#     # scaler = StandardScaler()
#     # X_scaled = scaler.fit_transform(X)
#     sample_size = 1000000  # Taille d'échantillon pour scaler et silhouette
#     X = rfm_df[features].fillna(0).astype(np.float32)
    
#     # Échantillonnage pour scaler et silhouette si dataset très grand
#     if len(X) > sample_size:
#         X_sample = X.sample(sample_size, random_state=42)
#     else:
#         X_sample = X

#     scaler = StandardScaler()
#     X_scaled_sample = scaler.fit_transform(X_sample)
    
#     # Scale l'ensemble des données par transformation des stats du sample
#     X_scaled = (X - X_sample.mean()) / X_sample.std()
    
    
#     inertias, silhouettes, kmeans_models = [], [], {}
    
#     for k in range(min_k, max_k+1):
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(X_scaled)
#         inertias.append(kmeans.inertia_)
#         sil_score = silhouette_score(X_scaled, labels, sample_size=1000) if k > 1 else None
#         silhouettes.append(sil_score)
#         kmeans_models[k] = kmeans
#         print(f"K={k}, Inertia={kmeans.inertia_}, Silhouette={sil_score}")
    
#     # Visualisation
#     fig, ax1 = plt.subplots(figsize=(10,5))
#     ax1.set_xlabel("k (nombre de clusters)")
#     ax1.set_ylabel("Inertia (WCSS)", color='tab:blue')
#     ax1.plot(range(min_k, max_k+1), inertias, marker='o', color='tab:blue')
    
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("Silhouette Score", color='tab:red')
#     ax2.plot(range(min_k, max_k+1), silhouettes, marker='x', color='tab:red')
    
#     plt.title("Méthode du Coude et Silhouette pour RFM Clustering")
#     plt.show()
    
#     return {"inertia": inertias, "silhouette": silhouettes, "kmeans_models": kmeans_models}
