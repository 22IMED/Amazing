from snowflake.snowpark import functions as F
from snowflake.snowpark import Window
from snowflake.snowpark import DataFrame as SnowparkDataFrame

# Config issu du notebook
COLUMNS_TO_FILL_NA = {
    "price": 0.0,
    "product_id": "unknown",
    "event_type": "unknown",
    "CATEGORY_CODE": "unknown",
    "brand": "unknown",
}
DATETIME_COLUMNS = ["EVENT_TIME"]
NUMERIC_COLUMNS = ["price"]
VALID_EVENT_TYPES = ["purchase", "cart", "view"]
PRICE_MIN_THRESHOLD = 0.01
PRICE_MAX_THRESHOLD = 10000.0


def clean_events(events: SnowparkDataFrame) -> SnowparkDataFrame:
    """Nettoyage des événements (nulls, types, valeurs manquantes)."""
    df = events.filter(F.col("user_id").is_not_null())

    # Remplir valeurs manquantes
    for col, default in COLUMNS_TO_FILL_NA.items():
        df = df.with_column(col, F.coalesce(F.col(col), F.lit(default)))

    # Convertir colonnes datetime
    for col in DATETIME_COLUMNS:
        df = df.with_column(col, F.to_date(F.col(col)))

    # Convertir colonnes numériques
    for col in NUMERIC_COLUMNS:
        df = df.with_column(col, F.col(col).cast("float"))

    return df


def filter_events(events: SnowparkDataFrame) -> SnowparkDataFrame:
    """Filtrer sur les types d'événements valides et prix dans l'intervalle."""
    df = events.filter(F.col("event_type").isin(VALID_EVENT_TYPES))

    # Supprimer doublons
    window_spec = (
        Window.partition_by("user_id", "EVENT_TIME", "event_type", "product_id")
        .order_by(F.col("EVENT_TIME"))
    )
    df = df.with_column("row_num", F.row_number().over(window_spec))
    df = df.filter(F.col("row_num") == 1).drop("row_num")

    # Filtrer sur les prix valides
    df = df.filter(
        (F.col("price") >= PRICE_MIN_THRESHOLD)
        & (F.col("price") <= PRICE_MAX_THRESHOLD)
    )

    return df


def user_event_counts(events: SnowparkDataFrame) -> SnowparkDataFrame:
    """Nombre d'événements par utilisateur (purchase/view/cart)."""
    event_counts = events.group_by("USER_ID").agg(
        F.sum(F.when(F.col("EVENT_TYPE") == "purchase", 1).otherwise(0)).alias("purchase"),
        F.sum(F.when(F.col("EVENT_TYPE") == "view", 1).otherwise(0)).alias("view"),
        F.sum(F.when(F.col("EVENT_TYPE") == "cart", 1).otherwise(0)).alias("cart"),
    )
    return event_counts


def user_spent_stats(events: SnowparkDataFrame) -> SnowparkDataFrame:
    """Total, prix moyen et date du dernier achat par utilisateur (uniquement events purchase)."""
    purchase_data = events.filter(F.col("EVENT_TYPE") == "purchase")
    spent_stats = purchase_data.group_by("USER_ID").agg(
        F.sum("PRICE").alias("total_spent"),
        F.avg("PRICE").alias("avg_purchase_price"),
        F.max("EVENT_TIME").alias("last_purchase_date")  
    )
    return spent_stats



def user_diversity_stats(events: SnowparkDataFrame) -> SnowparkDataFrame:
    """Nombre de catégories et marques uniques par utilisateur."""
    diversity = events.group_by("USER_ID").agg(
        F.count_distinct("CATEGORY_CODE").alias("unique_categories"),
        F.count_distinct("BRAND").alias("unique_brands"),
    )
    return diversity


def build_final_metrics(
    counts: SnowparkDataFrame, spent: SnowparkDataFrame, diversity: SnowparkDataFrame
) -> SnowparkDataFrame:
    """Jointure des features utilisateur et création de la table finale."""
    final_df = (
        counts.join(spent, on="USER_ID", how="left")
              .join(diversity, on="USER_ID", how="left")
    )

    # Remplacer les nulls par 0
    final_df = (
        final_df.with_column("total_spent", F.coalesce(F.col("total_spent"), F.lit(0.0)))
                .with_column("avg_purchase_price", F.coalesce(F.col("avg_purchase_price"), F.lit(0.0)))
                .with_column("unique_categories", F.coalesce(F.col("unique_categories"), F.lit(0)))
                .with_column("unique_brands", F.coalesce(F.col("unique_brands"), F.lit(0)))
    )

    # Conversion rate = purchases / views
    final_df = final_df.with_column(
        "conversion_rate",
        F.when(F.col("view") > 0, F.col("purchase") / F.col("view")).otherwise(0.0),
    )

    return final_df
