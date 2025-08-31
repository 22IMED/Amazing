# Image Python officielle
FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier les requirements et installer les packages Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copier le code source et la configuration
COPY src/ src/
COPY conf/ conf/

# Variables d'environnement Snowflake
ENV SNOWFLAKE_USER=""
ENV SNOWFLAKE_PASSWORD=""
ENV SNOWFLAKE_ACCOUNT=""
ENV SNOWFLAKE_WAREHOUSE=""
ENV SNOWFLAKE_DATABASE="AMAZING_DB"
ENV SNOWFLAKE_SCHEMA="SCHEMAS"

# Commande par défaut : exécuter uniquement le pipeline de prédiction
CMD ["kedro", "run", "--pipeline=prediction_pipeline"]
