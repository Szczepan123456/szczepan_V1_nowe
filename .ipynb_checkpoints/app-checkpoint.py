import json
import os
import uuid

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pycaret.clustering import load_model, predict_model
import plotly.express as px

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---- Wczytaj zmienne z .env ----
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# ---- Inicjalizacja Qdrant ----
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Tworzenie kolekcji, jeśli nie istnieje
EMBEDDING_DIM = 10  # Liczba wymiarów po kodowaniu cech
if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

# ---- Funkcja do przesyłania użytkownika do Qdrant ----
def upload_to_qdrant(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df)
    # Uzupełnij brakujące kolumny, jeśli są różnice w strukturze
    for col in required_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[required_columns]
    vector = df_encoded.values.astype(np.float32)[0]

    point = PointStruct(
        id=uuid.uuid4().int >> 64,
        vector=vector.tolist(),
        payload=df.iloc[0].to_dict()
    )

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[point]
    )
    st.success("✅ Użytkownik został zapisany do Qdrant.")

# ---- Dane wejściowe i model ----
MODEL_NAME = 'welcome_survey_clustering_pipeline_v1_model_szczepan_v1'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_DESC = 'welcome_survey_cluster_names_and_descriptions_szv1.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_descriptions():
    with open(CLUSTER_DESC, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_data():
    df = pd.read_csv(DATA, sep=';')
    return predict_model(model, data=df)

model = get_model()
cluster_names = get_cluster_descriptions()
all_df = get_all_data()

# ---- Interfejs użytkownika ----
with st.sidebar:
    st.header("👤 Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby o podobnych zainteresowaniach!")

    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender
    }])

# ---- Przewidywanie klastra ----
predicted_cluster = predict_model(model, data=person_df)["Cluster"].values[0]
cluster_data = cluster_names[str(predicted_cluster)]

# ---- Wyświetl klaster ----
st.header(f"🧠 Jesteś najbliżej grupy: {cluster_data['name']} 🧭")
st.markdown(cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster]
st.metric("👥 Liczba podobnych osób", len(same_cluster_df))

# ---- Wykresy kołowe ----
def pie_chart(data, column, title):
    fig = px.pie(data, names=column, title=title)
    st.plotly_chart(fig)

st.header("📊 Charakterystyka grupy")
pie_chart(same_cluster_df, "age", "Wiek")
pie_chart(same_cluster_df, "edu_level", "Wykształcenie")
pie_chart(same_cluster_df, "fav_animals", "Ulubione zwierzęta")
pie_chart(same_cluster_df, "fav_place", "Ulubione miejsca")
pie_chart(same_cluster_df, "gender", "Płeć")

# ---- Przycisk do Qdrant ----
# Przechowaj kolumny wymagane do spójności wektora
required_columns = list(pd.get_dummies(all_df[['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']]).columns)

if st.button("📤 Zapisz mnie do Qdrant"):
    upload_to_qdrant(person_df)
