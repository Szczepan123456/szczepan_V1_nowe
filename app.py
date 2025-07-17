import streamlit as st
import pycaret
import pandas as pd

st.write("PyCaret version:", pycaret.__version__)
st.write("Pandas version:", pd.__version__)

# dalej idzie Twój dotychczasowy kod




import json
import os
import uuid
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pycaret.clustering import load_model, predict_model
import plotly.express as px

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

import openai  # trzeba mieć zainstalowane openai pip install openai

# ---- Load env variables ----
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "szczepan_v1")

# ---- Qdrant client ----
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ---- Load model and data ----
MODEL_NAME = 'welcome_survey_clustering_pipeline_v1_model_szczepan_v1'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_DESC = 'welcome_survey_cluster_names_and_descriptions_szv1.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_descriptions():
    with open(CLUSTER_DESC, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_data():
    df = pd.read_csv(DATA, sep=';')
    return df

# --- UI sidebar for user input ---
with st.sidebar:
    st.header("🔑 Wprowadź swój klucz OpenAI")
    openai_key = st.text_input(
        "Klucz OpenAI",
        type="password",
        help="Wpisz swój indywidualny klucz API OpenAI"
    )
    if not openai_key:
        st.warning("Aby korzystać z aplikacji, wpisz proszę swój klucz OpenAI.")
        st.stop()  # zatrzymuje wykonanie dalszej części, dopóki nie wpisze klucza
    
    # Ustaw klucz OpenAI globalnie
    openai.api_key = openai_key

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

model = get_model()
cluster_names = get_cluster_descriptions()
all_df_raw = get_all_data()
all_df = predict_model(model, data=all_df_raw)

# ---- Predict cluster for current user ----
predicted_cluster = predict_model(model, data=person_df)["Cluster"].values[0]
cluster_data = cluster_names[str(predicted_cluster)]

st.header(f"🧠 Najbliżej Ci do grupy: {cluster_data['name']}")
st.markdown(cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster]
st.metric("👥 Liczba osób w Twojej grupie", len(same_cluster_df))

# ---- Pie charts function ----
def pie_chart(data, column, title):
    fig = px.pie(data, names=column, title=title)
    st.plotly_chart(fig)

st.header("📊 Charakterystyka grupy")
pie_chart(same_cluster_df, "age", "Wiek")
pie_chart(same_cluster_df, "edu_level", "Wykształcenie")
pie_chart(same_cluster_df, "fav_animals", "Ulubione zwierzęta")
pie_chart(same_cluster_df, "fav_place", "Ulubione miejsca")
pie_chart(same_cluster_df, "gender", "Płeć")

# ---- Prepare columns for one-hot encoding (from all data) ----
categorical_cols = ['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']
encoded_all = pd.get_dummies(all_df[categorical_cols])
required_columns = list(encoded_all.columns)
EMBEDDING_DIM = len(required_columns)
st.write(f"ℹ️ Liczba cech po one-hot encoding: {EMBEDDING_DIM}")

# ---- Create collection if not exists (AFTER knowing embedding size) ----
if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

# ---- Function to upload single user to Qdrant ----
def upload_single_user(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df[categorical_cols])
    for col in required_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[required_columns]

    vector = df_encoded.values.astype(np.float32)[0]

    if len(vector) != EMBEDDING_DIM:
        st.error(f"❌ Wektor ma {len(vector)} wymiarów, a Qdrant oczekuje {EMBEDDING_DIM}")
        return

    point = PointStruct(
        id=uuid.uuid4().int >> 64,
        vector=vector.tolist(),
        payload=df.iloc[0].to_dict()
    )

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[point]
    )
    st.success("✅ Twoje dane zostały zapisane do Qdrant.")

# ---- Function to upload all users from CSV to Qdrant ----
def upload_all_users(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df[categorical_cols])
    for col in required_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[required_columns]

    vectors = df_encoded.values.astype(np.float32)
    if vectors.shape[1] != EMBEDDING_DIM:
        st.error(f"❌ Wektory mają {vectors.shape[1]} wymiarów, a Qdrant oczekuje {EMBEDDING_DIM}")
        return

    points = []
    for idx, (vec, (_, row)) in enumerate(zip(vectors, df.iterrows())):
        point = PointStruct(
            id=uuid.uuid4().int >> 64,
            vector=vec.tolist(),
            payload=row.to_dict()
        )
        points.append(point)

    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i:i+batch_size]
        )
    st.success(f"✅ Wysłano {len(points)} użytkowników do Qdrant.")

# ---- Przyciski w UI ----
st.button("📤 Zapisz mnie do Qdrant", on_click=upload_single_user, args=(person_df,))
st.button("📥 Wyślij wszystkich z CSV do Qdrant", on_click=upload_all_users, args=(all_df_raw,))
