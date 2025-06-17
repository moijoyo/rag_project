import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from rag_system import answer_with_rag
from llm_system import answer_with_llm

# --- Setup ---
st.set_page_config(page_title="Evaluasi RAG vs LLM", layout="wide")
st.title("📊 Evaluasi Sistem Tanya Jawab: RAG vs LLM Lokal")

QUESTIONS_FILE = "data/questions.txt"
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load pertanyaan ---
if not os.path.exists(QUESTIONS_FILE):
    st.error(f"Tidak ditemukan file: {QUESTIONS_FILE}")
    st.stop()

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

st.success(f"Ditemukan {len(questions)} pertanyaan.")

# --- Evaluasi ---
results = []

with st.spinner("Sedang menjawab semua pertanyaan..."):
    for idx, question in enumerate(questions, 1):
        st.write(f"**#{idx}. {question}**")

        # Jawaban RAG
        try:
            start_rag = time.time()
            rag_answer = answer_with_rag(question)
            time_rag = time.time() - start_rag
        except Exception as e:
            rag_answer = f"⚠️ Error: {e}"
            time_rag = 0

        # Jawaban LLM
        try:
            start_llm = time.time()
            llm_answer = answer_with_llm(question)
            time_llm = time.time() - start_llm
        except Exception as e:
            llm_answer = f"⚠️ Error: {e}"
            time_llm = 0

        # Similarity Score
        if rag_answer and llm_answer and "⚠️" not in rag_answer and "⚠️" not in llm_answer:
            embeddings = similarity_model.encode([rag_answer, llm_answer], convert_to_tensor=True)
            similarity_score = float(util.cos_sim(embeddings[0], embeddings[1]))
        else:
            similarity_score = 0.0

        results.append({
            "Pertanyaan": question,
            "Jawaban_RAG": rag_answer.strip(),
            "Waktu_RAG": time_rag,
            "Jawaban_LLM": llm_answer.strip(),
            "Waktu_LLM": time_llm,
            "Similarity": similarity_score
        })

# --- Tampilkan Tabel ---
df = pd.DataFrame(results)
st.subheader("📋 Hasil Evaluasi")
st.dataframe(df[["Pertanyaan", "Waktu_RAG", "Waktu_LLM", "Similarity"]], use_container_width=True)

# --- Akurasi & Visualisasi ---
rag_accuracy = sum(bool(ans and "⚠️" not in ans) for ans in df["Jawaban_RAG"]) / len(df)
llm_accuracy = sum(bool(ans and "⚠️" not in ans) for ans in df["Jawaban_LLM"]) / len(df)
avg_rag_time = df["Waktu_RAG"].mean()
avg_llm_time = df["Waktu_LLM"].mean()
avg_similarity = df["Similarity"].mean()

# Grafik 1: Akurasi
st.subheader("📈 Akurasi Jawaban")
st.bar_chart(pd.DataFrame({
    "RAG": [rag_accuracy * 100],
    "LLM": [llm_accuracy * 100]
}).T.rename(columns={0: "Akurasi (%)"}))

# Grafik 2: Waktu Eksekusi
st.subheader("⏱️ Rata-Rata Waktu Eksekusi")
st.bar_chart(pd.DataFrame({
    "RAG": [avg_rag_time],
    "LLM": [avg_llm_time]
}).T.rename(columns={0: "Detik"}))

# Grafik 3: Similarity
st.subheader("🔍 Rata-Rata Kemiripan Semantik (Cosine Similarity)")
st.metric("Similarity Score", f"{avg_similarity:.4f}", delta=None)

# --- Simpan ke CSV ---
if st.button("💾 Simpan hasil ke CSV"):
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/evaluation_results.csv", index=False, encoding="utf-8")
    st.success("Hasil berhasil disimpan ke data/evaluation_results.csv")

