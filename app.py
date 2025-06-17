import streamlit as st
from rag_system import (
    load_articles,
    split_documents,
    build_faiss_index,
    run_rag_qa,
)
from llm_system import answer_with_llm
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Konfigurasi Model
LLM_MODEL_RAG = "qwen:1.8b-chat"
EMBEDDING_MODEL = "nomic-embed-text"

# Setup halaman
st.set_page_config(page_title="Perbandingan RAG vs LLM", layout="wide")
st.title("🤖 Perbandingan Tanya Jawab Islami: RAG vs LLM Murni")

# Caching untuk load sistem
@st.cache_resource(show_spinner="🔄 Memproses artikel... hanya pertama kali")
def load_vectorstore():
    docs = load_articles("articles")
    chunks = split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return build_faiss_index(chunks, embeddings)

vectorstore = load_vectorstore()
llm_rag = OllamaLLM(model=LLM_MODEL_RAG)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📋 Riwayat Pertanyaan")
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**{i}.** {chat['question']}")
    else:
        st.info("Belum ada pertanyaan.")

# Input pertanyaan
user_question = st.chat_input("Tulis pertanyaan Anda di sini...")

# Tampilkan riwayat
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📖 Jawaban Berdasarkan Artikel (RAG):**")
            st.markdown(chat["rag_answer"])
            if chat.get("rag_source"):
                st.caption(f"📁 Sumber artikel: `{chat['rag_source']}`")

        with col2:
            st.markdown("**💡 Jawaban Langsung dari LLM:**")
            st.markdown(chat["llm_answer"])

# Proses pertanyaan baru
if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("⏳ Sedang mencari jawaban..."):
        rag_answer, rag_source = run_rag_qa(user_question, vectorstore, llm_rag)
        llm_answer = answer_with_llm(user_question)

    with st.chat_message("assistant"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📖 Jawaban Berdasarkan Artikel (RAG):**")
            st.markdown(rag_answer)
            if rag_source:
                st.caption(f"📁 Sumber artikel: `{rag_source}`")

        with col2:
            st.markdown("**💡 Jawaban Langsung dari LLM:**")
            st.markdown(llm_answer)

    # Simpan riwayat
    st.session_state.chat_history.append({
        "question": user_question,
        "rag_answer": rag_answer,
        "rag_source": rag_source,
        "llm_answer": llm_answer,
    })
