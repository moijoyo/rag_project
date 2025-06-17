import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

# Konfigurasi Model
LLM_MODEL_NAME = "qwen:1.8b-chat"  # Bisa bahasa Indonesia & Arab
EMBEDDING_MODEL_NAME = "nomic-embed-text"

# 1. Load semua artikel
def load_articles(folder_path="articles"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": filename}))
    return docs

# 2. Bagi dokumen jadi chunk
def split_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# 3. Bangun FAISS index
def build_faiss_index(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)

# 4. Jalankan RAG QA system dengan prompt Islami
def run_rag_qa(question, vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    if not docs:
        return "Maaf, tidak ditemukan artikel yang relevan untuk menjawab pertanyaan ini.", None

    context = "\n\n".join([f"({i+1}) {doc.page_content.strip()}" for i, doc in enumerate(docs)])

    # Prompt tegas: Jawab hanya dari artikel
    prompt = f"""
Anda adalah asisten Islami terpercaya.

Tugas Anda adalah menjawab pertanyaan **hanya berdasarkan informasi dari artikel yang diberikan** di bawah ini. Jangan gunakan pengetahuan umum Anda. Jika jawabannya tidak ditemukan di artikel, katakan: "Maaf, jawaban tidak ditemukan dalam artikel yang tersedia."

📌 Pertanyaan:
{question}

📄 Artikel-artikel:
{context}

💬 Jawaban:
"""

    result = llm.invoke(prompt).strip()

    # Filter jawaban tidak relevan
    if not result or "tidak ditemukan" in result.lower() or len(result.split()) < 5:
        return "Maaf, jawaban tidak ditemukan dalam artikel yang tersedia.", None

    source_filename = docs[0].metadata.get("source", "unknown")
    return result, source_filename



# Untuk API atau evaluasi otomatis
def answer_with_rag(question: str) -> str:
    docs = load_articles("articles")
    chunks = split_documents(docs)
    vectorstore = build_faiss_index(chunks, OllamaEmbeddings(model=EMBEDDING_MODEL_NAME))
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    answer, _ = run_rag_qa(question, vectorstore, llm)
    return answer

# CLI Test
if __name__ == "__main__":
    print("🔍 Memuat artikel dari folder 'articles'...")
    docs = load_articles()
    if not docs:
        print("⚠️ Tidak ada artikel ditemukan.")
        exit()
    print(f"✅ {len(docs)} artikel dimuat")

    print("📚 Membagi dokumen jadi potongan...")
    chunks = split_documents(docs)

    print("📦 Membuat index FAISS...")
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = build_faiss_index(chunks, embedding_model)

    print(f"🤖 Memulai sesi tanya jawab dengan RAG + {LLM_MODEL_NAME} via Ollama...")
    llm = OllamaLLM(model=LLM_MODEL_NAME)

    while True:
        question = input("\n❓ Masukkan pertanyaan (atau ketik 'exit'): ")
        if question.lower() == "exit":
            break

        answer, sumber = run_rag_qa(question, vectorstore, llm)
        print("\n📄 Jawaban Berdasarkan Artikel:")
        print(answer)
        if sumber:
            print(f"\n📁 Sumber: {sumber}")
