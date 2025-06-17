# llm_system.py

from langchain_ollama import OllamaLLM

# Konfigurasi Model LLM lokal
LLM_MODEL_NAME = "mistral"  # bisa diganti: "llama3", "gemma", dsb

# Inisialisasi LLM
llm = OllamaLLM(model=LLM_MODEL_NAME)

# Fungsi untuk menjawab pertanyaan langsung dari LLM
def answer_directly_with_llm(question):
    prompt = f"""
Anda adalah asisten Islami terpercaya yang menjawab pertanyaan berdasarkan Al-Qur'an dan Hadis Shahih.
Berikan jawaban yang jelas, ringkas, dan sesuai dengan ajaran Islam.

Pertanyaan:
{question}

Jawaban Islami:
"""
    result = llm.invoke(prompt)
    return result.strip()

# Fungsi opsional untuk terjemahkan ke Bahasa Indonesia (jika jawaban dalam bahasa Inggris)
def translate_to_indonesian(text):
    prompt = f"""
Kamu adalah penerjemah profesional Bahasa Inggris ke Bahasa Indonesia.
Terjemahkan teks berikut ke Bahasa Indonesia dengan tata bahasa yang baik, alami, dan mudah dipahami seperti penulisan artikel Islami.

Teks asli:
{text}

Terjemahan Bahasa Indonesia yang bagus:
"""
    result = llm.invoke(prompt)
    return result.strip()

# --- MAIN ---
if __name__ == "__main__":
    print("🤖 Tanya Jawab Islami Langsung dengan LLM (tanpa retrieval)\n")
    while True:
        question = input("❓ Masukkan pertanyaan (atau ketik 'exit'): ")
        if question.lower() == "exit":
            break

        answer = answer_directly_with_llm(question)
        print("\n🕌 Jawaban LLM (original):\n")
        print(answer)

        translate = input("\n🌐 Terjemahkan ke Bahasa Indonesia? (y/n): ")
        if translate.lower() == "y":
            translation = translate_to_indonesian(answer)
            print("\n🇮🇩 Terjemahan Bahasa Indonesia:\n")
            print(translation)

def answer_with_llm(question: str) -> str:
    return answer_directly_with_llm(question)
