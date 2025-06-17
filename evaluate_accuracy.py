# evaluate_accuracy.py

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from rag_system import answer_with_rag
from llm_system import answer_with_llm

# Path file
PERTANYAAN_FILE = "data/pertanyaan.csv"
GROUND_TRUTH_FILE = "data/ground_truth.csv"
OUTPUT_CSV = "data/evaluation_results.csv"

# Buat folder data jika belum ada
os.makedirs("data", exist_ok=True)

# Baca pertanyaan dari CSV
pertanyaan_df = pd.read_csv(PERTANYAAN_FILE)
questions = pertanyaan_df["Pertanyaan"].dropna().tolist()

# Baca ground truth
ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
ground_truth_dict = dict(zip(ground_truth_df["Pertanyaan"], ground_truth_df["Jawaban_Benar"]))

print(f"📌 {len(questions)} pertanyaan ditemukan di '{PERTANYAAN_FILE}'\n")

results = []
correct_rag = 0
correct_llm = 0
total = 0

for idx, question in enumerate(questions, 1):
    print(f"❓ Pertanyaan #{idx}: {question}")

    ground_truth = ground_truth_dict.get(question)
    if not ground_truth:
        print(f"⚠️ Pertanyaan tidak ditemukan di ground truth: {question}")
        continue

    try:
        rag_answer = answer_with_rag(question)
    except Exception as e:
        print(f"⚠️ RAG error: {e}")
        rag_answer = ""

    try:
        llm_answer = answer_with_llm(question)
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        llm_answer = ""

    is_correct_rag = ground_truth.strip().lower() in rag_answer.strip().lower()
    is_correct_llm = ground_truth.strip().lower() in llm_answer.strip().lower()

    correct_rag += is_correct_rag
    correct_llm += is_correct_llm
    total += 1

    results.append([
        question,
        ground_truth,
        rag_answer,
        "✅" if is_correct_rag else "❌",
        llm_answer,
        "✅" if is_correct_llm else "❌"
    ])

# Simpan hasil evaluasi
with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Pertanyaan", "Jawaban_Benar", "Jawaban_RAG", "RAG_Correct", "Jawaban_LLM", "LLM_Correct"])
    writer.writerows(results)

# Hitung akurasi
if total == 0:
    print("\n❌ Tidak ada pertanyaan yang cocok di ground truth. Evaluasi dibatalkan.")
    exit()

rag_accuracy = correct_rag / total * 100
llm_accuracy = correct_llm / total * 100

# Tampilkan hasil akurasi
print("\n📊 Hasil Akurasi:")
print(f"🔸 RAG : {rag_accuracy:.2f}%")
print(f"🔸 LLM : {llm_accuracy:.2f}%")

# Visualisasi
plt.figure(figsize=(8, 5))
plt.bar(['RAG', 'LLM'], [rag_accuracy, llm_accuracy], color=['orange', 'skyblue'])
plt.ylim(0, 100)
plt.ylabel("Akurasi (%)")
plt.title("Perbandingan Akurasi: RAG vs LLM Lokal")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, acc in enumerate([rag_accuracy, llm_accuracy]):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("data/evaluation_chart.png")
print("📈 Grafik akurasi disimpan ke 'data/evaluation_chart.png'")
