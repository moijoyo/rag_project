import os #comment
import re
import requests
from bs4 import BeautifulSoup

ARTIKEL_URLS = [
    'https://rumaysho.com/40094-benarkah-sudah-bersyahadat-ini-makna-syahadat-muhammad-rasulullah-yang-sebenarnya.html',
    'https://rumaysho.com/40077-taubat-nasuha-syarat-tanda-diterima-dan-bahaya-menunda-taubat.html?utm_source=chatgpt.com',
    'https://rumaysho.com/39295-dampak-buruk-maksiat-pelajaran-dari-ibnul-qayyim.html',
    'https://rumaysho.com/38990-berapa-lama-safar-yang-membolehkan-qashar-shalat-penjelasan-berdasarkan-hadits.html?utm_source=chatgpt.com',
]

# Buat folder penyimpanan artikel
ARTICLES_DIR = 'articles'
os.makedirs(ARTICLES_DIR, exist_ok=True)


# Fungsi untuk membersihkan nama file dari karakter ilegal
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)


# Fungsi utama scraping dan simpan artikel
def scrape_and_save(url):
    print(f"🔍 Mengambil artikel dari: {url}")
    try:
        res = requests.get(url)
        res.raise_for_status()
    except Exception as e:
        print(f"❌ Gagal mengakses {url}: {e}")
        return

    soup = BeautifulSoup(res.text, 'html.parser')

    # Ambil judul dan sanitasi untuk nama file
    title = soup.find('title').text.strip()
    safe_title = sanitize_filename(title).replace(" ", "_")

    # Ambil konten utama artikel
    paragraphs = soup.find_all(['p', 'li'])
    text = '\n'.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])

    if not text:
        print(f"⚠️ Artikel dari {url} tidak memiliki konten cukup.")
        return

    file_path = os.path.join(ARTICLES_DIR, f"{safe_title}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"✅ Disimpan: {file_path}")


# Jalankan scraping untuk semua URL
for url in ARTIKEL_URLS:
    scrape_and_save(url)

print("\n✅ Semua artikel berhasil diambil dan disimpan.")
