# SmartShopper Agent

AI Agent **Personalized SmartShopper Assistant** yang bisa:
1. Memberikan **rekomendasi produk** yang dipersonalisasi per user.
2. Menjawab **pertanyaan umum e-commerce** (pengiriman, pembelian, pembayaran, refund) lewat RAG.

Agent otomatis melakukan **routing** antara kedua tools di atas berdasarkan jenis pertanyaan user.

## Struktur Project

```
smartshopper-agent/
├── data/
│   ├── products.csv              # katalog produk
│   ├── users.csv                 # data user (termasuk max_budget)
│   ├── interactions.csv          # histori click/wishlist/purchase user
│   └── common_information.json   # dataset Common Information (sumber RAG)
├── scripts/
│   └── store_common_info.py      # generate embedding + simpan ke MongoDB Atlas
├── smartshopper_agent/
│   ├── agent.py                  # definisi Agent + routing instruction
│   ├── product_tool.py           # FunctionTool: rekomendasi produk
│   └── common_info_tool.py       # FunctionTool: RAG common information
├── recommender.py                # logic embedding & scoring produk
└── testing/
    ├── test_mongodb.py           # cek koneksi ke MongoDB Atlas
    ├── test_retrieval.py         # cek vector search langsung ke MongoDB
    ├── test_recommender.py       # cek scoring rekomendasi produk
    └── test_common_info.py       # cek tool common information end-to-end
```

## Cara Kerja

### 1. Common Information → MongoDB Atlas (Storing)

Dataset `data/common_information.json` berisi dokumen informasi umum e-commerce (shipping, purchase, payment, refund), masing-masing punya field: `info_id`, `category`, `title`, `content`, `keywords`.

`scripts/store_common_info.py` melakukan:
1. Load semua dokumen dari JSON.
2. Gabungkan `category + title + content + keywords` jadi satu teks (`embedding_text`), supaya konteksnya lebih kaya saat di-embed.
3. Encode teks itu dengan model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (mendukung Bahasa Indonesia) → vector `embedding`.
4. Simpan ke MongoDB Atlas, collection `common_information`, dengan `upsert` berdasarkan `info_id` (unique index) — jadi aman dijalankan ulang tanpa duplikat data.

**Strategi penyimpanan:** setiap dokumen disimpan sebagai satu record berisi field aslinya (`info_id`, `category`, `title`, `content`, `keywords`) **plus** `embedding_text` dan `embedding` (vector). Vector inilah yang nanti dicari lewat MongoDB Atlas Vector Search index (`common_info_vector_index`) saat ada query dari user.

### 2. Tools

- **Product Recommendation** (`product_tool.py`): hitung similarity antara embedding histori interaksi user dan embedding produk, filter sesuai `max_budget`, return top-N produk.
- **Common Information** (`common_info_tool.py`): implementasi RAG —
  1. Encode query user.
  2. `$vectorSearch` ke MongoDB Atlas untuk ambil dokumen paling relevan.
  3. Susun dokumen jadi context.
  4. Generate jawaban kontekstual dengan Gemini, dibatasi hanya boleh jawab dari context (retry otomatis kalau API sedang 503).

Kedua tools dibungkus dengan `google.adk.tools.FunctionTool` agar bisa dipanggil oleh Agent.

### 3. Agent Routing

`agent.py` mendaftarkan kedua tools ke satu `Agent` (Google ADK) dengan instruction yang menentukan kapan pakai tools mana:
- Pertanyaan soal produk/rekomendasi → `get_product_recommendations`
- Pertanyaan soal shipping/pembelian/refund/dll → `retrieve_common_information`
- Pertanyaan gabungan → pakai kedua tools
- Di luar topik SmartShopper → agent akan bilang itu di luar scope

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Buat file `.env` di root project:
   ```
   MONGODB_URI=<connection string MongoDB Atlas>
   MONGODB_DB=smartshopper
   MONGODB_COLLECTION=common_information
   GOOGLE_API_KEY=<Gemini API key>
   GEMINI_MODEL=gemini-3.7-flash
   ```
3. Simpan Common Information ke MongoDB:
   ```bash
   python scripts/store_common_info.py
   ```
4. Buat Vector Search Index bernama `common_info_vector_index` di MongoDB Atlas untuk field `embedding` (dimension sesuai model, similarity `cosine`).

## Testing

```bash
python testing/test_mongodb.py        # cek koneksi MongoDB
python testing/test_retrieval.py      # cek vector search langsung
python testing/test_recommender.py    # cek scoring rekomendasi produk
python testing/test_common_info.py    # cek tool common information (RAG penuh)
```

Untuk validasi **routing agent secara end-to-end**, jalankan agent lewat ADK dan coba beberapa tipe pertanyaan:
- Pertanyaan produk saja (misal: "rekomendasi produk untuk user U001")
- Pertanyaan umum saja (misal: "gimana cara refund barang?")
- Pertanyaan gabungan (misal: "rekomendasi produk untuk U001, dan gimana cara pengirimannya?")

## Tools yang Digunakan

- Google ADK (`google-adk`) — agent framework & FunctionTool
- Google Gemini (`google-genai`) — LLM generation
- MongoDB Atlas + Vector Search — storage & retrieval Common Information
- Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) — embedding
- Pandas / scikit-learn — product recommendation logic
