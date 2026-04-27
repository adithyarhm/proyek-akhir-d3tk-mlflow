# Proyek Akhir D3 Teknologi Komputer - Gas Forecasting dengan MLflow

Repositori ini memuat sistem prediksi kadar gas berbahaya (H2S dan SO2) secara *multi-output* berdasarkan data historis dari beberapa node sensor. Keseluruhan alur (*pipeline*) dari pemrosesan data mentah, rekayasa fitur (*feature engineering*), hingga pelatihan model, dan pelacakan eksperimen menggunakan **MLflow**.

## Arsitektur Sistem & Pipeline Machine Learning

Sistem ini didesain secara modular, dimana setiap tahap memiliki tanggung jawab yang spesifik:

1. **Pengumpulan & Penggabungan Data (`src/data/data_loader.py`)**
   - Memuat file-file CSV mentah untuk masing-masing node dari direktori `data/raw/`.
   - Menambahkan metadata spasial seperti ID node, lokasi, ketinggian, garis lintang (latitude), dan garis bujur (longitude).
   - Menyesuaikan tipe data (terutama format datetime) lalu menggabungkan semua data node menjadi satu DataFrame utama.

2. **Pra-pemrosesan & Rekayasa Fitur (`src/preprocess.py`)**
   - **Penanganan Outlier:** Menerapkan metode Interquartile Range (IQR) secara per-node untuk fitur `h2s`, `so2`, `hum` (kelembapan), `temp` (suhu), dan `windspeed` (kecepatan angin).
   - **Rekayasa Fitur:** Mengekstraksi fitur waktu (`hour`, `minute`, `minute_of_day`), membuat fitur lag selisih per node (`h2s_diff`, `so2_diff`), serta menghitung rasio spesifik (`gas_ratio_so2_h2s`).

3. **Pelatihan Model & Pelacakan MLflow (`src/train.py`)**
   - Data latih dibersihkan dari baris `NaN` akibat pembuatan fitur lag.
   - Melatih 4 arsitektur model berbeda untuk **masing-masing node**:
     - Linear Regression
     - Random Forest Regressor
     - XGBoost
     - Support Vector Regressor (SVR)
   - Seluruh model menggunakan metode `MultiOutputRegressor` untuk memprediksi `h2s` dan `so2` sekaligus.
   - Semua metrik uji (seperti MAE, RMSE, R2) disimpan serta dilacak (*logged*) menggunakan MLflow di bawah nama eksperimen `gas-belerang-forecasting`.

4. **Orkestrasi Utama (`main.py`)**
   - Mengontrol urutan seluruh *pipeline*, dari membaca data mentah, memanggil pemrosesan, pelatihan, sampai ke pencetakan metrik performa akhir. Data hasil pemrosesan juga disimpan di `data/processed/node_combined_final.csv`.

## Struktur Direktori

```text
proyek-akhir-d3tk-mlflow/
│
├── data/
│   ├── raw/                  # Kumpulan file CSV data mentah dari tiap node
│   └── processed/            # File hasil pra-pemrosesan & rekayasa fitur
├── notebooks/                # Jupyter notebooks untuk eksplorasi dan percobaan awal
├── saved-models/             # Lokasi alternatif untuk menyimpan model tersertifikasi/terbaik
├── src/
│   ├── config.py             # Pusat konfigurasi (path direktori, kolom data, hyperparameter)
│   ├── preprocess.py         # Skrip pra-pemrosesan (penanganan outlier & feature engineering)
│   ├── train.py              # Logika utama melatih model (per-node) dan MLflow logging
│   ├── data/
│   │   └── data_loader.py    # Logika memuat data CSV mentah
│   └── models/               # (Opsional) Tempat defisini kelas/fungsi model terpisah
│
├── main.py                   # Skrip entry point untuk menjalankan keseluruhan sistem
├── README.md                 # Dokumentasi proyek (file ini)
└── mlruns/                   # Folder artefak lokal backend MLflow
```

## Cara Menjalankan

1. **Persiapan Data**
   Pastikan file-file dataset mentah tiap node (`node1.csv` - `node6.csv`) telah diletakkan di dalam folder `data/raw/`.

2. **Menjalankan Pipeline ML**
   Eksekusi file utama untuk memulai pipeline data hingga pelatihan model:
   ```bash
   python main.py
   ```
   Skrip ini secara otomatis memproses data, melatih semua model (LR, RFR, XGBoost, SVR) pada setiap node, dan mencetak metrik performa akhir ke dalam konsol terminal.

3. **Memantau Eksperimen dengan MLflow**
   Untuk melihat detail eksperimen, membandingkan performa model dari berbagai *run*, dan memeriksa model yang tersimpan, jalankan MLflow server lokal:
   ```bash
   mlflow ui
   ```
   Lalu buka tautan [http://localhost:5000](http://localhost:5000) di browser web Anda.

## Keputusan Desain Utama

- **Konfigurasi Tersentralisasi:** Semua *path* direktori, daftar fitur input dan target, serta hiperparameter ML diatur secara terpusat melalui `src/config.py`.
- **Pemrosesan Terisolasi Berdasarkan Node:** Kalkulasi *outlier capping* dan rekayasa fitur selisih waktu (*lag features*) dilakukan secara independen untuk tiap node demi menghindari pencampuran karakteristik tren temporal antar node.
- **Pendekatan *Multi-Output Regression*:** Dikarenakan tujuan utamanya adalah memprediksi dua variabel target secara simultan (`h2s` dan `so2`), `MultiOutputRegressor` dari Scikit-Learn disematkan pada keempat arsitektur algoritma.
