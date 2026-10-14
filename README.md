# Prediksi Tingkat Kerusakan Bangunan Akibat Gempa Nepal

**Proyek Machine Learning: Structural Damage Classification (Kaggle)**

## 1. Latar belakang

Tingkat kerusakan bangunan akibat gempa tidak selalu sama. Jenis fondasi, struktur, ukuran bangunan, umur bangunan, serta kondisi lingkungan dapat berkaitan dengan tingkat kerusakan. Dengan data karakteristik bangunan dan label kerusakan historis, kita dapat melatih model untuk memperkirakan tingkat kerusakan bangunan lain.

**Rumusan masalah:** Bagaimana membangun model yang mampu mengklasifikasikan `damage_grade` berdasarkan fitur bangunan yang tersedia?

## 2. Tujuan

1. Memahami struktur dataset dan distribusi tingkat kerusakan.
2. Menyiapkan fitur numerik dan kategorikal agar dapat diproses model.
3. Melatih **CatBoostClassifier** sebagai model baseline.
4. Mengevaluasi model menggunakan **Macro F1-Score**, *classification report*, dan *confusion matrix*.
5. Menghasilkan file prediksi untuk setiap `building_id` pada data test dan mengunggahnya ke Kaggle.

## 3. Dataset

| File | Peran | Jumlah data |
|---|---|---:|
| `train (2).csv` | Data fitur dengan target `damage_grade` | 533.465 baris × 17 kolom |
| `test (1).csv` | Data fitur tanpa target; diprediksi oleh model | 228.629 baris × 16 kolom |
| `metadata.csv` | Keterangan atribut | — |
| `sample_submission.csv` | Referensi susunan kolom submission | — |

Contoh fitur:

- **Numerik:** `age_building`, `count_floors_pre_eq`, `height_ft_pre_eq`, dan `plinth_area_sq_ft`.
- **Kategorikal:** `land_surface_condition`, `foundation_type`, `roof_type`, `superstructure`, serta kode wilayah seperti `district_id`, `vdcmun_id`, dan `ward_id`.

Distribusi target tidak seimbang: kelas `Grade 5` paling banyak, sedangkan `Grade 1` paling sedikit. Oleh sebab itu, evaluasi tidak cukup hanya menggunakan accuracy.

## 4. Alur eksperimen pertama

```text
Dataset CSV
    ↓
Data loading dan Exploratory Data Analysis (EDA)
    ↓
Memisahkan X (fitur) dan y (damage_grade)
    ↓
Menentukan fitur kategorikal dan numerik
    ↓
Stratified train–validation split (80% : 20%)
    ↓
Training CatBoostClassifier
    ↓
Evaluasi Macro F1, classification report, confusion matrix
    ↓
Analisis feature importance
    ↓
Training model final dengan seluruh data latih
    ↓
Prediksi data test → konversi label menjadi angka 1–5
    ↓
Simpan submission.csv → unggah ke Kaggle
    ↓
Skor publik: 0,53101
```

### 4.1 Data loading dan EDA

Notebook membaca data menggunakan `pandas.read_csv()`. Kemudian dicek jumlah baris dan kolom, nama fitur, nilai kosong, duplikasi `building_id`, dan distribusi `damage_grade`. Langkah ini membantu memahami data serta menentukan cara mengevaluasi model.

### 4.2 Preprocessing: memisahkan fitur dan target

- `X`: fitur yang digunakan untuk pelatihan.
- `y`: label `damage_grade` dari data train.
- `X_test`: fitur dari data test yang akan diprediksi.

### 4.3 Membagi data untuk validasi

Data train dibagi secara **stratified** agar proporsi kelas relatif terjaga:

| Bagian | Persentase | Jumlah baris |
|---|---:|---:|
| Training | 80% | 426.772 |
| Validation | 20% | 106.693 |

Data validation digunakan untuk menilai performa model pada data yang tidak digunakan untuk memperbarui bobot model.

### 4.4 Melatih model CatBoost

CatBoost merupakan algoritma **gradient boosting berbasis decision tree**. Algoritma ini digunakan karena mampu menangani fitur kategorikal dan cocok diuji pada data tabular.

Parameter baseline pada notebook pertama:

```python
model = CatBoostClassifier(
    iterations=600,
    depth=7,
    learning_rate=0.06,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=100,
    thread_count=-1
)
```

- **`iterations`**: batas maksimum iterasi pembentukan pohon boosting.
- **`depth`**: kedalaman pohon.
- **`learning_rate`**: besarnya pembaruan model tiap iterasi.
- **`MultiClass`**: objective untuk klasifikasi lima kelas.
- **`random_seed`**: membantu konsistensi eksperimen.

Notebook menggunakan validation set dan *early stopping* untuk menghindari iterasi yang tidak lagi memperbaiki hasil validasi menurut metrik model.

### 4.5 Evaluasi

Kompetisi menggunakan **Macro F1-Score**, yaitu rata-rata F1 seluruh kelas tanpa membobotkan berdasarkan jumlah sampel:

\[
F1 = \frac{2\,Precision\,Recall}{Precision+Recall}
\]

\[
Macro\ F1 = \frac{F1_1+F1_2+F1_3+F1_4+F1_5}{5}
\]

- **Precision**: proporsi prediksi suatu kelas yang benar.
- **Recall**: proporsi data aktual suatu kelas yang berhasil dikenali.
- **Confusion matrix**: menunjukkan kelas mana yang sering tertukar.
