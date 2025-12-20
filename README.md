# 📘 Facebook Post Interaction Prediction
*(Analisis Prediksi Interaksi Postingan Facebook Menggunakan Machine Learning & Deep Learning)*

## 👤 Informasi
- **Nama:** Shaffa Dwiaji Feryansyah Putra  
- **Repo:** https://github.com/ShaffaDwiaji/DS_Project_UAS
- **Video:** https://youtu.be/pZfH3Jg2JBw

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan prediksi popularitas konten sosial media (Regresi).
- Melakukan data preparation (Cleaning, Encoding, Scaling, & Leakage Prevention).
- Membangun 3 model: **Baseline** (Linear Regression), **Advanced** (Random Forest Tuned), **Deep Learning** (Multilayer Perceptron).
- Melakukan evaluasi komparatif menggunakan metrik RMSE & R2 Score.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Sulitnya memprediksi tingkat interaksi (engagement) sebuah postingan Facebook sebelum diterbitkan.
- Perlunya mengetahui faktor-faktor (fitur) apa yang paling mempengaruhi jumlah like, share, dan comment.

**Goals:**  
- Membangun model machine learning yang mampu memprediksi `Total Interactions` secara akurat.
- Membandingkan performa model sederhana (Linear) vs model kompleks (Neural Network) pada dataset berukuran kecil.

---
## 📁 Struktur Folder
```
project/
│
├── data/
│   ├── raw/                # Dataset asli (dataset_Facebook.csv)
│   └── processed/          # Data bersih (X_train_processed.csv, dll)
│
├── notebooks/              # Jupyter notebooks
│   └── 234311028_Shaffa Dwiaji F P_UAS_DATA_SCIENCE.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── best_model_lr.pkl   # Best Model
│   └── model_dl.h5         # Model Deep Learning
│
├── images/                 # Visualizations
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository (Facebook Metrics Data Set).
- **Jumlah Data:** 500 Baris.
- **Tipe:** Tabular (Regresi).

### Fitur Utama
| Fitur | Tipe Data | Deskripsi |
| ----- | --------- | --------- | 
| Page total likes | Numerik | Jumlah pengikut halaman saat posting. |
| Type | Kategorikal | "Jenis konten (Link, Photo, Status, Video)." |
| Post Month | Numerik | Bulan postingan dibuat (1-12). |
| Post Hour | Numerik | Jam postingan dibuat (0-23). |
| Paid | Biner | "Status postingan berbayar (0 = Tidak, 1 = Ya)." |
| Total Interactions | Target | Jumlah total Like + Comment + Share. |

---

# 4. 🔧 Data Preparation
- Cleaning: Mengisi missing values pada kolom Paid dengan modus, dan menghapus baris kosong pada like/share. 
- Leakage Prevention: Menghapus kolom like, share, dan comment dari fitur input agar tidak terjadi kebocoran data.  
- Tranformation: One-Hot Encoding & StandardScaler.
- Splitting (train/val/test): Data dibagi menjadi 3 bagian (70% Train, 15% Validation, 15% Test) untuk mencegah data leakage saat tuning model.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Linear Regression (Model regresi sederhana sebagai acuan dasar).
- **Model 2 – Advanced ML:** Random Forest Regressor (Ensemble learning dengan Hyperparameter Tuning menggunakan RandomizedSearchCV).
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP) dengan arsitektur 2 Hidden Layers (64 & 32 neuron) + Dropout. 

---

# 6. 🧪 Evaluation
**Metrik:** R-Squared (R2 Score), RMSE (Root Mean Squared Error), dan MAE (Mean Absolute Error).

### Hasil Singkat
| Model | MAE | RMSE | R2 Score | Catatan |
| ----- | --- | ---- | -------- | ------- | 
| Linear Regression | 35.37 | 59.97 | 0.93 | Model Terbaik (Error Terendah) |
| Random Forest | 66.97 | 138.6 | 0.66 | Mengalami kesulitan generalisasi |
| Deep Learning | 88.86 | 172.48 | 0.47 | Performa terendah (Overfitting/Underfitting) |

---

# 7. 🏁 Kesimpulan
- Model terbaik: Linear Regression
- Alasan: Dataset memiliki hubungan linear yang sangat kuat antara fitur (seperti jangkauan post) dengan target interaksi. Selain itu, karena ukuran dataset kecil (500 data), model kompleks seperti Deep Learning dan Random Forest cenderung gagal menangkap pola dengan baik dibandingkan model linear yang robust.
- Insight penting: Kompleksitas model tidak selalu menjamin hasil yang lebih baik. Pada data berukuran kecil dengan pola linear, model sederhana seringkali lebih unggul dan efisien.  

---

# 8. 🔮 Future Work
- [x] Tambah data  
- [x] Tuning model  
- [ ] Coba arsitektur DL lain  
- [x] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment Python 3.10 dan install dependencies:
```python
  pip install -r requirements.txt
```
