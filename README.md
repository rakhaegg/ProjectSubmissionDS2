# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Meskipun memiliki reputasi lulusan yang baik, institusi ini menghadapi masalah tingginya angka siswa yang tidak menyelesaikan studi (dropout), yang berdampak pada reputasi dan keberlanjutan finansial.

### Permasalahan Bisnis

* Tingginya persentase siswa yang drop out di tengah jalannya studi.
* Keterbatasan intervensi dini untuk siswa berisiko karena kurangnya sistem deteksi.
* Kurangnya integrasi data akademik dan finansial untuk pemantauan real-time.

### Cakupan Proyek

* Eksplorasi dan pembersihan data (EDA).
* Rekayasa fitur untuk menangkap sinyal akademik dan finansial.
* Pengembangan model machine learning untuk memprediksi risiko dropout.
* Pembuatan business dashboard di Tableau: "Early-Warning & Impact Center".
* Pembuatan prototype Streamlit untuk sistem prediksi online.

### Persiapan

**Sumber data:**

* Dataset "Students' Performance" (`students_performance.csv` / `dataset_final.csv` setelah preprocessing).

**Setup environment:**

```bash
git clone <repo_url>
cd <repo_folder>
pip install -r requirements.txt
```

`requirements.txt` mencakup:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
tableau-api-lib  # jika perlu integrasi API Tableau
shap
pingouin
```

## Business Dashboard

Dashboard "Early-Warning & Impact Center" di Tableau menampilkan:

* KPI ringkasan (Total siswa, Prediksi Dropout %, Recall, Precision).
* Distribusi Risk Level (No-risk / Yellow / Red).
* Komposisi finance\_risk dalam tiap risk\_level.

**Link akses dashboard:**
[https://tableau.server.domain/views/EarlyWarningImpactCenter](https://tableau.server.domain/views/EarlyWarningImpactCenter)

## Menjalankan Sistem Machine Learning

Prototype Streamlit memudahkan pengguna non-teknis menjalankan prediksi risiko.

```bash
# Setelah environment ter-setup di atas:
streamlit run app.py
```

Akses prototype di:
[https://jj-institut-dropout.streamlit.app](https://jj-institut-dropout.streamlit.app)

## Conclusion

Proyek ini berhasil membangun alur end-to-end dari pengumpulan data, preprocessing, model building (AUC ROC 0.92), hingga deployment visual analytics dan prototype online. Integrasi akademik dan finansial memungkinkan deteksi dini siswa berisiko sehingga intervensi dapat dilakukan lebih efektif.

### Rekomendasi Action Items

* Implementasi sistem peringatan dini (email/SMS) ketika `drop_score` ≥ 0.5.
* Program bimbingan akademik intensif untuk siswa dengan `pass_rate_sem1` < 0.5.
* Skema cicilan & beasiswa darurat bagi siswa `finance_risk` = 1.
* Audit kurikulum semester 1–2 untuk menyeimbangkan beban studi.
* Integrasi live data SIS & keuangan ke dashboard Tableau untuk refresh otomatis.
* Pelatihan staf konseling menggunakan SHAP explanations untuk interpretasi model.
