import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Prediksi Jumlah Balita Stunting",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. JUDUL APLIKASI
# =========================================================

st.title(
    "Prediksi Jumlah Balita Stunting "
    "5 Wilayah di Jawa Barat"
)

st.markdown(
    "Menggunakan algoritma **Random Forest Regressor** "
    "dan **Linear Regression**"
)

st.markdown(
    """
    Aplikasi ini menggunakan data 5 wilayah di Jawa Barat
    (Kabupaten Bogor, Kabupaten Indramayu, Kota Bandung,
    Kota Bekasi, dan Kota Depok) tahun **2018–2024** untuk
    membangun model prediksi dan melakukan estimasi jumlah
    balita stunting tahun **2025–2027**.
    """
)


# =========================================================
# 3. LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset_5_wilayah_2018_2024.csv"
    )

    df.columns = (
        df.columns
        .str.strip()
    )

    for kolom in df.columns:

        if df[kolom].dtype == "object" or pd.api.types.is_string_dtype(df[kolom]):

            df[kolom] = (
                df[kolom]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )

    kolom_numerik = [
        "kode_kabupaten_kota",
        "tahun",
        "jumlah_balita_stunting",
        "persentase_penduduk_miskin",
        "garis_kemiskinan",
        "persentase_sanitasi_layak",
        "jumlah_nakes_gizi"
    ]

    for kolom in kolom_numerik:

        if kolom in df.columns:

            df[kolom] = pd.to_numeric(
                df[kolom],
                errors="coerce"
            )

    df = df.dropna().copy()

    df["tahun"] = (
        df["tahun"]
        .astype(int)
    )

    df["kode_kabupaten_kota"] = (
        df["kode_kabupaten_kota"]
        .astype(int)
    )

    df = (
        df
        .sort_values(
            [
                "nama_kabupaten_kota",
                "tahun"
            ]
        )
        .reset_index(drop=True)
    )

    return df


# =========================================================
# 4. LOAD MODEL RANDOM FOREST
# =========================================================

@st.cache_resource
def load_rf_model():
    return joblib.load("random_forest_model.pkl")


# =========================================================
# 5. LOAD MODEL LINEAR REGRESSION
# =========================================================

@st.cache_resource
def load_lr_model():
    return joblib.load("linear_regression_model.pkl")


# =========================================================
# 6. LOAD HASIL PREDIKSI 2025–2027
# =========================================================

@st.cache_data
def load_future_prediction():

    df_future = pd.read_csv("hasil_prediksi_2025_2027.csv")
    df_future.columns = df_future.columns.str.strip()
    return df_future


# =========================================================
# 7. LOAD EVALUASI MODEL
# =========================================================

@st.cache_data
def load_evaluation():

    df_eval = pd.read_csv("hasil_evaluasi_model.csv")
    df_eval.columns = df_eval.columns.str.strip()
    return df_eval


# =========================================================
# 8. LOAD FEATURE IMPORTANCE
# =========================================================

@st.cache_data
def load_feature_importance():

    df_feature = pd.read_csv("feature_importance.csv")
    df_feature.columns = df_feature.columns.str.strip()
    return df_feature


# =========================================================
# 9. LOAD SEMUA DATA DAN MODEL
# =========================================================

try:

    df = load_data()
    rf = load_rf_model()
    lr = load_lr_model()
    df_future = load_future_prediction()
    df_eval = load_evaluation()
    df_feature = load_feature_importance()

except Exception as e:

    st.error(
        "Terjadi kesalahan saat memuat "
        "dataset, model, atau file hasil."
    )
    st.exception(e)
    st.stop()


# =========================================================
# 10. FITUR DAN TARGET
# =========================================================

fitur = [
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_nakes_gizi"
]

target = "jumlah_balita_stunting"


# =========================================================
# 11. VALIDASI KOLOM DATASET
# =========================================================

kolom_wajib = [
    "kode_kabupaten_kota",
    "nama_kabupaten_kota",
    "tahun",
    "jumlah_balita_stunting",
    "persentase_penduduk_miskin",
    "garis_kemiskinan",
    "persentase_sanitasi_layak",
    "jumlah_nakes_gizi"
]

kolom_hilang = [
    kolom
    for kolom in kolom_wajib
    if kolom not in df.columns
]

if kolom_hilang:

    st.error("Kolom berikut tidak ditemukan dalam dataset:")
    st.write(kolom_hilang)
    st.write("Kolom yang tersedia:")
    st.write(df.columns.tolist())
    st.stop()


# =========================================================
# 12. SIDEBAR
# =========================================================

st.sidebar.header("Informasi Penelitian")

daftar_wilayah = sorted(
    df["nama_kabupaten_kota"].unique()
)

st.sidebar.write("Wilayah: " + ", ".join(daftar_wilayah))
st.sidebar.write("Periode Data: 2018–2024")
st.sidebar.write("Periode Prediksi: 2025–2027")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write("Model Pembanding: Linear Regression")

st.sidebar.success(f"Total Data: {len(df)} baris")
st.sidebar.success(f"Rentang Tahun: {df['tahun'].min()}–{df['tahun'].max()}")
st.sidebar.success(f"Jumlah Wilayah: {len(daftar_wilayah)}")


# =========================================================
# 13. DATA AKTUAL 2018–2024
# =========================================================

df_aktual = df[
    (df["tahun"] >= 2018)
    &
    (df["tahun"] <= 2024)
].copy()

st.subheader("Data Aktual 5 Wilayah di Jawa Barat Tahun 2018–2024")
st.write(f"Jumlah data aktual: **{len(df_aktual)} baris**")

st.dataframe(
    df_aktual[kolom_wajib],
    use_container_width=True
)


# =========================================================
# 14. PREDIKSI DATA AKTUAL
# =========================================================

X_actual = df_aktual[fitur]

df_actual = df_aktual.copy()

df_actual["Prediksi Random Forest"] = rf.predict(X_actual)
df_actual["Prediksi Linear Regression"] = lr.predict(X_actual)

df_actual["Prediksi Random Forest"] = df_actual["Prediksi Random Forest"].round(0)
df_actual["Prediksi Linear Regression"] = df_actual["Prediksi Linear Regression"].round(0)


# =========================================================
# 15. HASIL PREDIKSI DATA AKTUAL
# =========================================================

st.subheader("Hasil Prediksi Data Aktual Tahun 2018–2024")

st.dataframe(
    df_actual[
        [
            "kode_kabupaten_kota",
            "nama_kabupaten_kota",
            "tahun",
            "jumlah_balita_stunting",
            "Prediksi Random Forest",
            "Prediksi Linear Regression"
        ]
    ],
    use_container_width=True
)


# =========================================================
# 16. PREDIKSI TAHUN 2025–2027
# =========================================================

st.subheader("Prediksi Jumlah Balita Stunting Tahun 2025–2027")

df_future = df_future.copy()

df_future["tahun"] = pd.to_numeric(df_future["tahun"], errors="coerce").astype(int)

df_future["Prediksi Random Forest"] = pd.to_numeric(
    df_future["Prediksi Random Forest"], errors="coerce"
)

df_future["Prediksi Linear Regression"] = pd.to_numeric(
    df_future["Prediksi Linear Regression"], errors="coerce"
)

df_future["Prediksi Random Forest"] = df_future["Prediksi Random Forest"].round(0)
df_future["Prediksi Linear Regression"] = df_future["Prediksi Linear Regression"].round(0)


# =========================================================
# 17. TABEL PREDIKSI MINIMALIS
# =========================================================

st.dataframe(
    df_future[
        [
            "nama_kabupaten_kota",
            "tahun",
            "Prediksi Random Forest",
            "Prediksi Linear Regression"
        ]
    ],
    use_container_width=True
)


# =========================================================
# 18. GRAFIK PREDIKSI 2025–2027 (VERSI DROPDOWN)
# =========================================================

st.subheader("Grafik Prediksi Tahun 2025–2027")

wilayah_terpilih_prediksi = st.selectbox(
    "Pilih Wilayah untuk Ditampilkan",
    sorted(df_future["nama_kabupaten_kota"].unique()),
    key="pilih_wilayah_prediksi"
)

data_wilayah = df_future[
    df_future["nama_kabupaten_kota"] == wilayah_terpilih_prediksi
].sort_values("tahun")

# Batas ATAS sumbu Y disamakan untuk semua wilayah
# (minimal sampai 10000) supaya semua grafik wilayah
# bisa dibandingkan.
batas_atas_prediksi = max(
    df_future["Prediksi Random Forest"].max(),
    df_future["Prediksi Linear Regression"].max(),
    10000
)

batas_atas_prediksi = batas_atas_prediksi * 1.05

# Batas bawah dihitung dari data wilayah yang dipilih saja
y_min_wilayah = min(
    data_wilayah["Prediksi Random Forest"].min(),
    data_wilayah["Prediksi Linear Regression"].min(),
    0
)

padding_bawah = abs(y_min_wilayah) * 0.1 + 200
batas_bawah_wilayah = y_min_wilayah - padding_bawah

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(
    data_wilayah["tahun"],
    data_wilayah["Prediksi Random Forest"],
    marker="o",
    label="Random Forest"
)

ax1.plot(
    data_wilayah["tahun"],
    data_wilayah["Prediksi Linear Regression"],
    marker="s",
    linestyle="--",
    label="Linear Regression"
)

ax1.set_title("Prediksi Jumlah Balita Stunting - " + str(wilayah_terpilih_prediksi))
ax1.set_xlabel("Tahun")
ax1.set_ylabel("Jumlah Balita Stunting")

ax1.set_ylim(batas_bawah_wilayah, batas_atas_prediksi)

ax1.legend()
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)
plt.close(fig1)


# =========================================================
# 19. GRAFIK TREN DATA AKTUAL DAN PREDIKSI (VERSI DROPDOWN)
# =========================================================

st.subheader("Grafik Tren Jumlah Balita Stunting Tahun 2018–2027")

wilayah_terpilih = st.selectbox(
    "Pilih Wilayah untuk Ditampilkan",
    sorted(df_aktual["nama_kabupaten_kota"].unique()),
    key="pilih_wilayah_tren"
)

data_aktual_terpilih = df_aktual[
    df_aktual["nama_kabupaten_kota"] == wilayah_terpilih
]

data_prediksi_terpilih = df_future[
    df_future["nama_kabupaten_kota"] == wilayah_terpilih
]

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    data_aktual_terpilih["tahun"],
    data_aktual_terpilih[target],
    marker="o",
    label="Data Aktual"
)

ax2.plot(
    data_prediksi_terpilih["tahun"],
    data_prediksi_terpilih["Prediksi Random Forest"],
    marker="s",
    linestyle="--",
    label="Prediksi Random Forest"
)

ax2.plot(
    data_prediksi_terpilih["tahun"],
    data_prediksi_terpilih["Prediksi Linear Regression"],
    marker="^",
    linestyle=":",
    label="Prediksi Linear Regression"
)

ax2.set_title(f"Tren Jumlah Balita Stunting - {wilayah_terpilih} Tahun 2018–2027")
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Jumlah Balita Stunting")
ax2.set_xticks(range(2018, 2028))
ax2.set_xticklabels(range(2018, 2028))
ax2.set_xlim(2018, 2027)
ax2.legend()
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)
plt.close(fig2)


# =========================================================
# 20. EVALUASI MODEL
# =========================================================

st.subheader("Evaluasi Model")

st.caption(
    "Evaluasi Random Forest Regressor "
    "dan Linear Regression berdasarkan "
    "hasil pelatihan model."
)

st.dataframe(
    df_eval.style.format({
        "R2 Score": "{:.4f}",
        "MAE": "{:,.2f}",
        "MSE": "{:,.2f}",
        "RMSE": "{:,.2f}"
    }),
    use_container_width=True
)


# =========================================================
# 21. FEATURE IMPORTANCE
# =========================================================

st.subheader("Feature Importance Random Forest")

st.dataframe(
    df_feature.style.format({
        "Importance": "{:.4f}"
    }),
    use_container_width=True
)


# =========================================================
# 22. GRAFIK FEATURE IMPORTANCE
# =========================================================

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.barh(
    df_feature["Fitur"],
    df_feature["Importance"]
)

ax3.set_xlabel("Nilai Importance")
ax3.set_ylabel("Variabel")
ax3.set_title("Feature Importance Random Forest")
ax3.invert_yaxis()
ax3.grid(axis="x", alpha=0.3)

st.pyplot(fig3)
plt.close(fig3)


# =========================================================
# 23. KESIMPULAN
# =========================================================

st.subheader("Kesimpulan Prediksi")

idx_rf = df_future["Prediksi Random Forest"].idxmax()
wilayah_rf = df_future.loc[idx_rf, "nama_kabupaten_kota"]
tahun_rf = int(df_future.loc[idx_rf, "tahun"])
nilai_rf = df_future.loc[idx_rf, "Prediksi Random Forest"]

idx_lr = df_future["Prediksi Linear Regression"].idxmax()
wilayah_lr = df_future.loc[idx_lr, "nama_kabupaten_kota"]
tahun_lr = int(df_future.loc[idx_lr, "tahun"])
nilai_lr = df_future.loc[idx_lr, "Prediksi Linear Regression"]

st.info(
    f"**Random Forest Regressor:** "
    f"Prediksi tertinggi diperkirakan terjadi "
    f"di {wilayah_rf} pada tahun {tahun_rf} "
    f"dengan jumlah sekitar {nilai_rf:,.0f} Jumlah Balita Stunting. "
    f"\n\n"
    f"**Linear Regression:** "
    f"Prediksi tertinggi diperkirakan terjadi "
    f"di {wilayah_lr} pada tahun {tahun_lr} "
    f"dengan jumlah sekitar {nilai_lr:,.0f} Jumlah Balita Stunting."
)


# =========================================================
# 24. INFORMASI AKHIR
# =========================================================

st.markdown("---")

st.caption(
    "Sistem prediksi jumlah balita stunting "
    "5 wilayah di Jawa Barat "
    "berdasarkan data tahun 2018–2024 "
    "dengan prediksi tahun 2025–2027."
)
