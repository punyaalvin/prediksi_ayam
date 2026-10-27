# Sistem Prediksi Penjualan Ayam Goreng (Regresi)
# Metode : Decision Tree Regressor
# Dibuat oleh : Alvin Adianto

# ===========================
# Import Library
# ===========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# ===========================
# Membaca Dataset
# ===========================
data = pd.read_excel(r"C:\Users\ALVIN\prediksi_ayam\Data Ayam regresi.xlsx")

print("=== Data Awal ===")
print(data.head(), "\n")

print("=== Cek Missing Value ===")
print(data.isnull().sum(), "\n")

# ===========================
# Menghitung Durasi Buka
# ===========================
data["Jam_Buka"] = data["Jam_Buka"].astype(int)
data["Jam_Tutup"] = data["Jam_Tutup"].astype(int)

data["Durasi_Buka"] = data["Jam_Tutup"] - data["Jam_Buka"]

# ===========================
# Label Encoding
# ===========================
le_hari = LabelEncoder()
le_cuaca = LabelEncoder()
le_promosi = LabelEncoder()
le_jenis_promosi = LabelEncoder()
le_hari_libur = LabelEncoder()

data["Hari"] = le_hari.fit_transform(data["Hari"])
data["Cuaca"] = le_cuaca.fit_transform(data["Cuaca"])
data["Promosi"] = le_promosi.fit_transform(data["Promosi"])
data["Jenis_Promosi"] = le_jenis_promosi.fit_transform(data["Jenis_Promosi"])
data["Hari_Libur"] = le_hari_libur.fit_transform(data["Hari_Libur"])

# ===========================
# Menentukan Variabel
# ===========================
x = data[
    [
        "Hari",
        "Cuaca",
        "Promosi",
        "Jenis_Promosi",
        "Hari_Libur",
        "Durasi_Buka",
    ]
]

y = data["Jumlah_Terjual"]

# ===========================
# Membagi Data Training & Testing
# ===========================
X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

print("Jumlah data train :", len(X_train))
print("Jumlah data test  :", len(X_test), "\n")

# ===========================
# Membuat / Memanggil Model
# ===========================
model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42,
)

if os.path.exists("model_decision_tree.pkl"):

    model = joblib.load("model_decision_tree.pkl")

    print("Status Model : Dimuat dari file (tanpa training ulang)\n")

else:

    model.fit(X_train, Y_train)

    joblib.dump(model, "model_decision_tree.pkl")

    joblib.dump(le_hari, "le_hari.pkl")
    joblib.dump(le_cuaca, "le_cuaca.pkl")
    joblib.dump(le_promosi, "le_promosi.pkl")
    joblib.dump(le_jenis_promosi, "le_jenis_promosi.pkl")
    joblib.dump(le_hari_libur, "le_hari_libur.pkl")

    print("Status Model : Dilatih dan disimpan menjadi model_decision_tree.pkl\n")

    # ===========================
# ===========================
# Evaluasi Model
# ===========================

Y_pred = model.predict(X_test)

# ===========================
# Perhitungan Manual MSE
# Rumus:
# MSE = (1/n) * Σ(y_true - y_pred)^2
# ===========================

mse_manual = np.mean((Y_test - Y_pred) ** 2)

# ===========================
# Perhitungan Manual R²
# Rumus:
# R² = 1 - (Σ(y - ŷ)^2 / Σ(y - ȳ)^2)
# ===========================

ss_res = np.sum((Y_test - Y_pred) ** 2)

ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)

r2_manual = 1 - (ss_res / ss_tot)

# ===========================
# Perhitungan Menggunakan Scikit-Learn
# ===========================

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("=== Evaluasi Model ===")

print("MSE Manual               :", round(mse_manual,2))
print("MSE Scikit-Learn         :", round(mse,2))

print()

print("R² Manual                :", round(r2_manual,2))
print("R² Scikit-Learn          :", round(r2,2))

print("============================================")

# ===========================
# Menyimpan Hasil Prediksi Dataset
# ===========================
data_hasil = data.copy()

data_hasil["Prediksi_Model"] = model.predict(x).round().astype(int)

# Mengembalikan data hasil encoding
data_hasil["Hari"] = le_hari.inverse_transform(data_hasil["Hari"])
data_hasil["Cuaca"] = le_cuaca.inverse_transform(data_hasil["Cuaca"])
data_hasil["Promosi"] = le_promosi.inverse_transform(data_hasil["Promosi"])
data_hasil["Jenis_Promosi"] = le_jenis_promosi.inverse_transform(
    data_hasil["Jenis_Promosi"]
)
data_hasil["Hari_Libur"] = le_hari_libur.inverse_transform(
    data_hasil["Hari_Libur"]
)

# Menyimpan ke Excel
data_hasil.to_excel("hasil_prediksi.xlsx", index=False)

print("File hasil_prediksi.xlsx berhasil dibuat.\n")

# ===========================
# Visualisasi Decision Tree
# ===========================
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=list(x.columns),
    filled=True,
    rounded=True,
)

plt.title("Visualisasi Pohon Keputusan - Prediksi Penjualan Ayam")

plt.show()

# ===========================
# Input Data Prediksi
# ===========================
print("============================================")
print("     SISTEM PREDIKSI PENJUALAN AYAM")
print("============================================")

hari_input = input("Masukkan Hari : ").strip().capitalize()

cuaca_input = input(
    "Masukkan Cuaca (Cerah/Hujan) [Boleh dikosongkan] : "
).strip().capitalize()

promosi_input = input(
    "Apakah ada promosi hari ini? (Ya/Tidak) : "
).strip().capitalize()

if promosi_input.lower() == "ya":

    jenis_promosi_input = input(
        "Jenis Promosi (Minuman/Extra Saus) : "
    ).strip().title()

else:

    jenis_promosi_input = "Tidak Ada"

hari_libur_input = input(
    "Apakah hari ini hari libur? (Ya/Tidak) : "
).strip().capitalize()

durasi_buka_input = input(
    "Berapa lama toko dibuka? (8/9/10/11 jam) : "
)

# ===========================
# Validasi Durasi Buka
# ===========================
try:

    durasi_buka_encoded = int(durasi_buka_input)

    if durasi_buka_encoded not in [8, 9, 10, 11]:
        print("Durasi buka hanya boleh 8, 9, 10, atau 11 jam!")
        exit()

except ValueError:

    print("Durasi buka harus berupa angka!")
    exit()

# ===========================
# Encoding Data Input
# ===========================
hari_encoded = le_hari.transform([hari_input])[0]
promosi_encoded = le_promosi.transform([promosi_input])[0]
jenis_promosi_encoded = le_jenis_promosi.transform([jenis_promosi_input])[0]
hari_libur_encoded = le_hari_libur.transform([hari_libur_input])[0]

# ==================================================
# Jika Cuaca Belum Diketahui
# ==================================================
if cuaca_input == "":

    cuaca_cerah = le_cuaca.transform(["Cerah"])[0]
    cuaca_hujan = le_cuaca.transform(["Hujan"])[0]

    data_cerah = pd.DataFrame({
        "Hari": [hari_encoded],
        "Cuaca": [cuaca_cerah],
        "Promosi": [promosi_encoded],
        "Jenis_Promosi": [jenis_promosi_encoded],
        "Hari_Libur": [hari_libur_encoded],
        "Durasi_Buka": [durasi_buka_encoded]
    })

    data_hujan = data_cerah.copy()
    data_hujan["Cuaca"] = cuaca_hujan

    pred_cerah = model.predict(data_cerah)[0]
    pred_hujan = model.predict(data_hujan)[0]

    print("\n")
    print("===================================================")
    print("        LAPORAN HASIL PREDIKSI PENJUALAN")
    print("===================================================")

    print("\nDATA INPUT")
    print("---------------------------------------------------")
    print(f"Hari            : {hari_input}")
    print("Cuaca           : Belum Diketahui")
    print(f"Promosi         : {promosi_input}")
    print(f"Jenis Promosi   : {jenis_promosi_input}")
    print(f"Hari Libur      : {hari_libur_input}")
    print(f"Durasi Buka     : {durasi_buka_encoded} Jam")

    print("\nHASIL PREDIKSI")
    print("---------------------------------------------------")
    print(f"Jika Cuaca Cerah : {round(pred_cerah)} Porsi Ayam")
    print(f"Jika Cuaca Hujan : {round(pred_hujan)} Porsi Ayam")

    print("\nKETERANGAN")
    print("---------------------------------------------------")
    print("Karena kondisi cuaca belum diketahui, sistem")
    print("menampilkan dua kemungkinan hasil prediksi")
    print("berdasarkan kondisi cuaca cerah dan hujan.")

    print("===================================================")

# ==================================================
# Jika Cuaca Diketahui
# ==================================================
else:

    cuaca_encoded = le_cuaca.transform([cuaca_input])[0]

    data_input = pd.DataFrame({
        "Hari": [hari_encoded],
        "Cuaca": [cuaca_encoded],
        "Promosi": [promosi_encoded],
        "Jenis_Promosi": [jenis_promosi_encoded],
        "Hari_Libur": [hari_libur_encoded],
        "Durasi_Buka": [durasi_buka_encoded]
    })

    prediksi = model.predict(data_input)[0]

    print("\n")
    print("===================================================")
    print("        LAPORAN HASIL PREDIKSI PENJUALAN")
    print("===================================================")

    print("\nDATA INPUT")
    print("---------------------------------------------------")
    print(f"Hari            : {hari_input}")
    print(f"Cuaca           : {cuaca_input}")
    print(f"Promosi         : {promosi_input}")
    print(f"Jenis Promosi   : {jenis_promosi_input}")
    print(f"Hari Libur      : {hari_libur_input}")
    print(f"Durasi Buka     : {durasi_buka_encoded} Jam")

    print("\nHASIL PREDIKSI")
    print("---------------------------------------------------")
    print(f"Prediksi Penjualan : {round(prediksi)} Porsi Ayam")

    print("\nKETERANGAN")
    print("---------------------------------------------------")
    print(
        f"Berdasarkan data yang dimasukkan, sistem "
        f"memprediksi jumlah penjualan ayam sebanyak "
        f"{round(prediksi)} porsi. Hasil prediksi ini "
        f"dapat dijadikan sebagai acuan dalam menentukan "
        f"jumlah produksi ayam."
    )

    print("===================================================")
    