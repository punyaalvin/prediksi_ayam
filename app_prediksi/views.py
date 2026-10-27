from django.shortcuts import render
import joblib
import pandas as pd
import os

# ==========================================
# Ambil path folder project
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# Load model dan Label Encoder
# ==========================================
model = joblib.load(os.path.join(BASE_DIR, 'model_decision_tree.pkl'))

le_hari = joblib.load(os.path.join(BASE_DIR, 'le_hari.pkl'))
le_cuaca = joblib.load(os.path.join(BASE_DIR, 'le_cuaca.pkl'))
le_promosi = joblib.load(os.path.join(BASE_DIR, 'le_promosi.pkl'))
le_jenis_promosi = joblib.load(os.path.join(BASE_DIR, 'le_jenis_promosi.pkl'))
le_hari_libur = joblib.load(os.path.join(BASE_DIR, 'le_hari_libur.pkl'))


# ==========================================
# Halaman Home
# ==========================================
def home(request):

    laporan = None

    if request.method == "POST":

        hari = request.POST.get("hari")
        cuaca = request.POST.get("cuaca")
        promosi = request.POST.get("promosi")
        jenis_promosi = request.POST.get("jenis_promosi")
        hari_libur = request.POST.get("hari_libur")
        durasi_buka = request.POST.get("durasi_buka")

        try:

            # ==========================
            # Validasi Durasi Buka
            # ==========================
            durasi_buka = int(durasi_buka)

            if durasi_buka not in [8, 9, 10, 11]:
                raise Exception("Durasi buka hanya boleh 8, 9, 10 atau 11 jam.")

            # ==========================
            # Jika tidak ada promosi
            # ==========================
            if promosi == "Tidak":
                jenis_promosi = "Tidak Ada"

            # ==========================
            # Label Encoding
            # ==========================
            hari_encoded = le_hari.transform([hari])[0]
            promosi_encoded = le_promosi.transform([promosi])[0]
            jenis_promosi_encoded = le_jenis_promosi.transform([jenis_promosi])[0]
            hari_libur_encoded = le_hari_libur.transform([hari_libur])[0]

            # =====================================================
            # Jika Cuaca Belum Diketahui
            # =====================================================
            if cuaca == "":

                cuaca_cerah = le_cuaca.transform(["Cerah"])[0]
                cuaca_hujan = le_cuaca.transform(["Hujan"])[0]

                data_cerah = pd.DataFrame({

                    "Hari": [hari_encoded],
                    "Cuaca": [cuaca_cerah],
                    "Promosi": [promosi_encoded],
                    "Jenis_Promosi": [jenis_promosi_encoded],
                    "Hari_Libur": [hari_libur_encoded],
                    "Durasi_Buka": [durasi_buka]

                })

                data_hujan = data_cerah.copy()
                data_hujan["Cuaca"] = cuaca_hujan

                prediksi_cerah = round(model.predict(data_cerah)[0])
                prediksi_hujan = round(model.predict(data_hujan)[0])

                laporan = {

                    "hari": hari,
                    "cuaca": "Belum Diketahui",
                    "promosi": promosi,
                    "jenis_promosi": jenis_promosi,
                    "hari_libur": hari_libur,
                    "durasi_buka": durasi_buka,

                    "prediksi_cerah": prediksi_cerah,
                    "prediksi_hujan": prediksi_hujan,

                    "keterangan":
                    "Karena kondisi cuaca belum diketahui, sistem menampilkan dua kemungkinan hasil prediksi berdasarkan kondisi cuaca cerah dan hujan."

                }

            # =====================================================
            # Jika Cuaca Sudah Diketahui
            # =====================================================
            else:

                cuaca_encoded = le_cuaca.transform([cuaca])[0]

                data_input = pd.DataFrame({

                    "Hari": [hari_encoded],
                    "Cuaca": [cuaca_encoded],
                    "Promosi": [promosi_encoded],
                    "Jenis_Promosi": [jenis_promosi_encoded],
                    "Hari_Libur": [hari_libur_encoded],
                    "Durasi_Buka": [durasi_buka]

                })

                prediksi = round(model.predict(data_input)[0])

                laporan = {

                    "hari": hari,
                    "cuaca": cuaca,
                    "promosi": promosi,
                    "jenis_promosi": jenis_promosi,
                    "hari_libur": hari_libur,
                    "durasi_buka": durasi_buka,

                    "prediksi": prediksi,

                    "keterangan":
                    "Prediksi dihasilkan berdasarkan data masukan pengguna menggunakan algoritma Decision Tree Regressor."

                }

        except Exception as e:
            print("ERROR :", e)

            laporan = {
                "error": str(e)
            }

    return render(
        request,
        "app_prediksi/home.html",
        {
            "laporan": laporan
        }
    )