# streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul aplikasi
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("📊 Prediksi Mahasiswa Dropout vs Graduate")

@st.cache_data
def load_cleaned_data(path="students_performance_cleaned.csv"):
    """
    Load dataset yang telah dibersihkan (cleaned) untuk
    mengambil daftar nilai unik fitur kategorikal.
    """
    df = pd.read_csv(path)
    # Kita hanya simpan baris dengan Status = 'Dropout' atau 'Graduate'
    df = df[df["Status"].isin(["Dropout", "Graduate"])]
    return df

@st.cache_resource
def load_model(path="students_performance_logreg.sav"):
    """
    Load model pipeline (LogisticRegression terbaru) yang sudah disimpan.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# Muat data dan model
df_cleaned = load_cleaned_data()
model_pipeline = load_model()

st.markdown(
    """
    <div style="text-align: left; margin-bottom: 1rem;">
    Aplikasi ini memprediksi apakah seorang mahasiswa berpotensi <b>Dropout</b> atau <b>Graduate</b> 
    berdasarkan fitur akademik dan demografis. Silakan isi detail berikut:
    </div>
    """,
    unsafe_allow_html=True
)

# Daftar fitur kategorikal dan numerik sesuai pipeline
cat_features = [
    "Marital_status",
    "Application_mode",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Displaced",
    "Educational_special_needs",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "International",
    "Course",
]

num_features = [
    "Admission_grade",
    "Previous_qualification_grade",
    "Age_at_enrollment",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_1st_sem_credited",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_without_evaluations",
]

st.sidebar.header("🔧 Input Fitur Kategorikal")
inputs = {}

# --- Ambil daftar nilai unik langsung dari df_cleaned untuk setiap fitur kategorikal --- #

for col in cat_features:
    options = sorted(df_cleaned[col].unique())
    choice = st.sidebar.selectbox(f"{col.replace('_', ' ').capitalize()}", options)
    inputs[col] = choice

st.sidebar.header("🔧 Input Fitur Numerik")

# 1. Admission_grade
admission_grade = st.sidebar.slider(
    "Admission grade", min_value=95.0, max_value=190.0, value=80.0, step=0.1
)
inputs["Admission_grade"] = float(admission_grade)

# 2. Previous_qualification_grade
prev_qual_grade = st.sidebar.slider(
    "Previous qualification grade", min_value=95.0, max_value=190.0, value=130.0, step=0.1
)
inputs["Previous_qualification_grade"] = float(prev_qual_grade)

# 3. Age_at_enrollment
age = st.sidebar.slider(
    "Age at enrollment", min_value=17, max_value=70, value=22, step=1
)
inputs["Age_at_enrollment"] = int(age)

# 4. Curricular_units_1st_sem_grade
cu1_grade = st.sidebar.slider(
    "Curricular units 1st sem (grade)", 
    min_value=0.0, max_value=18.8, value=6.0, step=0.1
)
inputs["Curricular_units_1st_sem_grade"] = float(cu1_grade)

# 5. Curricular_units_2nd_sem_grade
cu2_grade = st.sidebar.slider(
    "Curricular units 2nd sem (grade)", 
    min_value=0.0, max_value=18.57, value=9.0, step=0.1
)
inputs["Curricular_units_2nd_sem_grade"] = float(cu2_grade)

# 6. Curricular_units_1st_sem_credited
cred1 = st.sidebar.slider(
    "Units 1st sem (credited)", min_value=0, max_value=20, value=6, step=1
)
inputs["Curricular_units_1st_sem_credited"] = int(cred1)

# 7. Curricular_units_1st_sem_enrolled
enr1 = st.sidebar.slider(
    "Units 1st sem (enrolled)", min_value=0, max_value=26, value=8, step=1
)
inputs["Curricular_units_1st_sem_enrolled"] = int(enr1)

# 8. Curricular_units_1st_sem_evaluations
eval1 = st.sidebar.slider(
    "Units 1st sem (evaluations)", min_value=0, max_value=45, value=10, step=1
)
inputs["Curricular_units_1st_sem_evaluations"] = int(eval1)

# 9. Curricular_units_1st_sem_approved
app1 = st.sidebar.slider(
    "Units 1st sem (approved)", min_value=0, max_value=26, value=7, step=1
)
inputs["Curricular_units_1st_sem_approved"] = int(app1)

# 10. Curricular_units_1st_sem_without_evaluations
we1 = st.sidebar.slider(
    "Units 1st sem (without evaluations)", min_value=0, max_value=12, value=0, step=1
)
inputs["Curricular_units_1st_sem_without_evaluations"] = int(we1)

# 11. Curricular_units_2nd_sem_credited
cred2 = st.sidebar.slider(
    "Units 2nd sem (credited)", min_value=0, max_value=19, value=5, step=1
)
inputs["Curricular_units_2nd_sem_credited"] = int(cred2)

# 12. Curricular_units_2nd_sem_enrolled
enr2 = st.sidebar.slider(
    "Units 2nd sem (enrolled)", min_value=0, max_value=23, value=6, step=1
)
inputs["Curricular_units_2nd_sem_enrolled"] = int(enr2)

# 13. Curricular_units_2nd_sem_evaluations
eval2 = st.sidebar.slider(
    "Units 2nd sem (evaluations)", min_value=0, max_value=33, value=10, step=1
)
inputs["Curricular_units_2nd_sem_evaluations"] = int(eval2)

# 14. Curricular_units_2nd_sem_approved
app2 = st.sidebar.slider(
    "Units 2nd sem (approved)", min_value=0, max_value=20, value=5, step=1
)
inputs["Curricular_units_2nd_sem_approved"] = int(app2)

# 15. Curricular_units_2nd_sem_without_evaluations
we2 = st.sidebar.slider(
    "Units 2nd sem (without evaluations)", min_value=0, max_value=12, value=0, step=1
)
inputs["Curricular_units_2nd_sem_without_evaluations"] = int(we2)

st.markdown(
    """
    <div style="text-align: left; margin-top: 1rem;">
    Setelah semua input diisi, klik tombol di bawah untuk memprediksi status mahasiswa:
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("🔍 Prediksi Status"):
    # Konversi inputs ke DataFrame satu baris
    input_df = pd.DataFrame([inputs])

    # Gunakan pipeline untuk memprediksi
    prediction = model_pipeline.predict(input_df)[0]
    proba = model_pipeline.predict_proba(input_df)[0][1]  # Prob untuk class = 1 (Dropout)

    if prediction == 1:
        st.error(
            f"⚠️ Prediksi: **Dropout** (Probabilitas = {proba*100:.2f}%)"
        )
        st.write(
            """
            **Rekomendasi Awal:**  
            - Pastikan mahasiswa mendapat bimbingan akademik intensif.  
            - Cek kembali jumlah mata kuliah yang telah disetujui di semester 2.  
            - Tinjau status pembayaran biaya kuliah dan berikan bantuan finansial jika diperlukan.  
            """
        )
    else:
        st.success(
            f"✅ Prediksi: **Graduate** (Probabilitas Dropout = {proba*100:.2f}%)"
        )
        st.write(
            """
            **Kondisi Baik:**  
            - Mahasiswa menunjukkan indicator akademik yang cukup kuat.  
            - Terus pantau performa semester berikutnya.  
            """
        )

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; color: gray;">
    &copy; 2025 - Sistem Prediksi Dropout Mahasiswa<br>
    <i><b>Author:</b> B. Hilda Nida Alistiqlal</i>
    </div>
    """,
    unsafe_allow_html=True
)
