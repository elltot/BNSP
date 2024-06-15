import streamlit as st
import pickle
import numpy as np

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    with open('catboost.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Memuat model
model = load_model()

# Menampilkan judul di tengah halaman
st.markdown("""
<style>
h1 {
  text-align: center;
}
</style>
""", unsafe_allow_html=True)
st.title('Prediksi Penyakit Jantung Menggunakan Model CatBoost')

# Menambahkan keterangan informasi dataset di tengah halaman
st.markdown("""
# Heart Disease Dataset

Dataset ini berisi informasi tentang pasien-pasien yang telah dianalisis untuk deteksi penyakit jantung. Berikut adalah deskripsi dari variabel-variabel yang terdapat dalam dataset :

| Variabel          | Deskripsi                                                                                         |
|-------------------|---------------------------------------------------------------------------------------------------|
| Age               | Usia pasien dalam tahun                                                                            |
| Sex               | Jenis kelamin pasien, dengan opsi 0 untuk perempuan (Female) dan 1 untuk laki-laki (Male)          |
| ChestPainType     | Jenis nyeri dada yang dialami oleh pasien, dengan opsi 0 untuk ATA (Aching), 1 untuk NAP (Not Heart-related), 2 untuk ASY (Asymptomatic), dan 3 untuk TA (Typical Angina) |
| RestingBP         | Tekanan darah istirahat pasien dalam mm Hg (milimeter raksa)                                       |
| Cholesterol       | Kolesterol pasien dalam mg/dl (miligram per desiliter)                                             |
| FastingBS         | Tingkat gula darah puasa pasien dalam mg/dl (miligram per desiliter), dengan 0 untuk tidak ada dan 1 untuk ada |
| RestingECG        | Hasil elektrokardiogram (EKG) istirahat pasien, dengan opsi 0 untuk Normal, 1 untuk ST-T abnormal, dan 2 untuk LVH |
| MaxHR             | Denyut jantung maksimum yang dicapai oleh pasien dalam bpm (denyut per menit)                      |
| ExerciseAngina    | Kehadiran angina yang dipicu oleh olahraga pada pasien, dengan 0 untuk tidak ada dan 1 untuk ada   |
| Oldpeak           | Depresi segmen ST yang diinduksi oleh olahraga relatif terhadap istirahat                          |
| ST_Slope          | Kemiringan segmen ST selama olahraga, dengan opsi 2 untuk Up (meningkat), 1 untuk Flat (datar), dan 0 untuk Down (menurun) |
| HighBloodPressure | Tekanan darah tinggi pada pasien, True untuk ada dan False untuk tidak ada                         |
| HeartDisease      | Kehadiran penyakit jantung pada pasien, dengan 0 untuk tidak ada dan 1 untuk ada                   |
""", unsafe_allow_html=True)

# Input dari pengguna
st.header('Masukkan Fitur-fitur untuk Prediksi')

Age = st.number_input('Umur', min_value=0, max_value=120, value=30)
Sex = st.selectbox('Jenis Kelamin', ['Female', 'Male'])  # Menggunakan string untuk input jenis kelamin
ChestPainType = st.selectbox('Tipe Nyeri Dada', ['ATA', 'NAP', 'ASY', 'TA'])  # Menggunakan string untuk input jenis nyeri dada
RestingBP = st.number_input('Tekanan Darah Istirahat', min_value=0, max_value=250, value=120)
Cholesterol = st.number_input('Kolesterol', min_value=0, max_value=600, value=200)
FastingBS = st.selectbox('Gula Darah Puasa > 120 mg/dl', [0, 1])
RestingECG = st.selectbox('Elektrokardiografi Istirahat', ['Normal', 'ST-T abnormal', 'LVH'])  # Menggunakan string untuk input hasil EKG istirahat
MaxHR = st.number_input('Detak Jantung Maksimal', min_value=0, max_value=250, value=150)
ExerciseAngina = st.selectbox('Angina Induced oleh Olahraga', ['Tidak', 'Ya'])  # Menggunakan string untuk input kehadiran angina
Oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
ST_Slope = st.selectbox('Kemiringan ST', ['Up', 'Flat', 'Down'])  # Menggunakan string untuk input kemiringan ST
HighBloodPressure = st.selectbox('Tekanan Darah Tinggi', [False, True])

# Mapping input string ke nilai numerik sesuai dengan dataset
sex_mapping = {'Female': 0, 'Male': 1}
chest_pain_mapping = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
resting_ecg_mapping = {'Normal': 0, 'ST-T abnormal': 1, 'LVH': 2}
exercise_angina_mapping = {'Tidak': 0, 'Ya': 1}
st_slope_mapping = {'Up': 2, 'Flat': 1, 'Down': 0}

# Membuat array dari fitur-fitur yang diinput
features = np.array([[Age,
                      sex_mapping[Sex],
                      chest_pain_mapping[ChestPainType],
                      RestingBP,
                      Cholesterol,
                      FastingBS,
                      resting_ecg_mapping[RestingECG],
                      MaxHR,
                      exercise_angina_mapping[ExerciseAngina],
                      Oldpeak,
                      st_slope_mapping[ST_Slope],
                      HighBloodPressure]])

# Prediksi menggunakan model
if st.button('Prediksi'):
    prediction = model.predict(features)
    # st.write(f'Kehadiran penyakit jantung pada pasien: {"Ada" if prediction[0] == 1 else "Tidak Ada"}')

    if(prediction[0] == 1) :
        prediction = 'Pasien Terkena Penyakit Jantung'
        st.success(prediction)
    else :
        prediction = 'Pasien Tidak Terkena Penyakit Jantung'
        st.error(prediction)
        
    