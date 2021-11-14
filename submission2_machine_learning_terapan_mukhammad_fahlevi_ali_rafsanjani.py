# -*- coding: utf-8 -*-
"""Submission2_Machine Learning Terapan_Mukhammad Fahlevi Ali Rafsanjani.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yPPw573S0IWN4rS9bPr7nhzUd4WFPqJi

# Import Library
"""

# Basic Libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Get Dataset Liblary
import os
from os.path import join

# Visualization Liblary
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as pylab
import plotly.graph_objects as go

# Preprocessing
import re

# Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

"""# Import Kaggle"""

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

"""# Import Dataset"""

!kaggle datasets download -d CooperUnion/anime-recommendations-database

!unzip anime-recommendations-database.zip -d anime-recommendations-database

df1 = pd.read_csv('/content/anime-recommendations-database/anime.csv')

"""# Exploratory Data Analysis

Disini saya melakukan Eksplorasi Data Analyis (EDA) dan Data Cleaning untuk memeriksa dataset yang akan saya training. Tujuannya adalah untuk melihat isi dari dataset tersebut apakah ada yang perlu dibuang atau tidak, apakah ada NaN, melihat tipe datanya, melihat jumlah kolom dan baris dan lainnya.
"""

print('===================EXPLORATORY DATA ANALYSIS=================================')
long_string='========================================================================'
def printByInformation(dataset, option=False):
  if option:
    pd.set_option('display.max_columns',None)
    print(f'current rows:{dataset.shape[0]}')
    print(f'current col:{dataset.shape[1]}')
    print(long_string)
    print('======================DATA CLEANING=======================================')
    print(f'jumlah NaN {dataset.isnull().sum().sum()} dari NaN yg ditemukan')
    print(long_string)
    print(f'jumlah NaN tiap Column\n{dataset.isnull().sum()}')
    print(long_string)
    print(f'Name Columns: {list(dataset.columns)}')
    print(long_string)
    print(f'{dataset.info()}')
    print(long_string)
    print(f'{dataset.describe()}')
printByInformation(df1,True)

"""Dataset yang dipakai adalah [Anime Recommendations Database](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020?select=anime.csv)
yang memiliki 320.000 users dan 16.000 anime dari [myanimelist.net](https://myanimelist.net/)

anime data:
* anime_id - kode unik untuk mengindentifikasi setiap anime
* name - judul dari anime
* genre - list genre yang dipisahkan tanda koma untuk setiap anime.
* type - tipe dari anime
* episodes - berapa banyak episode dari anime. (1 jika movie).
* rating - rating rata-rata dari 10.
* members - jumlah anggota komunitas yang ada di "grup" anime.
"""

type_count = df1['type'].value_counts()

sns.barplot(x=type_count.values,
            y=type_count.index,
            palette='Set1').set_title('Anime Types')

plt.tight_layout()
plt.show()

"""Insight yang saya dapatkan disni adalah:
- Anime Types "TV" yang paling banyak dibandingkan tipe lain
- Anime Types "Music" yang paling sedikit dibandingkan tipe lain
"""

df1_data=df1.copy()

"""Melakukan duplikasi data

# Data Cleaning
"""

null_features = df1_data.columns[df1_data.isna().any()]
df1_data[null_features].isna().sum()

df1_data.dropna(inplace=True)
df1_data[null_features].isna().sum()

"""karena data memiliki NaN maka saya membuang data yang mempunyai NaN

# Text Cleaning

selanjutnya saya melakukan text cleaning pada kolom **name** dengan cara menggunakan teknik Regex lalu saya membuat function yang bernama text cleaning dan mengaplikasikannya pada df1_data
"""

df1_data['name'].unique()[3:10]

def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    return text

"""selanjutnya saya membuat function dengan nama text_cleaning dengan tujuannya yaitu untuk menghilangkan simbol atau text yang tidak diperlukan"""

df1_data['name'] = df1_data['name'].apply(text_cleaning)

df1_data['name'].unique()[0:10]

"""# Modeling
Selanjutnya ini adalah tahap modeling saya menggunakan teknik sistem rekomendasi content based filtering. 

Di sini saya akan menggunakannya genre, sehingga saya dapat merekomendasikan pengguna berdasarkan konten genre.

- Disini saya menggunakan **TF-IDF Vectorizer** sebagai model yang digunakan untuk menemukan representasi fitur penting dari setiap genre anime.
"""

# Inisialisasi TfidfVectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None,
                      strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words = 'english')

genres_str = df1_data['genre'].str.split(',').astype(str)

genres_str

tfv_matrix = tfv.fit_transform(genres_str)

tfv_matrix

"""## Sigmoid Kernel
Selanjutnya kita perlu menetapkan 1 sebagai tanda anime yang direkomendasikan dan 0 sebagai anime yang tidak direkomendasikan. Dengan begitu saya menggunakan kernel sigmoid karena paling cocok untuk hasil binary
"""

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

sig

indices = pd.Series(df1_data.index, index=df1_data['name']).drop_duplicates()

"""## Membuat Recomendation Function
Selanjutnya membuat fungsi untuk mendapatkan rekomendasi anime. dengan cara kerja mengubah skor kesamaan menjadi list menggunakan fungsi enumerate, lalu mengurutkan daftar dan memilih 10 skor teratas untuk direkomendasikan dan menghapus data yang dicari agar tidak dimunculkan dalam list rekomendasi
"""

def get_recomendation(title, sig=sig):
    # mendapatkan indeks yang sesuai dengan original_title
    idx = indices[title]

    # mendapatkan skor kesamaan berpasangan
    sig_scores = list(enumerate(sig[idx]))

    # urutkan film
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # mendapatkan skor dari 10 film yang paling mirip
    sig_scores = sig_scores[0:11]

    # indeks film
    anime_indices = [i[0] for i in sig_scores]

    # Drop nama anime yang dicari agar tidak muncul dalam daftar rekomendasi
    anime_indices = [t for t in anime_indices if t != idx]

    # mengembalikan nilai 10 film paling mirip
    return pd.DataFrame({'Anime name': df1_data['name'].iloc[anime_indices].values,
                         'Genre': df1_data['genre'].iloc[anime_indices].values,
                         'Rating': df1_data['rating'].iloc[anime_indices].values})

"""# Evaluation
Untuk mengevaluasi model saya melakukan uji coba langsung dengan memanggil fungsi yang sudah saya buat dan memasukan judul anime yang pengguna suka sebelumnya.
"""

print("rekomendasi anime untuk One Piece")
get_recomendation('One Piece')

feature_recomendation = get_recomendation('One Piece')

feature = df1_data[df1_data['name'] == 'One Piece']

feature

get_feature_genre=[]
for i in range(len(feature.genre)):
    for x in feature.genre.str.split(','):
        if x not in get_feature_genre:
            get_feature_genre.append(x)

get_feature_genre

feature_recomendation

for i in get_feature_genre[0]:
  print(i + ": " + str((
      (feature_recomendation['Genre'].str.contains(i).count()/feature_recomendation['Genre'].count())*100)
  ))

"""disini dapat kita lihat bahwa sistem rekomendasi memberitahu bahwa genre terebut 100.0 atau 100% akurat benar"""
