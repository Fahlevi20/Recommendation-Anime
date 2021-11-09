# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Ada beberapa masalah yang biasanya ditemukan dalam konten steraming dan salah satunya adalah Pengguna dapat menghabiskan waktu berjam-jam untuk mencari ratusan bahkan ribuan untuk mencari anime yang mereka sukai.

Untuk memecahkan masalah tersebut pengguna perlu diberikan saran berdasarkan kesukaan dan kebutuhan pengguna. Maka dari itu perlu adanya sistem rekomendasi untuk menciptakan lingkungan streaming yang lebih baik yang meningkatkan pendapatan dan mempersingkat waktu yang dihabiskan pengguna didalam situs streaming untuk mencari anime yang pengguna sukai.

referensi:\
[myanimelist.net](https://myanimelist.net/)\
[Streaming Platform and Strategic Recommendation Bias](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3338744)

## Business Understanding

Pengguna dapat menghabiskan waktu berjam-jam untuk mencari ratusan bahkan ribuan untuk mencari anime yang mereka sukai. Maka dari itu perlu adanya sistem rekomendasi yang dapat memberikan saran kepada pengguna sehingga pengguna dapat mendapat beberapa pilihan yang mungkin disukai pengguna

### Problem Statements

Pengguna perlu diberikan saran berdasarkan kesukaan dan kebutuhan pengguna untuk membantu pengguna dalam mencari anime yang mereka sukai

### Goals

Model dapat memberikan 10 rekomendasi teratas anime yang mungkin pengguna sukai

### Solution approach

Membangun sistem rekomendasi yang merekomendasikan 10 anime teratas berdasarkan kesukaan dan kebutuhan mereka dengan menggunakan teknik rekomendasi Content-Based Filtering.

Content-Based Filtering adalah teknik merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Content-Based Filtering bekerja dengan data yang diberikan pengguna, baik secara eksplisit (peringkat) atau implisit (mengklik tautan). Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna.

## Data Understanding

Dataset yang dipakai adalah [Anime Recommendation Database 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020)
yang memiliki 320.000 users dan 16.000 anime dari [myanimelist.net](https://myanimelist.net/)

anime data:

- anime_id - kode unik untuk mengindentifikasi setiap anime
- name - judul dari anime
- genre - list genre yang dipisahkan tanda koma untuk setiap anime.
- type - tipe dari anime
- episodes - berapa banyak episode dari anime. (1 jika movie).
- rating - rating rata-rata dari 10.
- members - jumlah anggota komunitas yang ada di "grup" anime.

lalu selanjutnya saya menganalisis data dengan menggunakan teknik Univariate Exploratory Data Analysis hal pertama yang saya lakukan adalah menampilkan 5 data teratas dari dataset dengan menggunakan code **anime_df.head()**

kemudian saya menganalisis Dimensi data dengan menggunakan code **anime_df.shape**

dengan hasil keluaran

```python
(12294, 7)
```

dan terakhir saya menganalisis tipe data dari setiap variabel dengan menggunakan code **anime_df.info()**
dengan hasil keluaran

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12294 entries, 0 to 12293
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   anime_id  12294 non-null  int64
 1   name      12294 non-null  object
 2   genre     12232 non-null  object
 3   type      12269 non-null  object
 4   episodes  12294 non-null  object
 5   rating    12064 non-null  float64
 6   members   12294 non-null  int64
dtypes: float64(1), int64(2), object(4)
memory usage: 672.5+ KB
```

sehingga saya mengetahui tipe data pada setiap variabel pada dataset anime.

lalu yang terakhir saya menganalisis anime berdasarkan tipenya dengan cara memvisualisasikannya agar lebih mudah dipahami. visualisasi yang dikeluarkan berdasarkan tipe anime menunjukan bahwa TV menjadi tipe yang paling banyak dan musik yang paling sedikit.

![feature_recomendation](https://i.ibb.co/XzVZS2v/1.jpg)

## Data Preparation

langkah pertama yang saya lakukan pada data preparation adalah menduplikasi variabel yang menampung dataset **anime.csv** yaitu variabel **'anime_df'** lalu data duplikasi ditampung pada variabel **'anime_data'** sehingga dataset pada variabel **'anime_df'** yang menampung dataset induk tidak terkontaminasi dan bisa digunakan kembali jika saya ingin mengembangkan model rekomendasi.

selanjutnya saya menganalisis apakah data memiliki null value atau tidak dengan menggunakan code

```python
null_features = anime_df.columns[anime_df.isna().any()]
anime_df[null_features].isna().sum()
```

dengan hasil keluaran

```python
genre      62
type       25
rating    230
dtype: int64
```

karena data memiliki null value maka saya membuang data yang mempunyai null value menggunakan code

```python
anime_df.dropna(inplace=True)
anime_df[null_features].isna().sum()
```

```python
genre     0
type      0
rating    0
dtype: int64
```

sehingga dataset tidak memiliki null value lagi

dan terakhir saya melakukan penghapusan karakter Jepang atau khusus pada variabel **name** yang bisa membuat dataframe tidak bisa membacanya. teknik yang digunakan adalah Regex.

berikut adalah contoh data sebelum dilakukan penghapusan dan sesudahnya:

sebelum

```python
array(['Steins;Gate', 'Gintama&#039;',
       'Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou',
       'Hunter x Hunter (2011)', 'Ginga Eiyuu Densetsu',
       'Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare',
       'Gintama&#039;: Enchousen'], dtype=object)
```

sesudah

```python
array(['Kimi no Na wa.', 'Fullmetal Alchemist: Brotherhood', 'Gintama°',
       'Steins;Gate', 'Gintama',
       'Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou',
       'Hunter x Hunter (2011)', 'Ginga Eiyuu Densetsu',
       'Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare',
       'Gintama: Enchousen'], dtype=object)
```

## Modeling

Pada tahap modeling saya menggunakan teknik sistem rekomendasi content based filtering.

Di sini saya akan menggunakannya genre sebagai acuan content based filteringnnya sehingga saya dapat merekomendasikan pengguna berdasarkan konten genre.

Tahap pertama yang saya lakukan adalah TF-IDF Vectorizer  
TF-IDF Vectorizer digunakan untuk menemukan representasi fitur penting dari setiap genre anime.

untuk mengimplementasikannya tahap pertama saya Inisialisasi TfidfVectorizer terlebih dahulu dengan code

```python
tfv = TfidfVectorizer(min_df=3,  max_features=None,
                      strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words = 'english')
```

karena setiap anime memiliki beberapa genre maka saya melakukan split terlebih dahulu

```python
genres_str = anime_data['genre'].str.split(',').astype(str)
```

lalu selanjutnya mengimplementasikannya dengan menggunakan code

```python
tfv_matrix = tfv.fit_transform(genres_str)
```

Selanjutnya kita perlu menetapkan 1 sebagai tanda anime yang direkomendasikan dan 0 sebagai anime yang tidak direkomendasikan. Dengan begitu saya menggunakan kernel sigmoid
dengan code sebagai beriku

```python
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(anime_data.index, index=anime_data['name']).drop_duplicates()
```

dan langkah terakhir dalam fase modeling ini saya membuat fungsi untuk mendapatkan rekomendasi anime. dengan cara kerja mengubah skor kesamaan menjadi list menggunakan fungsi enumerate, lalu mengurutkan daftar dan memilih n skor teratas untuk direkomendasikan dan menghapus data yang dicari agar tidak dimunculkan dalam list.

ada beberapa parameter yang ada pada function:

1. title: parameter ini bernilai string yang berfungsi untuk mendapatkan judul anime yang kemudian digunakan sebagai data untuk mendapatkan rekomendasi.
2. printTitle: parameter ini bernilai boolean yang berfungsi untuk menampilkan judul anime dan jumlah rekomendasi yang ingin di dapatkan.(default: false)
3. k: parmeter ini bernilai int yang berfungsi untuk mendapatkan berapa banyak rekomendasi yang ingin didapatkan.(default 10)

langkah yang dilakukan pada function tersebut adalah sebagai berikut:

1. mendapatkan indeks yang sesuai dengan judul anime
2. mendapatkan skor kesamaan berpasangan
3. megurutkan data anime dari score sigmoid_kernel tertinggi
4. mendapatkan indeks data anime
5. drop nama anime yang dicari agar tidak muncul dalam daftar rekomendasi
6. mendapatkan data rekomendasi sebanyak k
7. print title jika parameter title bernilai **True**
8. mengembalikan nilai sebanyak k anime paling mirip

Berikut adalah Top-N hasil dari rekomendasi dari model yang sudah saya buat
![menyajikan top 10 rekomendasi anime](https://i.ibb.co/4NwZbrJ/3-2.png)

## Evaluation

Untuk proses evalusi model saya menggunakan matriks precision

Precision adalah sebuah metrics yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat.

Formula Precision  
TP – True Positives  
FP – False Positives  
Precision – Accuracy of positive predictions.  
Precision = TP/(TP + FP)

Langkah pertama yang saya lakukan untuk mengevaluasi model adalah menampung terlebih dahulu data anime yang akan menjadi data uji coba, dalam kasus ini saya mencoba untuk menampung data anime yang mempunyai judul "Death Note" dan saya tampung pada variabel **feature**

Lalu selanjutnya saya menampung genre yang ada pada data uji coba untuk selanjutnya dipakai untuk evaluasi model.

Kamudian saya menampung data hasil dari model rekomendasi dengan menggunakan judul dari anime data uji coba dan dalam hal ini mendapatkan 10 data teratas.

berikut adalah data hasil dari model rekomendasi
![feature_recomendation](https://i.ibb.co/ByLmFsr/2.jpg)

Dan langkah terakhir yang saya lakukan adalah membuat perulangan berdasarkan genre pada data uji coba dan melakukan implementasi dari formula precision.

Berikut adalah hasil keluaran dari implementasi formula precision

```python
Mystery: 100.0
 Police: 100.0
 Psychological: 100.0
 Supernatural: 100.0
 Thriller: 100.0
```

hasil yang diberikan cukup baik sehingga dari sini saya bisa mengetahui bahwa model yang saya kembangkan berjalan sesuai yang diharpkan
