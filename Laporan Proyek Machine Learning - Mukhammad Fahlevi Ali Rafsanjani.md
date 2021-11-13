# Laporan Proyek Machine Learning - Mukhammad Fahlevi Ali Rafsanjani

## Project Overview

Domain proyek yang saya pilih dalam proyek _Machine Learning_ ini adalah mengenai **Hiburan** dengan judul "Sistem Rekomendasi Anime".
## Latar Belakang
Anime adalah istilah Bahasa Jepang yang berasal dari Bahasa Inggris yaitu _animation_. Dalam Bahasa Indonesia, kata tersebut diserap menjadi animasi.  Anime banyak sekali diminati oleh orang di seluruh dunia salah satunya di Indonesia. Selain itu Anime dapat ditonton dimana saja dan kapan saja karena sudah tersedia di banyak media seperti TV, Netflix, AbemaTV, YouTube, dan media lainnya. Sistem rekomendasi adalah suatu aplikasi yang digunakan untuk memberikan rekomendasi dalam membuat suatu keputusan yang diinginkan pengguna. Untuk meningkatkan _user experience_ dalam menemukan anime yang menarik dan yang sesuai dengan yang user inginkan, maka sistem rekomendasi adalah pilihan yang tepat untuk diterapkan. Dengan adanya sistem rekomendasi, _user experience_ tentu akan lebih baik karena pengguna bisa mendapatkan rekomendasi anime yang ingin ditonton.

## Business Understanding

### Problem Statements
- Bagaimana cara membuat model _Machine Learning_ untuk merekomendasikan anime kepada pengguna yang dipersonalisasi dengan teknik _content-based filtering_?

### Goals
- Membuat model _Machine Learning_ untuk menghasilkan sejumlah rekomendasi anime yang dipersonalisasi berdasarkan pengguna dengan teknik _content-based filtering_.
- mengetahui hasil *precision* dari *Genre.*

### Solution statements

Untuk menyelesaikan masalah ini, saya mengajukan sebuah solusi yaitu teknik _content-based filtering_. Berikut adalah penjelasan teknik-teknik yang akan digunakan untuk masalah ini :

- **Content-Based Filtering** : Merupakan cara untuk memberi rekomendasi bedasarkan genre atau fitur pada item yang disukai oleh pengguna. _Content-based filtering_ mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.

## Data Understanding



Berikut adalah informasi dari dataset yang digunakan pada proyek ini :

| Jenis                   | Keterangan                                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------|
| Sumber                  | [Anime Recommendations Database](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020?select=anime.csv) |
| Lisensi                 | CC0: Public Domain                                                                                       |
| Kategori                | movies and tv shows, anime and manga, comics and animation, popular culture                                                                                     |
| Usability       | 8.2                                                                                                |
| Jenis dan Ukuran Berkas | CSV (112 MB)                                                                                             |

Data yang saya pakai adalah Database Rekomendasi Anime yang berisi informasi tentang data preferensi pengguna dari 73.516 pengguna di 12.294 anime di myanimelist.net. dengan sebuah file yaitu Anime.csv yang memiliki 7 fitur. Berikut penjelasannya :


1. `anime_id` ID unik yang mengidentifikasi anime
2. `name` Judul anime
3. `genre` Daftar genre pada anime yang dipisahkan dengan tanda koma
4. `type` movie, TV, OVA, dll
5. `episodes` Jumlah episode (1 jika movie)
6. `rating` Rata-rata rating untuk anime
7. `members` Jumlah anggota komunitas yang ada di anime


 Data Loading sebagai berikut

|anime_id|name                                                                                                |genre                                                                                                                        |type   |episodes|rating|members|
|--------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------|--------|------|-------|
|32281   |Kimi no Na wa.                                                                                      |Drama, Romance, School, Supernatural                                                                                         |Movie  |1       |9.37  |200630 |
|5114    |Fullmetal Alchemist: Brotherhood                                                                    |Action, Adventure, Drama, Fantasy, Magic, Military, Shounen                                                                  |TV     |64      |9.26  |793665 |
|28977   |Gintama°                                                                                            |Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen                                                                 |TV     |51      |9.25  |114262 |
|9253    |Steins;Gate                                                                                         |Sci-Fi, Thriller                                                                                                             |TV     |24      |9.17  |673572 |
|9969    |Gintama&#039;                                                                                       |Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen                                                                 |TV     |51      |9.16  |151266 |
|32935   |Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou                                           |Comedy, Drama, School, Shounen, Sports                                                                                       |TV     |10      |9.15  |93351  |
|11061   |Hunter x Hunter (2011)                                                                              |Action, Adventure, Shounen, Super Power                                                                                      |TV     |148     |9.13  |425855 |
|820     |Ginga Eiyuu Densetsu                                                                                |Drama, Military, Sci-Fi, Space                                                                                               |OVA    |110     |9.11  |80679  |
|15335   |Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare                                                 |Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen                                                                 |Movie  |1       |9.10  |72534  |
|15417   |Gintama&#039;: Enchousen                                                                            |Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen                                                                 |TV     |13      |9.11  |81109  |
|4181    |Clannad: After Story                                                                                |Drama, Fantasy, Romance, Slice of Life, Supernatural                                                                         |TV     |24      |9.06  |456749 |
|---|---|---|---|---|---|---|---|---|
|1078    |Cardcaptor Sakura: Leave It to Kero-chan                                                            |Comedy                                                                                                                       |Movie  |1       |7.46  |20410  |
|4138    |Chiisana Pengin: Lolo no Bouken                                                                     |Adventure, Drama, Fantasy, Kids, Slice of Life                                                                               |OVA    |3       |7.46  |1459   |
|5593    |Dogs: Bullets &amp; Carnage                                                                         |Action, Seinen                                                                                                               |OVA    |4       |7.46  |44826  |
|2665    |Doraemon Movie 07: Nobita to Tetsujin Heidan                                                        |Adventure, Comedy, Fantasy, Kids, Shounen                                                                                    |Movie  |1       |7.46  |1272   |
|2676    |Doraemon Movie 15: Nobita to Mugen Sankenshi                                                        |Adventure, Fantasy, Kids, Sci-Fi, Shounen                                                                                    |Movie  |1       |7.46  |1204   |
|408     |Final Fantasy VII: Last Order                                                                       |Action, Adventure, Drama, Fantasy, Sci-Fi                                                                                    |OVA    |1       |7.46  |48041  |
|32696   |Fukigen na Mononokean                                                                               |Comedy, Supernatural                                                                                                         |TV     |13      |7.46  |44786  |

- Berdasarkan output di atas, dapat diketahui bahwa file Anime.csv memiliki 12294 entri.

|      |     anime_id |       rating  |     members|
|------|--------------|---------------|-------------|
|count|  12294.000000|  12064.000000  |1.229400e+04|
|mean|   14058.221653 |     6.473902  |1.807134e+04|
|std|    11455.294701 |     1.026746  |5.482068e+04|
|min|        1.000000 |     1.670000  |5.000000e+00|
|25%|     3484.250000 |     5.880000 | 2.250000e+02|
|50%|    10260.500000 |     6.570000 | 1.550000e+03|
|75%|    24794.500000 |     7.180000 | 9.437000e+03|
|max|    34527.000000 |    10.000000|  1.013917e+06|

- Dari output di atas, dapat disimpulkan bahwa nilai maksimum rating adalah 10 dan nilai minimumnya adalah 1.67 (1). Artinya, skala rating berkisar antara 1 hingga 10.
Pada tahap ini, saya membersihkan data NaN lalu kemudian mengubah tipe data pada kolom rating menjadi integer untuk menyamakan dengan kolom rating pada variabel rating.

- jumlah NaN 317 dari NaN yg ditemukan berdasarkan kolom :

|Kolom|NaN|
|-------------|---|
|anime_id     | 0|
|name        |  0|
|genre      |  62|
|type      |   25|
|episodes |     0|
|rating  |    230|
|members|       0|

- Ini adalah informasi yang didapatkan dari hasil eksplorasi pada variabel anime.

#### Visualization Data
![animeTypes](https://raw.githubusercontent.com/Fahlevi20/Recommendation-Anime/main/visualization/SharedScreenshot.jpg)

- Insight yang saya dapatkan disni adalah:
  - Anime Types "TV" yang paling banyak dibandingkan tipe lain
  - Anime Types "Music" yang paling sedikit dibandingkan tipe lain




## Data Preparation
sebelum membuat model, perlunya  melakukan pada data preparation adalah menduplikasi variabel dan juga text cleaning agar dapat memberikan hasil rekomendasi yang baik
- **Duplikasi variabel dataset**
melakukan duplikasi pada variabel **`df1`** lalu data duplikasi ditampung pada variabel **`df1_data`** sehingga dataset pada variabel **`df1`** yang menampung dataset induk tidak terkontaminasi dan bisa digunakan kembali jika saya ingin mengembangkan model rekomendasi.
- **Text Cleaning**
 melakukan text cleaning pada kolom **`name`** untuk menghilangkan simbol atau teks yang tidak diperlukan dengan cara menggunakan teknik Regex agar membuat function yang bernama `text cleaning` dan mengaplikasikannya pada **`df1_data`**

## Modeling and Result
Pada Proyek yang dibuat, tahapan modelling yang digunakan dalam teknik sistem rekomendasi ***Content Based Filtering***. Karena dapat merekomendasikan pengguna berdasarkan konten genre. Sehingga acuan yang dibuat berdasarkan genre.
### Content-based Filtering

- saya menggunakan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap genre anime. Fungsi yang saya gunakan adalah tfidfvectorizer() dari library sklearn. Berikut sebagian outputnya :
![genre](https://raw.githubusercontent.com/Fahlevi20/Recommendation-Anime/97c6b782ed5b9957ba61121b56466931267522a8/visualization/genre.jpg)
- Selanjutnya saya melakukan fit dan transformasi ke dalam bentuk matriks. Outputnya adalah matriks yang merupakan matrik genre anime.
Untuk menghitung derajat kesamaan (similarity degree) antar anime, saya menggunakan teknik cosine similarity dengan fungsi cosine_similarity dari library sklearn. Berikut adalah rumusnya :

![image](https://raw.githubusercontent.com/Fahlevi20/Recommendation-Anime/97429cbf2d4cfa4be90a05c89fe78f9da7a0d140/visualization/download.png)

- Berikut adalah outputnya :

![image](https://raw.githubusercontent.com/Fahlevi20/Recommendation-Anime/fd06c6a9616d27649452110418d41a1d93f18042/visualization/tfidf.jpg)

Langkah selanjutnya adalah saya menggunakan argpartition untuk mengambil sejumlah nilai k tertinggi dari similarity data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah.

Kemudian saya menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi Anime yang mirip dengan **One Pice**. Berikut adalah detail informasi Anime **One Piece** :

![image](https://raw.githubusercontent.com/Fahlevi20/Recommendation-Anime/0df7e2f9dafdbcbf019dfb7b27efa420c41a4954/visualization/rekom.jpg)

Berdasarkan output di atas, dapat dilihat bahwa Anime dengan judul One Piece memiliki genre Action, Adventure, Comedy, Drama, Fantasy, Shounen, dan Super Power. Rekomendasi yang diharapkan adalah Anime dengan genre yang sama.

Berikut adalah rekomendasi yang diberikan oleh model yang telah dibuat :

![image](https://raw.githubusercontent.com/Fahlevi20/Recommendation-Anime/b1bd58a0d7db5e5583430c2201c6f048bbdc97e2/visualization/rekomen%2010%20.jpg)

Model berhasil memberikan rekomendasi 10 judul Anime dengan genre yang sama seperti yang diharapkan, yaitu Action, Adventure, Comedy, Drama, Fantasy, Shounen, dan Super Power.



## Evaluation

pada tahap ini, saya menggunakan metriks precision. Berikut penjelasannya :

- **Precision** Adalah sebuah metrics yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat.Berikut adalah rumusnya :

![image](https://www.mydatamodels.com/wp-content/uploads/2020/10/5.-Precision-formula.png)

Formula Precision  
TP – True Positives  
FP – False Positives  
Precision – Accuracy of positive predictions.  
Precision = TP/(TP + FP)

- kelebihan
  - Sangat baik untuk klasifikasi
  - Dokumen yang dipilih secara acak dari kumpulan dokumen yang diambil adalah relevan.
  - Precision bagus untuk kasus di mana kelasnya seimbang
- Kekurangan
  - Tidak baik untuk data yang *Imbalance*
  - hanya hasil teratas yang dikembalikan oleh sistem

Untuk mengevaluasi model adalah menampung terlebih dahulu data anime yang akan menjadi data uji coba, dalam kasus ini saya mencoba untuk menampung data anime yang mempunyai judul "One Piece" dan saya tampung pada variabel **feature**

Lalu selanjutnya saya menampung genre yang ada pada data uji coba untuk selanjutnya dipakai untuk evaluasi model.

Dan langkah terakhir yang saya lakukan adalah membuat perulangan berdasarkan genre pada data uji coba dan melakukan implementasi dari formula precision.

Berikut adalah hasil keluaran dari implementasi formula precision

```python
Action: 100.0
 Adventure: 100.0
 Comedy: 100.0
 Drama: 100.0
 Fantasy: 100.0
 Shounen: 100.0
 Super Power: 100.0
```


hasil yang diberikan cukup baik sehingga dari sini saya bisa mengetahui bahwa model yang saya kembangkan berjalan sesuai yang diharpkan

