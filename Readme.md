# Laporan Proyek Machine Learning - Mukhammad Fahlevi Ali Rafsanjani

## Project Overview

Domain proyek yang saya pilih dalam proyek _Machine Learning_ ini adalah mengenai **Hiburan** dengan judul "Sistem Rekomendasi Anime".
## Latar Belakang
Anime adalah istilah Bahasa Jepang yang berasal dari Bahasa Inggris yaitu _animation_. Dalam Bahasa Indonesia, kata tersebut diserap menjadi animasi.  Anime banyak sekali diminati oleh orang di seluruh dunia salah satunya di Indonesia. Selain itu Anime dapat ditonton dimana saja dan kapan saja karena sudah tersedia di banyak media seperti TV, Netflix, AbemaTV, YouTube, dan media lainnya. Sistem rekomendasi adalah suatu aplikasi yang digunakan untuk memberikan rekomendasi dalam membuat suatu keputusan yang diinginkan pengguna. Untuk meningkatkan _user experience_ dalam menemukan anime yang menarik dan yang sesuai dengan yang user inginkan, maka sistem rekomendasi adalah pilihan yang tepat untuk diterapkan. Dengan adanya sistem rekomendasi, _user experience_ tentu akan lebih baik karena pengguna bisa mendapatkan rekomendasi anime yang ingin ditonton.

## Business Understanding

### Problem Statements
- Bagaimana cara membuat model _Machine Learning_ untuk merekomendasikan anime kepada pengguna yang dipersonalisasi dengan teknik _content-based filtering_?
- Bagaimana cara membuat model _Machine Learning_ untuk merekomendasikan anime yang mungkin disukai dan belum pernah ditonton oleh pengguna?

### Goals
- Membuat model _Machine Learning_ untuk menghasilkan sejumlah rekomendasi anime yang dipersonalisasi berdasarkan pengguna dengan teknik _content-based filtering_.
- mengetahui hasil persentasi prediksi genre

### Solution statements

Untuk menyelesaikan masalah ini, saya mengajukan sebuah solusi yaitu teknik _content-based filtering_. Berikut adalah penjelasan teknik-teknik yang akan digunakan untuk masalah ini :

- **Content-Based Filtering** : Merupakan cara untuk memberi rekomendasi bedasarkan genre atau fitur pada item yang disukai oleh pengguna. _Content-based filtering_ mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna.

## Data Understanding

![image](https://user-images.githubusercontent.com/87566521/139109868-1ef63ec2-d447-468a-926c-18691e8bd070.png)

Berikut adalah informasi dari dataset yang digunakan pada proyek ini :

| Jenis                   | Keterangan                                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------|
| Sumber                  | [Kaggle Dataset : Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database) |
| Lisensi                 | CC0: Public Domain                                                                                       |
| Kategori                | Anime, Manga                                                                                             |
| Rating Penggunaan       | 8.2 (Gold)                                                                                               |
| Jenis dan Ukuran Berkas | CSV (112 MB)                                                                                             |

Data yang saya pakai adalah Database Rekomendasi Anime yang berisi informasi tentang data preferensi pengguna dari 73.516 pengguna di 12.294 anime di myanimelist.net. dengan sebuah file yaitu Anime.csv yang memiliki 7 fitur. Berikut penjelasannya :

**Anime.csv**

1. `anime_id` ID unik yang mengidentifikasi anime
2. `name` Judul anime
3. `genre` Daftar genre pada anime yang dipisahkan dengan tanda koma
4. `type` movie, TV, OVA, dll
5. `episodes` Jumlah episode (1 jika movie)
6. `rating` Rata-rata rating untuk anime
7. `member` Jumlah anggota komunitas yang ada di anime

> *Catatan : kolom rating pada file Anime.csv adalah rating yang berasal dari ulasan pada situs web.*

Saya juga melakukan eksplorasi data untuk memahami variabel-variabel pada data serta korelasi antar variabel. Berikut penjelasannya :

**Anime.csv**

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

Berdasarkan output di atas, dapat diketahui bahwa file Anime.csv memiliki 12294 entri.

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
Dari output di atas, dapat disimpulkan bahwa nilai maksimum rating adalah 10 dan nilai minimumnya adalah 1.027 (1). Artinya, skala rating berkisar antara 1 hingga 10.
Pada tahap ini, saya membersihkan data NaN lalu kemudian mengubah tipe data pada kolom rating menjadi integer untuk menyamakan dengan kolom rating pada variabel rating.

jumlah NaN 317 dari NaN yg ditemukan berdasarkan kolom :
|Kolom|NaN|
|-------------|---|
|anime_id     | 0|
|name        |  0|
|genre      |  62|
|type      |   25|
|episodes |     0|
|rating  |    230|
|members|       0|

Ini adalah informasi yang didapatkan dari hasil eksplorasi pada variabel anime.

#### Visualization Data



## Data Preprocessing

Pada tahap ini, saya menggabungkan variabel anime dan variabel rating dengan fungsi merge dari library pandas. Berikut adalah hasil penggabungan variabel :

![image](https://user-images.githubusercontent.com/87566521/139245032-7b8218cd-f37f-4f07-89fc-731f4f24713e.png)

Setelah itu, saya mengganti nama kolom _name_ dan _ratinguser_ agar dapat lebih mudah dipahami. Berikut hasilnya :

![image](https://user-images.githubusercontent.com/87566521/139245393-b77f6b46-2fc2-488d-9b77-733e96a282fc.png)




## Data Preparation

Pada data preparation, saya memeriksa rating dengan nilai -1 yang kemudian dianggap sebagai outlier dan saya ubah menjadi NaN lalu saya bersihkan. Setelah itu, saya membuat variabel fix_anime yang berisi dataframe hasil penggabungan variabel anime dan variabel rating kemudian saya urutkan berdasarkan ID anime.

![image](https://user-images.githubusercontent.com/87566521/139247004-6fcf2488-8242-4426-9d23-c30e56996e34.png)

Pada proyek ini, saya hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, data yang duplikat akan saya hapus dengan fungsi drop_duplicates(). Dalam hal ini, saya membuang data duplikat pada kolom ‘anime_id’.

![image](https://user-images.githubusercontent.com/87566521/139247281-17121e4f-b30c-4895-9fbf-7d6c8301b8c4.png)

Setelah menghapus data duplikat, jumlah data saat ini adalah 9892 baris dan 9 kolom. Setelah itu, saya juga membuat visualisasi data untuk kolom `type`, kolom `rating`, dan kolom `rating_user` agar informasi dapat lebih mudah dipahami.

![image](https://user-images.githubusercontent.com/87566521/139247657-1e7ab0ae-6b4a-4893-b072-89d10024a56b.png)

Dari hasil visualisasi di atas, dapat disimpulkan bahwa anime ditayangkan lebih banyak di TV dan di OVA.

![image](https://user-images.githubusercontent.com/87566521/139247793-0f6d92a8-2bad-482c-b3fb-ebd9dc5c626e.png)

Dari hasil visualisasi di atas, dapat disimpulkan bahwa rata-rata rating melalui ulasan website adalah kebanyakan di antara 6 dan 7.

![image](https://user-images.githubusercontent.com/87566521/139247919-7dda3518-6e18-4bb3-a02e-9e9d268894ea.png)

Berdasarkan visualisasi di atas, dapat disimpulkan bahwa rating berdasarkan ID Pengguna paling banyak adalah 7 dan diikuti dengan 6 dan 8.

Saya juga menampilkan genre mana saja yang paling sering muncul dengan menggunakan word clouds plot.

![image](https://user-images.githubusercontent.com/87566521/139248116-14b2afa4-81f9-438c-8e1a-6a5134b3db42.png)

Berdasarkan plot yang telah dibuat, dapat dilihat bahwa genre yang paling sering muncul adalah Sci Fi, Comedy, Adventure, Slice Life, dan Action.

Langkah selanjutnya adalah saya melakukan konversi data series menjadi list. Dalam hal ini, saya menggunakan fungsi tolist() dari library numpy. Lalu saya akan melakukan persiapan data untuk menyandikan (encode) kolom ‘user_id’ dan ‘anime_id’ ke dalam indeks integer. Berikut adalah sebagian outputnya :

![image](https://user-images.githubusercontent.com/87566521/139248366-f8b26e0d-7de9-4274-8e5f-5efb8a67e356.png)

## Modeling and Result

Untuk tahap modeling, saya akan menggunakan Neural Network dan Cosine Similarity. Model Deep Learning akan saya gunakan untuk Sistem Rekomendasi berbasis Collaborative Filtering yang mana model ini akan menghasilkan rekomendasi untuk satu pengguna. Cosine Similarity akan saya gunakan untuk Sistem Rekomendasi berbasis Content-Based Filtering yang akan menghitung kemiripan antara satu film dengan lainnya berdasarkan fitur yang terdapat pada satu film. Berikut penjelasan tahapannya :

### Content-based Filtering

Langkah pertama, saya menggunakan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap genre anime. Fungsi yang saya gunakan adalah tfidfvectorizer() dari library sklearn. Berikut sebagian outputnya :

![image](https://user-images.githubusercontent.com/87566521/139250019-95314289-02fd-4302-9420-bb9258abde37.png)

Selanjutnya saya melakukan fit dan transformasi ke dalam bentuk matriks. Outputnya adalah matriks berukuran (9892, 47). Nilai 9892 merupakan ukuran data dan 47 merupakan matrik genre anime.

Untuk menghitung derajat kesamaan (similarity degree) antar anime, saya menggunakan teknik cosine similarity dengan fungsi cosine_similarity dari library sklearn. Berikut adalah rumusnya :

![image](https://user-images.githubusercontent.com/87566521/139251819-b8374a64-67ac-4fc3-82f5-02ff7a6b1b05.png)

Berikut adalah outputnya :

![image](https://user-images.githubusercontent.com/87566521/139250214-086929de-b21b-48ab-9f58-8e29bff795a1.png)

Langkah selanjutnya adalah saya menggunakan argpartition untuk mengambil sejumlah nilai k tertinggi dari similarity data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah.

Kemudian saya menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi Anime yang mirip dengan **Boruto: Naruto the Movie**. Berikut adalah detail informasi Anime **Boruto: Naruto the Movie** :

![image](https://user-images.githubusercontent.com/87566521/139252515-90f2552a-6f19-41b8-8260-2a5b94f4e3a4.png)

Berdasarkan output di atas, dapat dilihat bahwa Anime dengan judul Boruto: Naruto the Movie memiliki genre Action, Comedy, Martial Arts, Shounen, dan Super Power. Rekomendasi yang diharapkan adalah Anime dengan genre yang sama.

Berikut adalah rekomendasi yang diberikan oleh model yang telah dibuat :

![image](https://user-images.githubusercontent.com/87566521/139252620-bae93aa7-4a40-4db0-a95e-10634f5995a5.png)

Model berhasil memberikan rekomendasi 10 judul Anime dengan genre yang sama seperti yang diharapkan, yaitu Action, Comedy, Martial Arts, Shounen, dan Super Power.

### Collaborative Filtering

Tahapan yang saya lakukan untuk membuat sistem rekomendasi dengan teknik Collaborative Filtering adalah menghapus kolom yang tidak digunakan, melakukan mapping pada `user_id` dan `anime_id`, lalu mengubah nilai rating menjadi float. Saya membagi data untuk training dan validasi dengan komposisi 80:20. Namun sebelumnya, saya mengacak datanya terlebih dahulu agar distribusinya menjadi random. Lalu saya memetakan (mapping) data user dan anime menjadi satu value terlebih dahulu. Kemudian membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training. Berikut adalah outputnya :

![image](https://user-images.githubusercontent.com/87566521/139253831-6a4f6097-8972-4cf4-a60b-f75919c0ee44.png)

Saya melakukan proses embedding terhadap data user dan anime. Lalu melakukan operasi perkalian dot product antara embedding user dan anime. Selain itu, saya juga menambahkan bias untuk setiap user dan anime. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Untuk mendapatkan rekomendasi anime, saya mengambil sampel user secara acak dan mendefinisikan variabel anime_not_watched yang merupakan daftar anime yang belum pernah ditonton oleh pengguna.

Berikut adalah 10 rekomendasi anime untuk pengguna dengan ID 4643 :

![image](https://user-images.githubusercontent.com/87566521/139254122-bc3530eb-3446-478f-8148-8e51130d3bf2.png)

## Evaluation

Pada tahap ini, saya menggunakan Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE) sebagai metrics evaluation. Berikut penjelasannya :

1. `Mean Absolute Error (MAE)` mengukur besarnya rata-rata kesalahan dalam serangkaian prediksi yang sudah dilatih kepada data yang akan dites, tanpa mempertimbangkan arahnya. Semakin rendah nilai MAE (mean absolute error) maka semakin baik dan akurat model yang dibuat.

Berikut rumusnya :

![image](https://user-images.githubusercontent.com/87566521/139152819-30500f63-40a3-40ed-86fd-a62e517adbb4.png)

Berikut adalah plottingnya :

![image](https://user-images.githubusercontent.com/87566521/139255075-c754fea2-878f-4c91-9fd4-2eb2c3071a19.png)

2. `Root mean squared error (RMSE)` adalah aturan penilaian kuadrat yang juga mengukur besarnya rata-rata kesalahan. Sama seperti MAE, semakin rendahnya nilai root mean square error juga menandakan semakin baik model tersebut dalam melakukan prediksi.

Berikut rumusnya :

![image](https://user-images.githubusercontent.com/87566521/139154262-7eca086f-2007-41e1-9737-5f9fe68a8f49.png)

Berikut adalah plottingnya :

![image](https://user-images.githubusercontent.com/87566521/139255363-73417d98-1749-488d-8c7f-034e305ea7e5.png)

Berdasarkan plotting proses training yang telah dibuat, dapat dilihat bahwa proses training model cukup smooth dan model konvergen pada epochs sekitar 90. Dari hasil model ini, Mean Absolute Error yang didapat adalah 0.0077 pada training dan 0.1375 pada test. Untuk Root Mean Squared Error, diperoleh nilai error akhir sebesar 0.0126 pada training dan 0.1823 pada test. Hal ini menunjukan bahwa model ini memiliki error dibawah 20% jika menggunakan MAE dan dibawah 20% jika menggunakan RMSE.

## Conclusion

Model Sistem rekomendasi Anime telah selesai dibuat dan model ini dapat digunakan untuk merekomendasikan data yang sebenarnya. Berdasarkan model tersebut, dapat diketahui bahwa sistem rekomendasi berbasis Collaborative Filtering dan Content-Based Filtering dapat merekomendasikan anime kepada pengguna seperti yang diharapkan. Namun, beberapa pengembangan lain masih dapat dilakukan untuk membuat model yang memiliki akurasi lebih tinggi.


**---Ini adalah bagian akhir laporan---**
