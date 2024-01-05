# Classification-of-Hand-Posture-Analysis-Using-a-V-ELM
Deep Learning: Classification of Hand Posture Analysis Using Algorithm Voting Based Extreme Learning Machine

## Explain Detail

### V-ELM (Voting Based Extreme Learning Machine)
Pada penelitian yang dilakukan oleh Cao et al (2012), disebutkan bahwa ELM biasa masih terdapat kesalahan pada sampel yang dekat dengan batas klasifikasi antar kelas. Kemungkinan terjadi kesalahan klasifikasi dan keputusan berdasarkan realisasi tunggal ELM kurang dapat diandalkan. Batas pemisah berbagai nonlinear dibangun dengan berbagai hidden node learning parameter yang acak menyebabkan variasi dalam klasifikasi. Menurut penelitian yang dilakukan oleh Ginting, dkk pada tahun 2017, permasalahan ini dapat diselesaikan dengan metode Voting-Based Extreme Learning Machine (V-ELM). Metode ini meningkatkan kinerja klasifikasi dari ELM dengan menggabungkan beberapa ELM bebas dan pengambilan keputusan dengan metode voting mayoritas.

### Algorithm
Pada V-ELM digunakan K (jumlah jaringan ELM independen) untuk memberikan hasil output kemudian dilakukan voting terhadap hasil tersebut sehingga menghasilkan kelas dari output tersebut. Algoritma setiap jaringan ELM independen sama seperti ELM biasa dengan menggunakan matriks Hessian untuk mendapatkan hasil keluaran. Pada proses pelatihan V-ELM, dibangun sejumlah K jaringan ELM untuk mendapatkan nilai keluaran hidden layer sehingga diperoleh output nilai weight, beta, dan MAPE untuk masing-masing jaringan ELM independen. Nilai ini akan digunakan pada proses pengujian. Sama seperti proses pelatihan, hasil output tadi diuji pada masing-masing jaringan ELM untuk menghasilkan hasil keluaran berupa kelas dari masing-masing data uji. Hasil keluaran yang diperoleh kemudian dihitung dan keluaran terakhir didapatkan berdasarkan nilai mayoritas dari seluruh jaringan ELM independen.

## Issues Raised & Data Used
Dataset yang akan kami gunakan dalam proyek ini adalah MoCap Hand Postures. Dataset tersebut berisi lima jenis postur tangan yang direkam dari 12 orang berbeda dengan menggunakan Vicon Motion Capture Camera. Perekaman postur tangan pada sarung tangan Motion Capture hanya dilakukan pada tangan kiri pengguna dengan total 11 penanda. Terdapat 3 penanda pada ibu jari dan 2 penanda untuk 4 jari lain. Terdapat banyak data perekaman yang memiliki missing value dikarenakan faktor posisi tangan/jari serta pembatasan resolusi dari volume perekaman. 
Dataset MoCap Hand Postures memiliki total 78.095 baris data, 38 atribut/kolom, dan berisi bilangan-bilangan real yang menyatakan koordinat penanda. Atribut “Class” menyatakan jenis postur tangan yang dibentuk dan dinyatakan dalam bilangan bulat dari angka 1 hingga 5. Atribut “User” menyatakan ID dari pengguna yang dinyatakan dalam bilangan bulat dari 0-14 dan hanya berfungsi untuk membedakan postur tangan yang dibentuk dari setiap pengguna. 12 ID berasal dari hasil perekaman dan 3 ID lainnya adalah duplikasi dengan nilai yang hampir menyerupai nilai-nilai asal duplikasinya. Terakhir, terdapat atribut “Xi”, “Yi”, dan “Zi” yang menyatakan posisi sumbu x, sumbu y, dan sumbu z dari setiap penanda dengan i adalah nomor penanda yang berada pada rentang 0-11.

## Analytical Approach
### Sample Dataset for Manualization
![](https://github.com/Chafithafid30/Classification-of-Hand-Posture-Analysis-Using-a-V-ELM/blob/master/Pendekatan%20Analitik.png)

### Pre Processing
Sebelum pelatihan dilakukan, perlu diperhatikan bahwa data pada dataset yang digunakan tidak ada yang bernilai null atau memiliki range yang sangat tinggi. Dalam mengatasi hal ini, perlu dilakukan imputasi dan normalisasi. Selain itu, sebelum normalisasi dilakukan, dataset perlu dibagi menjadi data latih dan data uji terlebih dahulu. Data uji merepresentasikan data dalam dunia nyata sehingga informasi apapun dari data uji ke data latih tidak boleh diberikan (Myrianthous, 2022). 
Imputasi dilakukan dengan mengganti nilai NaN dengan nilai mean dari setiap data pada satu kolom, sedangkan normalisasi dilakukan menggunakan metode MinMax. Hasil dari imputasi dan normalisasi adalah sebagai berikut:

### Pre Processing Training Data Results
![](https://github.com/Chafithafid30/Classification-of-Hand-Posture-Analysis-Using-a-V-ELM/blob/master/Data%20Latih%20Hasil%20PreProses.png)

### Pre Processing Result Test Data
![](https://github.com/Chafithafid30/Classification-of-Hand-Posture-Analysis-Using-a-V-ELM/blob/master/Data%20Uji%20Hasil%20PreProses.png)

### Training
Pada V-ELM digunakan K (jumlah jaringan ELM independen) untuk memberikan hasil output kemudian dilakukan voting. Pada pelatihan ini, dimisalkan nilai K=3 sehingga dilakukan pelatihan pada 3 jaringan ELM independen yang berbeda. Pada masing-masing jaringan ELM dilakukan perhitungan sebagai berikut.

1. Menyiapkan matriks data latih (X) dengan jumlah fitur (d)  dan matriks target data latih (t).
2. Menentukan jumlah hidden neuron (h), kemudian inisialisasi matriks bobot awal(W0) dengan ukuran d dan h.
3. Menghitung inisialisasi dari nilai matriks H init dengan mengalikan nilai matriks data latih (X) dan matriks bobot awal yang telah ditranspose (WTranspose).
4. Menghitung matriks output dari hidden layer (H) dengan persamaan H= 11 + e -Hinit  pada masing-masing data pada matriks H init.
5. Menghitung output weight atau beta (β) dengan persamaan  = H+ * t  dan  H+=(HT * H)-1 * HT.
6. Menghitung hasil prediksi dengan persamaan y = H *  dan menghitung nilai Mean Absolute Percentage Error (MAPE) dengan persamaan MAPE  = 100%N i=1Nyi-tiyi untuk mengkalkulasi berapa error yang didapatkan dari parameter learning yang digunakan.
NB: Dari nilai error yang didapatkan, beberapa algoritma ELM lain akan mengulangi pelatihan hingga didapatkan nilai MAPE yang paling kecil. Namun, nilai MAPE pada manualisasi ini hanya digunakan sebagai penunjuk seberapa besar error dari bobot yang digunakan.

### Testing
Setelah nilai bobot (W) dan beta (β) didapatkan dari proses pelatihan sebelumnya, proses perhitungan pengujian dilakukan. Pada masing-masing jaringan ELM dilakukan perhitungan sebagai berikut:

1. Menyiapkan matriks data uji (X uji) dengan jumlah fitur (d)  dan matriks target data uji (t uji)
2. Menghitung inisialisasi dari nilai matriks H init dengan mengalikan nilai matriks data uji (X uji) dan matriks bobot yang didapatkan dari proses pelatihan dan telah ditranspose (W Transpose)
3. Menghitung matriks output dari hidden layer (H) dengan persamaan H= 11 + e -Hinit  pada masing-masing data pada matriks H init.
4. Menghitung hasil prediksi akhir dengan persamaan Y =  round(H * )

### Final Testing
Setelah masing-masing jaringan ELM sejumlah K=3 telah menghasilkan hasil prediksi, hasil prediksi akhir dihitung berdasarkan nilai voting mayoritas dari seluruh hasil prediksi jaringan ELM. Perhitungan hasil voting tersebut adalah sebagai berikut:

### Final Test Table
![](https://github.com/Chafithafid30/Classification-of-Hand-Posture-Analysis-Using-a-V-ELM/blob/master/Tabel%20Pengujian%20Akhir.png)

NB: Hasil voting diambil dari nilai Y masing-masing data yang keluar paling banyak.

Dari tabel di atas, didapatkan hasil prediksi akhir adalah [ 2, 3, 5]. Kemudian dihitung akurasi dari hasil prediksi tersebut dengan perhitungan sebagai berikut.

### Final Test Accuracy Table
![](https://github.com/Chafithafid30/Classification-of-Hand-Posture-Analysis-Using-a-V-ELM/blob/master/Tabel%20Akurasi%20Pengujian%20Akhir.png)
