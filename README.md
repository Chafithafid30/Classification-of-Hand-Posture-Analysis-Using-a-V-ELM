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
