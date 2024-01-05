# Classification-of-Hand-Posture-Analysis-Using-a-V-ELM
Deep Learning: Classification of Hand Posture Analysis Using Algorithm Voting Based Extreme Learning Machine

## Explain Detail

### V-ELM (Voting Based Extreme Learning Machine)
Pada penelitian yang dilakukan oleh Cao et al (2012), disebutkan bahwa ELM biasa masih terdapat kesalahan pada sampel yang dekat dengan batas klasifikasi antar kelas. Kemungkinan terjadi kesalahan klasifikasi dan keputusan berdasarkan realisasi tunggal ELM kurang dapat diandalkan. Batas pemisah berbagai nonlinear dibangun dengan berbagai hidden node learning parameter yang acak menyebabkan variasi dalam klasifikasi. Menurut penelitian yang dilakukan oleh Ginting, dkk pada tahun 2017, permasalahan ini dapat diselesaikan dengan metode Voting-Based Extreme Learning Machine (V-ELM). Metode ini meningkatkan kinerja klasifikasi dari ELM dengan menggabungkan beberapa ELM bebas dan pengambilan keputusan dengan metode voting mayoritas.

### Algorithm
Pada V-ELM digunakan K (jumlah jaringan ELM independen) untuk memberikan hasil output kemudian dilakukan voting terhadap hasil tersebut sehingga menghasilkan kelas dari output tersebut. Algoritma setiap jaringan ELM independen sama seperti ELM biasa dengan menggunakan matriks Hessian untuk mendapatkan hasil keluaran. Pada proses pelatihan V-ELM, dibangun sejumlah K jaringan ELM untuk mendapatkan nilai keluaran hidden layer sehingga diperoleh output nilai weight, beta, dan MAPE untuk masing-masing jaringan ELM independen. Nilai ini akan digunakan pada proses pengujian. Sama seperti proses pelatihan, hasil output tadi diuji pada masing-masing jaringan ELM untuk menghasilkan hasil keluaran berupa kelas dari masing-masing data uji. Hasil keluaran yang diperoleh kemudian dihitung dan keluaran terakhir didapatkan berdasarkan nilai mayoritas dari seluruh jaringan ELM independen.
