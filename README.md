# Comparative Analysis of Discriminative, Reconstructive, and Generative Deep Learning Models on Fashion-MNIST

Eksplorasi teoretis dan empiris mengenai pergeseran paradigma pemodelan dari jaringan diskriminatif (*MLP Classifier*), representasi berbasis kompresi laten (*Autoencoder Family*), hingga pemodelan distribusi generatif (*GAN Family* dan *Denoising Diffusion Probabilistic Models*) menggunakan PyTorch pada dataset **Fashion-MNIST**.

---

## Daftar Isi
1. [Tujuan Eksperimen](#tujuan-eksperimen)
2. [Struktur Repositori](#struktur-repositori)
3. [Spesifikasi Arsitektur & Hyperparameter](#spesifikasi-arsitektur--hyperparameter)
4. [Instalasi & Panduan Eksekusi](#instalasi--panduan-eksekusi)
5. [Hasil Eksperimen & Analisis Kuantitatif](#hasil-eksperimen--analisis-kuantitatif)
6. [Analisis Kompleksitas Komputasi](#analisis-kompleksitas-komputasi)

---

## Tujuan Eksperimen
1. **Membuktikan Batasan Model Diskriminatif**: Menguji secara empiris mengapa bobot *classifier* berakurasi tinggi tidak dapat dibalik secara langsung (*Top-Down Weight Transposition*) untuk mensintesis citra baru, serta mengamati fenomena *information bottleneck* pada ruang laten diskriminatif.
2. **Mengevaluasi Trade-off Rekonstruksi vs. Regularisasi Laten**: Membandingkan akurasi rekonstruksi piksel antara *Vanilla Autoencoder*, *Sparse Autoencoder* (dengan penalti $L_1$), dan *Variational Autoencoder* (VAE).
3. **Membandingkan Paradigma Generatif**: Menganalisis kualitas visual, stabilitas konvergensi, dan distribusi fitur (*FID Score*) antara arsitektur *Fully-Connected* (*Vanilla GAN*), *Convolutional* (*DCGAN*), dan *Iterative Denoising* (*DDPM*).

---

## Struktur Repositori

```text
Deep-Learning-Assignment-6-7/
├── dataset.py                  # Pipeline pemuatan Fashion-MNIST (Train, Val, Test DataLoader)
├── models.py                   # Implementasi 7 arsitektur model (MLP, AE, SAE, VAE, GAN, DCGAN, DDPM)
├── train.py                    # Modular training loops dan penyimpanan bobot (.pth)
├── evaluate.py                 # Kalkulasi metrik MSE, PSNR, SSIM, FID Score (Inception-v3), & plot grid
├── utils.py                    # Utilitas random seed, deteksi GPU, hitung parameter, & simpan visualisasi
├── requirements.txt            # Daftar dependensi pustaka Python
├── generative_comparison.png   # Grid komparasi visual hasil sintesis citra antar-model generatif
└── README.md                   # Dokumentasi proyek dan laporan hasil eksperimen
