# GenreClas
Music Genre Classification: A Comparative Study (LightGBM vs. CNN)This repository contains the implementation and analysis conducted for a Master's Thesis focused on benchmarking the performance and computational efficiency of two fundamentally different machine learning paradigms in the field of Music Information Retrieval (MIR).Project GoalThe primary objective is to compare the effectiveness of a Gradient Boosting model (LightGBM) using handcrafted vector features against a Convolutional Neural Network (CNN) using visual representations (spectrograms) for classifying music genres.Key Features and TechniquesDual-Path Modeling: Implementation of two separate classification pipelines to test performance differences based on data representation:Tabular Path (LightGBM): Utilizes engineered features (MFCC, Chroma, Spectral Centroid, etc.) extracted from audio files.Visual Path (CNN): Trains a deep learning model directly on Mel-Spectrograms (image data) to learn complex time-frequency patterns.Dataset: Experiments are conducted using the industry-standard GTZAN Dataset (10 genres, 1000 tracks).Computational Efficiency Analysis: Detailed comparison of training time and inference time between the highly-optimized LightGBM framework and the GPU-intensive CNN architecture.Repository Structure.
├── data/
│   ├── GTZAN/
│   └── spectrograms/  (Generated Mel-Spectrogram images)
├── src/
│   ├── 1_feature_extraction.py  (Librosa -> MFCC/CSV)
│   ├── 2_lightgbm_model.ipynb   (LightGBM implementation & tuning)
│   └── 3_cnn_model.ipynb        (CNN architecture and training)
├── results/
│   ├── final_metrics.csv      (Comparative results table)
│   └── confusion_matrix_cnn.png
├── README.md
└── requirements.txt
Core Technologies  CategoryTools / Libraries
Language  Python 3.x
Audio Processing  Librosa
Deep Learning  TensorFlow / Keras (for CNN implementation)
Classic ML  LightGBM, Scikit-learn
Data Handling  Pandas, NumPy
Visualization  Matplotlib, SeabornKey 

Conclusion Highlight The project demonstrates proficiency in developing end-to-end ML solutions and conducting critical performance benchmarking. The analysis provides insights into the trade-off between high accuracy (CNN) and computational efficiency (LightGBM) in resource-constrained environments.
