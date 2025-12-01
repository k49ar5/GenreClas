import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_DIR = '../data/GTZAN'
CSV_PATH = '../data/features.csv'
SPECTROGRAMS_DIR = '../data/spectrograms/'
N_MFCC = 40
SAMPLE_RATE = 22050

all_features = []
os.makedirs(SPECTROGRAMS_DIR , exist_ok = True)

def extract_and_save_features(file_path, genre, index):
    y, sr = None, None
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)
    except Exception as e:
        print(f"Error while opening{file_path}:{e}")
        return
    """
    Extracting Vector Features (LightGBM)
    """
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr =sr, n_mfcc=N_MFCC)
    mfccs_mean = np.mean(mfccs.T, axis= 0)

    #Another statistic features

    chroma = librosa.feature.chroma_stft(y=y, sr =sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features_dict = {
        'filename': os.path.basename(file_path),
        'genre': genre,
        'spectral_centroid': np.mean(spec_cent),
        'zero_crossing_rate': np.mean(zcr),
        'chroma_stft_mean': np.mean(chroma)
    }
    for i in range(N_MFCC):
        features_dict[f'mfcc_{i + 1}'] = mfccs_mean[i]

    all_features.append(features_dict)

    """
    Generating Spectograms
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)


    plt.figure(figsize=(10, 4), frameon=False)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    base_file_name = os.path.basename(file_path)
    # Rozdziela 'utwor.mp3' na ('utwor', '.mp3')
    name_without_ext, _ = os.path.splitext(base_file_name)

    spectro_filename = os.path.join(SPECTROGRAMS_DIR, f'{index}_{name_without_ext}.png')

    plt.savefig(spectro_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

all_genres = os.listdir(AUDIO_DIR)
file_index = 0

for genre in all_genres:

    if os.path.isdir(os.path.join(AUDIO_DIR, genre)):

        genre_path = os.path.join(AUDIO_DIR, genre)

        print(f"Genre: {genre} is finished")


        for file_name in os.listdir(genre_path):
            if file_name.endswith(('.wav', '.mp3')):
                file_path = os.path.join(genre_path, file_name)


                extract_and_save_features(file_path, genre, file_index)
                file_index += 1

                if file_index % 100 == 0:
                    print(f"--- Processed {file_index} files. ---")

"""
Saving Data
"""

df = pd.DataFrame(all_features)


df.to_csv(CSV_PATH, index=False)

print("\n--- Extracting data is done ---")
print(f"Feature files for LightGBM has been saved in: {CSV_PATH}")
print(f"Numbers of row in CSV: {len(df)}")
