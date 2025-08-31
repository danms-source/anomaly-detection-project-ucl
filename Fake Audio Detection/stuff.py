import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import fft
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import gaussian_kde
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import fft
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import gaussian_kde
import json
import os
import json
import numpy as np
from tqdm import tqdm
import soundfile as sf  # for saving audio
import os
import random
import shutil
from collections import defaultdict
import math  # for ceiling and floor
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from collections import Counter

# Get folder where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(base_dir, 'validation')

#file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
#print(f"Number of files in folder: {file_count}")

# -------------------------------------------------------------------------------------------------------------------
# Converts mp3 to wav for consistency

"""
# Source and target folders
source_folder = os.path.join(base_dir, 'anthem_1308_librivox')
target_folder = os.path.join(source_folder, "converted_wav")


# Create target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Get sorted list of MP3 files
mp3_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".mp3")]
mp3_files.sort()  # Ensure consistent order

# Convert first 12 files
for i, mp3_file in enumerate(mp3_files[:12]):
    mp3_path = os.path.join(source_folder, mp3_file)
    wav_path = os.path.join(target_folder, os.path.splitext(mp3_file)[0] + ".wav")
    
    try:
        audio = AudioSegment.from_file(mp3_path)
        audio.export(wav_path, format="wav")
        print(f"Converted: {mp3_file} -> {os.path.basename(wav_path)}")
    except Exception as e:
        print(f"Failed to convert {mp3_file}: {e}")
"""




# File paths
real_audio_folder = os.path.join(base_dir, 'anthem_1308_librivox', 'converted_wav')
fake_clips_root = os.path.join(base_dir, 'fake_clips')


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------------------------

def load_audio(path, target_sr=16000, offset=0.0, duration=None):
    y, sr = librosa.load(path, sr=target_sr, mono=True, offset=offset, duration=duration)
    return y.astype(np.float32), sr

def estimate_speech_rate(y, sr):
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        sum(abs(y[i:i + frame_length] ** 2))
        for i in range(0, len(y), hop_length)
    ])
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))  # normalize
    peaks, _ = find_peaks(energy, height=0.3, distance=10)
    syllable_count = len(peaks)
    duration = librosa.get_duration(y=y, sr=sr)
    speech_rate = syllable_count / duration if duration > 0 else 0
    return speech_rate


def extract_spectral_features(y, sr):
    hop_length = 512
    duration = len(y) / sr  # Duration in seconds

    # === Spectral Centroid ===
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).flatten()
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    centroid_range = np.max(centroid) - np.min(centroid)

    increments = np.diff(centroid)
    std_increment = np.std(increments)
    max_abs_increment = np.max(np.abs(increments))

    # === Smoothness ratio ===
    smoothness_ratio = np.mean(np.abs(increments)) / (std_increment + 1e-8)

    # === Residual energy from linear regression ===
    times = librosa.times_like(centroid, sr=sr, hop_length=hop_length).reshape(-1, 1)
    model = LinearRegression().fit(times, centroid)
    predictions = model.predict(times)
    residuals = centroid - predictions
    residual_energy = np.mean(residuals ** 2)

    # === Spectral Bandwidth ===
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).flatten()
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # === Spectral Flatness ===
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length).flatten()
    flatness_mean = np.mean(flatness)
    flatness_std = np.std(flatness)

    # === Spectral Rolloff ===
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, hop_length=hop_length).flatten()
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # === Zero Crossing Rate ===
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).flatten()
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # === Speech Rate (onsets per second) ===
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    speech_rate = len(onsets) / duration if duration > 0 else 0

    # === Zero Crossings per second ===
    zero_crossings = np.sum(librosa.zero_crossings(y, pad=False))
    zero_crossings_per_sec = zero_crossings / duration if duration > 0 else 0

    # === MFCCs ===
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_means = {f"mfcc_{i+1}_mean": np.mean(mfccs[i]) for i in range(13)}
    mfcc_stds = {f"mfcc_{i+1}_std": np.std(mfccs[i]) for i in range(13)}

    # === Combine all features ===
    features = {
        "spectral_centroid_mean": centroid_mean,
        "spectral_centroid_std": centroid_std,
        "centroid_range": centroid_range,
        "std_increment": std_increment,
        "max_abs_increment": max_abs_increment,
        "smoothness_ratio": smoothness_ratio,
        "residual_energy": residual_energy,
        "spectral_bandwidth_mean": bandwidth_mean,
        "spectral_bandwidth_std": bandwidth_std,
        "spectral_flatness_mean": flatness_mean,
        "spectral_flatness_std": flatness_std,
        "spectral_rolloff_mean": rolloff_mean,
        "spectral_rolloff_std": rolloff_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "speech_rate": speech_rate,
        "zero_crossings_per_sec": zero_crossings_per_sec,
    }

    features.update(mfcc_means)
    features.update(mfcc_stds)

    return features

def extract_all_features(path, y, sr):
    spectral_features = extract_spectral_features(y, sr)
    return {
        **spectral_features
    }

def process_folder(folder_path, label):
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = load_audio(file_path)
                features = extract_all_features(file_path, y, sr)
                features["filename"] = filename
                features["label"] = label
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return features_list

def process_folder_segments(folder_path, label, segment_duration):
    features_list = []
    for filename in tqdm(os.listdir(real_audio_folder), desc="Processing real audio"):
        if filename.endswith(('.mp3', '.wav', '.flac', '.m4a')):
            file_path = os.path.join(folder_path, filename)
            try:
                total_duration = get_audio_duration(file_path)
                num_segments = int(total_duration // segment_duration)
                
                for i in range(num_segments):
                    offset = i * segment_duration
                    samples, sr = load_audio(file_path, offset=offset, duration=segment_duration)
                    if len(samples) < sr * 0.5:
                        print(f"Skipping short segment {i} in {filename}")
                        continue
                    features = extract_all_features(file_path, samples, sr)
                    features["filename"] = f"{filename}_segment{i}"
                    features["label"] = label
                    features_list.append(features)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return features_list

import os
import json
import numpy as np
from tqdm import tqdm
import soundfile as sf  # for saving audio

def process_segments_from_json(json_path, folder_path, label, save_dir="corpus_segments"):
    features_list = []

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create directory to save audio segments
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over chapters in the JSON
    for chapter_key, chapter_info in data.items():
        filename = chapter_info["filename"]
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping.")
            continue

        segments = chapter_info.get("segments", [])
        total_duration = chapter_info.get("total_duration_seconds")

        if total_duration is None:
            print(f"Missing total_duration_seconds for {filename}. Skipping chapter.")
            continue

        for i, segment in enumerate(tqdm(segments, desc=f"Processing {filename}")):
            try:
                offset = segment["starting_time_seconds"]
                text = segment.get("text", "")

                # Calculate duration using the next segment or total_duration
                next_offset = (
                    segments[i + 1]["starting_time_seconds"]
                    if i + 1 < len(segments)
                    else total_duration
                )
                duration = next_offset - offset

                if duration <= 0:
                    print(f"Invalid duration for segment {i} in {filename}: {duration:.2f}s. Skipping.")
                    continue

                samples, sr = load_audio(file_path, offset=offset, duration=duration)

                print(f"{filename} segment {i}: start={offset:.2f}s, end={next_offset:.2f}s, "
                      f"duration={duration:.2f}s, samples={len(samples)}, text=\"{text[:80]}...\"")

                if len(samples) < sr * 0.5:
                    print(f"Skipping short segment {i} in {filename}")
                    continue

                # Save the segment as a .wav file
                segment_filename = f"{os.path.splitext(filename)[0]}_segment{i}.wav"
                segment_path = os.path.join(save_dir, segment_filename)
                sf.write(segment_path, samples, sr)

                # Extract features
                features = extract_all_features(file_path, samples, sr)
                features["filename"] = segment_filename
                features["label"] = label
                features["text"] = text
                features_list.append(features)

            except Exception as e:
                print(f"Error processing segment {i} of {filename}: {e}")

    return features_list


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # in seconds


# One method was splitting the real audio into the mean fake clip durations
"""fake_durations = []
for subfolder in os.listdir(fake_clips_root):
    full_path = os.path.join(fake_clips_root, subfolder)
    if os.path.isdir(full_path):
        for filename in os.listdir(full_path):
            if filename.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                path = os.path.join(full_path, filename)
                try:
                    duration = get_audio_duration(path)
                    fake_durations.append(duration)
                except Exception as e:
                    print(f"Error reading {path}: {e}")

mean_fake_duration = np.mean(fake_durations)
print(f"Mean fake duration: {mean_fake_duration:.2f} seconds")"""

# Other
json_path = os.path.join(base_dir, 'anthem_transcriptions.json')
corpus_path = os.path.join(base_dir, 'corpus')

real_data = process_segments_from_json(json_path, real_audio_folder, "Real", corpus_path)
"""
fake_data = []
for subfolder in os.listdir(fake_clips_root):
    full_path = os.path.join(fake_clips_root, subfolder)
    if os.path.isdir(full_path):
        fake_data.extend(process_folder(full_path, "Fake"))

df_all = pd.DataFrame(real_data + fake_data)
print(df_all.round(2))

# Optional: save to CSV
df_all.to_csv("audio_features_dataset.csv", index=False)"""
import os
import random
import shutil
from collections import defaultdict
import math  # for ceiling and floor

fake_clips_root = r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\fake_clips"
fake_validation_folder = r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\validation"
fake_test_folder = r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\test"
"""
os.makedirs(fake_validation_folder, exist_ok=True)
os.makedirs(fake_test_folder, exist_ok=True)

# Group files by chapter folder name (assumes folder name = chapter name)
chapter_files = defaultdict(list)
for subfolder in os.listdir(fake_clips_root):
    full_path = os.path.join(fake_clips_root, subfolder)
    if os.path.isdir(full_path):
        for filename in os.listdir(full_path):
            if filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                chapter_files[subfolder].append(os.path.join(full_path, filename))

random.seed(42)  # for reproducibility

for chapter, files in chapter_files.items():
    random.shuffle(files)
    n = len(files)
    val_count = math.ceil(n / 2)   # larger half to validation
    test_count = n - val_count    # remaining to test

    val_files = files[:val_count]
    test_files = files[val_count:]

    # Copy validation files
    for f in val_files:
        shutil.copy(f, fake_validation_folder)
    # Copy test files
    for f in test_files:
        shutil.copy(f, fake_test_folder)

    print(f"Chapter {chapter}: copied {len(val_files)} to validation, {len(test_files)} to test")

print("Stratified split by chapter complete.")
"""

import os
import random
import shutil
from collections import defaultdict

random.seed(42)

corpus_folder = r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\corpus"           # where all corpus segments are now
validation_folder = r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\validation"   # target folder for validation files
test_folder = r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\test"               # target folder for test files
"""
# Create target folders if not existing
for folder in [validation_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Organize files by chapter based on filename pattern
chapter_files = defaultdict(list)
for filename in os.listdir(corpus_folder):
    if filename.lower().endswith(('.wav', '.mp3')):
        # Assuming filenames like "chapter_01_segment_03.wav"
        # Adjust this split logic if your naming differs
        parts = filename.split('_')
        if len(parts) >= 2:
            chapter = parts[1]  # e.g. "01" from "chapter_01_segment_03.wav"
            chapter_files[chapter].append(filename)
        else:
            print(f"Filename {filename} does not match expected pattern.")

# For each chapter, move 15% of files to validation and 15% to test
for chapter, files in chapter_files.items():
    random.shuffle(files)
    n = len(files)
    n_val = max(1, int(n * 0.15))  # at least 1 file if possible
    n_test = max(1, int(n * 0.15))

    val_files = files[:n_val]
    test_files = files[n_val:n_val + n_test]
    remaining_files = files[n_val + n_test:]

    # Move validation files
    for f in val_files:
        src = os.path.join(corpus_folder, f)
        dst = os.path.join(validation_folder, f)
        shutil.move(src, dst)

    # Move test files
    for f in test_files:
        src = os.path.join(corpus_folder, f)
        dst = os.path.join(test_folder, f)
        shutil.move(src, dst)

    print(f"Chapter {chapter}: {len(val_files)} to validation, {len(test_files)} to test, {len(remaining_files)} remain in corpus")

print("Stratified split complete.")
"""
df_all = pd.read_csv(r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\audio_features_dataset.csv")
features = [col for col in df_all.columns if col not in ["filename", "label", "text"]]


"""
# Pairplot to visualize separation
pairplot = sns.pairplot(df_all, hue='label', diag_kind='kde', corner=True)

# Adjust spacing to avoid cutting off labels
pairplot.fig.subplots_adjust(top=0.95, left=0.05, bottom=0.05, right=0.95)

# Add a super title
plt.suptitle("Pairplot of Spectral Features (Incl. Speech Rate)", fontsize=16, y=1.02)

# Show the plot
plt.show()"""
"""sns.set(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Variables to plot (excluding filename and label)
features_to_plot = [col for col in df_all.columns if col not in ['filename', 'label']]

df_all[features_to_plot].isna().sum().sort_values(ascending=False)

# Number of plots per row
cols = 3
rows = int(np.ceil(len(features_to_plot) / cols))

# Create subplots
fig, axs = plt.subplots(rows, cols, figsize=(18, 5 * rows))
axs = axs.flatten()

# Plot histograms with KDE for each feature
for i, feature in enumerate(features_to_plot[:]):
    sns.histplot(data=df_all, x=feature, hue='label', kde=True, ax=axs[i], palette=['#2e86c1', '#e74c3c'], element='step', stat='density', common_norm=False)
    axs[i].set_title(f"Distribution of {feature}", fontsize=13)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel("Density")

# Remove any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()"""

df_real = df_all[df_all["label"] == "Real"]
df_fake = df_all[df_all["label"] == "Fake"]

def build_kde_estimators(df_real, df_fake, features):
    kde_real = {}
    kde_fake = {}
    for feat in features:
        kde_real[feat] = gaussian_kde(df_real[feat])
        kde_fake[feat] = gaussian_kde(df_fake[feat])
    return kde_real, kde_fake

def classify_with_kde(file_path, kde_real, kde_fake, features):
    try:
        y, sr = load_audio(file_path)
        features_vals = extract_all_features(file_path, y, sr)
        
        weighted_votes = {'REAL': 0.0, 'FAKE': 0.0}
        feature_results = {}
        
        for feat in features:
            val = features_vals.get(feat, None)
            if val is None:
                continue  # skip missing features
            
            p_real = kde_real[feat].evaluate(val)[0]
            p_fake = kde_fake[feat].evaluate(val)[0]
            
            total_p = p_real + p_fake
            if total_p == 0:
                prob_real = prob_fake = 0.5
            else:
                prob_real = p_real / total_p
                prob_fake = p_fake / total_p
            
            if prob_real > prob_fake:
                decision = 'REAL'
                confidence = prob_real - prob_fake
            else:
                decision = 'FAKE'
                confidence = prob_fake - prob_real
            
            weighted_votes[decision] += confidence
            
            feature_results[feat] = {
                'value': val,
                'prob_real': prob_real,
                'prob_fake': prob_fake,
                'decision': decision,
                'confidence': confidence
            }
        
        # Final decision and confidence
        total_confidence = weighted_votes['REAL'] + weighted_votes['FAKE']
        if total_confidence == 0:
            final_pred = 'UNDECIDED'
            final_conf = 0.0
        elif weighted_votes['REAL'] > weighted_votes['FAKE']:
            final_pred = 'REAL'
            final_conf = weighted_votes['REAL'] / total_confidence
        elif weighted_votes['FAKE'] > weighted_votes['REAL']:
            final_pred = 'FAKE'
            final_conf = weighted_votes['FAKE'] / total_confidence
        else:
            final_pred = 'UNDECIDED'
            final_conf = 0.0
        
        return {
            'prediction': final_pred,
            'confidence': f"{final_conf*100:.1f}%",
            'weighted_votes': {k: float(v) for k, v in weighted_votes.items()},
            'features': feature_results
        }
        
    except Exception as e:
        return {'error': str(e)}


# Build KDE models once
kde_real, kde_fake = build_kde_estimators(df_real, df_fake, features)

test_files = [
    r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\ElevenLabs_Text_to_Speech_audio.wav",
    r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\test_real\real_01.wav",
    r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\test_real\real_02.wav",
    r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\test_real\real_03.wav",
    r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\test_real\real_04.wav",
    r"C:\Users\danie\OneDrive\Documents\Coding\Fake Audio Detection\fake_clips\Chap_5\tmpfokcab5w.wav"
]
for file in test_files:
    if os.path.exists(file):
        result = classify_with_kde(file, kde_real, kde_fake, features)
        print(f"\nFile: {os.path.basename(file)}")
        print(f"Final Decision: {result['prediction']} (Confidence: {result['confidence']})")
        print("Weighted votes:", result['weighted_votes'])
        print("Feature details:")
        for feat, info in result['features'].items():
            print(f"  {feat:20}: value={info['value']:.3f}, real_prob={info['prob_real']:.3f}, fake_prob={info['prob_fake']:.3f}, decision={info['decision']}, confidence={info['confidence']:.3f}")
    else:
        print(f"\nFile not found: {file}")
import matplotlib.pyplot as plt

"""
import matplotlib.pyplot as plt
import numpy as np

for feat in features:
    # Prepare the x-axis grid
    vals = np.concatenate([df_real[feat], df_fake[feat]])
    x_grid = np.linspace(min(vals), max(vals), 1000)

    # Evaluate KDEs
    kde_r = kde_real[feat](x_grid)
    kde_f = kde_fake[feat](x_grid)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, kde_r, label="REAL", color="blue")
    plt.plot(x_grid, kde_f, label="FAKE", color="red")
    plt.title(f"KDE Plot for Feature: {feat}", fontsize=14)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional pause if running as a script
    # input("Press Enter to see next plot...")

for feat in features:
    plt.figure(figsize=(8, 5))
    
    # Histogram for REAL
    plt.hist(df_real[feat], bins=30, alpha=0.6, label='REAL', color='blue', density=True)
    
    # Histogram for FAKE
    plt.hist(df_fake[feat], bins=30, alpha=0.6, label='FAKE', color='red', density=True)
    
    plt.title(f"Histogram for Feature: {feat}", fontsize=14)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""

    # Optional pause for script use
    # input("Press Enter to see next histogram...")

def classify_row_kde_from_df(row):
    weighted_votes = {'REAL': 0.0, 'FAKE': 0.0}
    for feat in features:
        val = row[feat]
        p_real = kde_real[feat].evaluate([val])[0]
        p_fake = kde_fake[feat].evaluate([val])[0]
        total = p_real + p_fake
        if total == 0:
            prob_real = prob_fake = 0.5
        else:
            prob_real = p_real / total
            prob_fake = p_fake / total
        if prob_real > prob_fake:
            decision = 'REAL'
            confidence = prob_real - prob_fake
        else:
            decision = 'FAKE'
            confidence = prob_fake - prob_real
        weighted_votes[decision] += confidence
    if weighted_votes['REAL'] > weighted_votes['FAKE']:
        return 'REAL'
    elif weighted_votes['FAKE'] > weighted_votes['REAL']:
        return 'FAKE'
    else:
        return 'UNDECIDED'

# Run classification over entire df_all
df_all["prediction"] = df_all.apply(classify_row_kde_from_df, axis=1)

# Confusion matrix
y_true = df_all["label"].str.upper()
y_pred = df_all["prediction"]

cm = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KDE Classifier Confusion Matrix")
plt.show()

# Optional: Detailed metrics
print("\nAccuracy:")
print(accuracy_score(y_true=y_true, y_pred=y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))


def feature_importance_from_confidence(df_all, kde_real, kde_fake, features):
    #Calculate feature importance based on confidence scores
    feature_importance = {feat: 0 for feat in features}
    counts = {feat: 0 for feat in features}
    
    for _, row in df_all.iterrows():
        for feat in features:
            val = row[feat]
            p_real = kde_real[feat].evaluate([val])[0]
            p_fake = kde_fake[feat].evaluate([val])[0]
            total = p_real + p_fake
            if total > 0:
                confidence = abs(p_real - p_fake) / total
                feature_importance[feat] += confidence
                counts[feat] += 1
    
    # Normalize by counts
    for feat in features:
        if counts[feat] > 0:
            feature_importance[feat] /= counts[feat]
    
    return feature_importance

# Calculate importance
conf_importance = feature_importance_from_confidence(df_all, kde_real, kde_fake, features)

# Sort features
sorted_conf = sorted(conf_importance.items(), key=lambda x: x[1], reverse=True)

# Plot
plt.figure(figsize=(12, 8))
features_sorted = [x[0] for x in sorted_conf]
importance_sorted = [x[1] for x in sorted_conf]
plt.barh(features_sorted, importance_sorted)
plt.xlabel('Average Confidence Contribution')
plt.title('Feature Importance based on Confidence Scores')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

def classify_row_kde_from_df(row):
    #Modfied to return both prediction and confidence
    weighted_votes = {'REAL': 0.0, 'FAKE': 0.0}
    feature_details = {}
    
    for feat in features:
        val = row[feat]
        p_real = kde_real[feat].evaluate([val])[0]
        p_fake = kde_fake[feat].evaluate([val])[0]
        total = p_real + p_fake
        
        if total == 0:
            prob_real = prob_fake = 0.5
        else:
            prob_real = p_real / total
            prob_fake = p_fake / total
            
        if prob_real > prob_fake:
            decision = 'REAL'
            confidence = prob_real - prob_fake
        else:
            decision = 'FAKE'
            confidence = prob_fake - prob_real
            
        weighted_votes[decision] += confidence
        feature_details[feat] = confidence
    
    total_confidence = weighted_votes['REAL'] + weighted_votes['FAKE']
    
    if total_confidence == 0:
        final_confidence = 0.0
        prediction = 'UNDECIDED'
    elif weighted_votes['REAL'] > weighted_votes['FAKE']:
        prediction = 'REAL'
        final_confidence = weighted_votes['REAL'] / total_confidence
    else:
        prediction = 'FAKE'
        final_confidence = weighted_votes['FAKE'] / total_confidence
    
    return pd.Series({
        'prediction': prediction,
        'confidence': final_confidence,
        'feature_confidences': feature_details
    })

# Run classification over entire df_all and get confidence scores
df_all[['prediction', 'confidence', 'feature_confidences']] = df_all.apply(
    classify_row_kde_from_df, axis=1)

# Confusion matrix and metrics
y_true = df_all["label"].str.upper()
y_pred = df_all["prediction"]
confidences = df_all["confidence"]

cm = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KDE Classifier Confusion Matrix")
plt.show()

# Calculate overall confidence metrics
def calculate_confidence_metrics(y_true, y_pred, confidences):
    correct_conf = confidences[y_true == y_pred]
    incorrect_conf = confidences[y_true != y_pred]
    
    metrics = {
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'mean_confidence': np.mean(confidences),
        'median_confidence': np.median(confidences),
        'correct_mean_confidence': np.mean(correct_conf) if len(correct_conf) > 0 else 0,
        'incorrect_mean_confidence': np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0,
        'confidence_std': np.std(confidences),
        'confidence_range': (np.min(confidences), np.max(confidences)),
        'confidence_accuracy_correlation': np.corrcoef(
            [1 if p == t else 0 for p, t in zip(y_pred, y_true)],
            confidences
        )[0, 1]
    }
    return metrics

# Get confidence metrics
conf_metrics = calculate_confidence_metrics(y_true, y_pred, confidences)

# Print all metrics
print("\n=== Classification Metrics ===")
print(f"Accuracy: {conf_metrics['overall_accuracy']:.3f}")
print("\n=== Confidence Metrics ===")
print(f"Mean confidence: {conf_metrics['mean_confidence']:.3f}")
print(f"Median confidence: {conf_metrics['median_confidence']:.3f}")
print(f"Correct predictions mean confidence: {conf_metrics['correct_mean_confidence']:.3f}")
print(f"Incorrect predictions mean confidence: {conf_metrics['incorrect_mean_confidence']:.3f}")
print(f"Confidence std: {conf_metrics['confidence_std']:.3f}")
print(f"Confidence range: {conf_metrics['confidence_range']}")
print(f"Confidence-accuracy correlation: {conf_metrics['confidence_accuracy_correlation']:.3f}")

# Feature importance based on confidence contributions
feature_importance = {}
for feat in features:
    feat_confs = df_all['feature_confidences'].apply(lambda x: x.get(feat, 0))
    feature_importance[feat] = np.mean(feat_confs)

# Sort and display top features
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("\n=== Top 10 Features by Confidence Contribution ===")
for feat, importance in sorted_features[:10]:
    print(f"{feat:30}: {importance:.4f}")

# Plot confidence distribution
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=20, alpha=0.7)
plt.title('Distribution of Prediction Confidences')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()




def get_filenames(folder):
    return [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]

# Get filenames in each folder
corpus_files = set(get_filenames(corpus_folder))
validation_files = set(get_filenames(validation_folder))
test_files = set(get_filenames(test_folder))

# Filter dataframe by filenames in each folder
df_corpus = df_all[df_all['filename'].isin(corpus_files)].reset_index(drop=True)
df_validation = df_all[df_all['filename'].isin(validation_files)].reset_index(drop=True)
df_test = df_all[df_all['filename'].isin(test_files)].reset_index(drop=True)

print(f"Train label distribution: {Counter(df_corpus['label'])}")
print(f"Validation label distribution: {Counter(df_validation['label'])}")
print(f"Test label distribution: {Counter(df_test['label'])}")

# Use all columns except 'filename', 'label', 'text' as features
features = [col for col in df_all.columns if col not in ["filename", "label", "text"]]

# Prepare training data (corpus only contains Real)
X_train = df_corpus[features].values

# Compute mean vector and covariance matrix of the corpus (Real only)
mean_vec = np.mean(X_train, axis=0)
cov_matrix = np.cov(X_train, rowvar=False)

# Invert covariance matrix (add a tiny value to diagonal for numerical stability if needed)
epsilon = 1e-10
cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
inv_cov_matrix = np.linalg.inv(cov_matrix)

def compute_mahalanobis_distances(X, mean_vec, inv_cov_matrix):
    #Compute Mahalanobis distances for each row in X.
    distances = []
    for x in X:
        dist = mahalanobis(x, mean_vec, inv_cov_matrix)
        distances.append(dist)
    return np.array(distances)

def predict_mahalanobis(X, mean_vec, inv_cov_matrix, threshold):
    #Predict 'Real' if Mahalanobis distance <= threshold else 'Fake'.
    #Returns predicted labels, distances, and confidence scores.
    #Confidence = 1 - normalized_distance (scaled between 0 and 1).

    distances = compute_mahalanobis_distances(X, mean_vec, inv_cov_matrix)
    preds = ['Real' if d <= threshold else 'Fake' for d in distances]
    
    # Normalize distances to [0,1] for confidence
    d_min, d_max = distances.min(), distances.max()
    norm_dist = (distances - d_min) / (d_max - d_min + 1e-10)
    confidences = 1 - norm_dist  # Closer to mean = higher confidence
    
    return preds, distances, confidences

# Prepare validation and test data
X_val = df_validation[features].values
y_val = df_validation['label'].values

X_test = df_test[features].values
y_test = df_test['label'].values

# Compute validation distances without threshold to find candidate thresholds
_, val_distances, _ = predict_mahalanobis(X_val, mean_vec, inv_cov_matrix, threshold=np.inf)

# Sweep thresholds across range of validation distances to tune best threshold
candidate_thresholds = np.linspace(val_distances.min(), val_distances.max(), 100)

best_threshold = None
best_f1 = -1

for thresh in candidate_thresholds:
    preds = ['Real' if d <= thresh else 'Fake' for d in val_distances]
    score = f1_score(y_val, preds, pos_label='Fake')
    if score > best_f1:
        best_f1 = score
        best_threshold = thresh

print(f"Best threshold based on validation F1-score: {best_threshold:.4f} with F1: {best_f1:.4f}")

# Evaluate on validation with best threshold
val_preds, val_distances, val_confidences = predict_mahalanobis(X_val, mean_vec, inv_cov_matrix, best_threshold)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

# Evaluate on test with best threshold
test_preds, test_distances, test_confidences = predict_mahalanobis(X_test, mean_vec, inv_cov_matrix, best_threshold)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))

# Confusion matrix plot for test
cm = confusion_matrix(y_test, test_preds, labels=['Real', 'Fake'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real Predicted', 'Fake Predicted'], yticklabels=['Real True', 'Fake True'])
plt.title('Confusion Matrix on Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Optional: show some confidence scores summary on test set
print("\nConfidence scores summary on test set:")
print(f"Mean confidence for Real predictions: {np.mean([conf for conf, pred in zip(test_confidences, test_preds) if pred=='Real']):.3f}")
print(f"Mean confidence for Fake predictions: {np.mean([conf for conf, pred in zip(test_confidences, test_preds) if pred=='Fake']):.3f}")

fake_confidences = [conf for conf, pred in zip(test_confidences, test_preds) if pred == 'Fake']

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(fake_confidences, bins=10, color='salmon', edgecolor='black')
plt.title('Histogram of Confidence Scores for Fake Predictions')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)

# Optional: display the mean on the plot
mean_conf = np.mean(fake_confidences)
plt.axvline(mean_conf, color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean_conf:.3f}')
plt.legend()

plt.tight_layout()
plt.show()



"""
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ========================================================
# CORRECTED ISOLATION FOREST IMPLEMENTATION
# (With proper validation set usage)
# ========================================================

# 1. Prepare Data (using your existing splits)
X_train_iso = df_corpus[features].values  # Training data (real samples only)
X_val_iso = df_validation[features].values  # Validation data
y_val_iso = df_validation['label'].values
X_test_iso = df_test[features].values      # Test data (only for final eval)
y_test_iso = df_test['label'].values

# 2. Train Isolation Forest (on REAL samples only)
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination='auto',  # Let model estimate anomaly ratio
    random_state=42,
    verbose=1
)
iso_forest.fit(X_train_iso)

# 3. Tune Threshold on VALIDATION SET (not test set!)
val_scores = iso_forest.decision_function(X_val_iso)

# Find best threshold to maximize F1 for "Fake" class
best_thresh = 0
best_f1 = -1

for thresh in np.linspace(val_scores.min(), val_scores.max(), 100):
    val_preds = np.where(val_scores < thresh, 'Fake', 'Real')
    report = classification_report(y_val_iso, val_preds, output_dict=True, zero_division=0)
    f1 = report['Fake']['f1-score']
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Best Validation Threshold: {best_thresh:.4f} (F1={best_f1:.4f})")

# 4. Evaluate on TEST SET (final evaluation only)
test_scores = iso_forest.decision_function(X_test_iso)
test_preds_iso = np.where(test_scores < best_thresh, 'Fake', 'Real')

# Metrics
print("\n=== Final Test Results ===")
print("Test Accuracy:", accuracy_score(y_test_iso, test_preds_iso))
print(classification_report(y_test_iso, test_preds_iso, zero_division=0))

# Confusion Matrix
plt.figure(figsize=(8,6))
conf_mat = confusion_matrix(y_test_iso, test_preds_iso, labels=['Real', 'Fake'])
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title("Isolation Forest Test Set Confusion Matrix")
plt.show()

# Score Distribution (Validation vs Test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Validation set scores
ax1.hist(val_scores[y_val_iso == 'Real'], bins=50, alpha=0.5, label='Real')
ax1.hist(val_scores[y_val_iso == 'Fake'], bins=50, alpha=0.5, label='Fake')
ax1.axvline(best_thresh, color='red', linestyle='--', label='Optimal Threshold')
ax1.set_title("Validation Set Score Distribution")
ax1.legend()

# Test set scores
ax2.hist(test_scores[y_test_iso == 'Real'], bins=50, alpha=0.5, label='Real')
ax2.hist(test_scores[y_test_iso == 'Fake'], bins=50, alpha=0.5, label='Fake')
ax2.axvline(best_thresh, color='red', linestyle='--', label='Optimal Threshold')
ax2.set_title("Test Set Score Distribution")
ax2.legend()

plt.tight_layout()
plt.show()

# ==============================================
# NEW: Feature Importance Analysis
# ==============================================

def isolation_forest_feature_importance(model, X, features, n_iterations=5):
    Calculate feature importance by permutation importance
    baseline_scores = model.decision_function(X)
    feature_imp = np.zeros(len(features))
    
    for i, feature in enumerate(features):
        X_perturbed = X.copy()
        for _ in range(n_iterations):  # Multiple iterations for stability
            np.random.shuffle(X_perturbed[:, i])  # Shuffle only this feature
            perturbed_scores = model.decision_function(X_perturbed)
            # Importance = how much scores change when feature is shuffled
            feature_imp[i] += np.mean(baseline_scores - perturbed_scores)
        feature_imp[i] /= n_iterations
    
    return pd.Series(feature_imp, index=features).sort_values(ascending=False)

# Calculate importance on validation set (where we tuned the threshold)
feature_importances = isolation_forest_feature_importance(
    iso_forest, 
    X_val_iso, 
    features,
    n_iterations=5
)

print("\n=== Feature Importance ===")
print(feature_importances)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.head(20).plot(kind='barh')
plt.title('Top 20 Most Important Features (Isolation Forest)')
plt.xlabel('Importance Score (higher = more important)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
"""