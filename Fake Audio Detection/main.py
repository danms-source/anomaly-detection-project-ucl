import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from preprocessing import process_segments_from_json, clear_folder, stratify_split_fake, stratify_split_real, process_folder
from algorithm import count_abnormal_features, get_filenames, predict_zscore_anomaly, plot_feature_abnormality_diff, feature_abnormality_diff
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import shap


# base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# seaborn
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------------------------------------------------------------------
# Initial Spectogram Plot
# -------------------------------------------------------------------------------------------------------------------
real_file = os.path.join(base_dir, 'anthem_1308_librivox', 'converted_wav', 'anthem_01_rand_64kb.wav')
fake_file = os.path.join(base_dir, 'fake_clips', 'Chap_1', 'tmplk_7udax.wav')


y_real, sr_real = librosa.load(real_file, sr=None, duration=10, offset=22.4)
y_fake, sr_fake = librosa.load(fake_file, sr=None, duration=10)

n_fft = 2048
hop_length = 512
n_mels = 128

S_mel_real = librosa.feature.melspectrogram(y=y_real, sr=sr_real, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_mel_fake = librosa.feature.melspectrogram(y=y_fake, sr=sr_fake, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

S_db_real = librosa.power_to_db(S_mel_real, ref=np.max)
S_db_fake = librosa.power_to_db(S_mel_fake, ref=np.max)

fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex='col')

librosa.display.waveshow(y_real, sr=sr_real, ax=axs[0, 0], color='#2e86c1', alpha=0.8)
axs[0, 0].set_title("Waveform - Real Audio", fontsize=14, fontweight='bold')
axs[0, 0].set_ylabel("Amplitude")

librosa.display.waveshow(y_fake, sr=sr_fake, ax=axs[0, 1], color='#e74c3c', alpha=0.8)
axs[0, 1].set_title("Waveform - Fake Audio", fontsize=14, fontweight='bold')

img1 = librosa.display.specshow(S_db_real, sr=sr_real, hop_length=hop_length, x_axis='time', y_axis='mel',
                                cmap='magma', ax=axs[1, 0])
axs[1, 0].set_title("Mel Spectrogram - Real Audio", fontsize=14, fontweight='bold')
axs[1, 0].set_ylabel("Mel Frequency")
fig.colorbar(img1, ax=axs[1, 0], format="%+2.0f dB")

img2 = librosa.display.specshow(S_db_fake, sr=sr_fake, hop_length=hop_length, x_axis='time', y_axis='mel',
                                cmap='plasma', ax=axs[1, 1])
axs[1, 1].set_title("Mel Spectrogram - Fake Audio", fontsize=14, fontweight='bold')
fig.colorbar(img2, ax=axs[1, 1], format="%+2.0f dB")

for ax in axs[1, :]:
    ax.set_xlabel("Time (seconds)")
for ax in axs.flat:
    ax.label_outer()

plt.suptitle("Real vs Fake Audio: Waveform and Mel Spectrogram", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Compute spectral centroids
centroid_real = librosa.feature.spectral_centroid(y=y_real, sr=sr_real, hop_length=hop_length)[0]
centroid_fake = librosa.feature.spectral_centroid(y=y_fake, sr=sr_fake, hop_length=hop_length)[0]

# Time axis for centroid plots
t_real = librosa.frames_to_time(np.arange(len(centroid_real)), sr=sr_real, hop_length=hop_length)
t_fake = librosa.frames_to_time(np.arange(len(centroid_fake)), sr=sr_fake, hop_length=hop_length)

# Plot
plt.figure(figsize=(12, 5))

plt.plot(t_real, centroid_real, label='Real Audio', color='#2e86c1', alpha=0.8)
plt.plot(t_fake, centroid_fake, label='Fake Audio', color='#e74c3c', alpha=0.8)

plt.title('Spectral Centroid Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Centroid Frequency (Hz)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------------------

# File paths
real_audio_path = os.path.join(base_dir, 'anthem_1308_librivox', 'converted_wav')
fake_clips_path = os.path.join(base_dir, 'fake_clips')

# -------------------------------------------------------------------------------------------------------------------
# Converting real mp3 audio to wav for consistency
# -------------------------------------------------------------------------------------------------------------------

source_folder = os.path.join(base_dir, "anthem_1308_librivox")
target_folder = os.path.join(source_folder, "converted_wav")
#convert_mp3_to_wav(source_folder, target_folder)



# -------------------------------------------------------------------------------------------------------------------
# One method was splitting the real audio into the mean fake clip durations
# -------------------------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------------------------
# The method currently used is splitting the real audio into roughly equal segments based on pauses in speech for better accuracy
# -------------------------------------------------------------------------------------------------------------------

json_path = os.path.join(base_dir, 'anthem_transcriptions.json')
corpus_path = os.path.join(base_dir, 'corpus')
validation_path = os.path.join(base_dir, 'validation')
test_path = os.path.join(base_dir, 'test')

# -------------------------------------------------------------------------------------------------------------------
# Processing of the real and fake data feature extraction into a dataframe
# -------------------------------------------------------------------------------------------------------------------

real_data = process_segments_from_json(json_path, real_audio_path, "Real", corpus_path)
fake_data = []
for subfolder in os.listdir(fake_clips_path):
    full_path = os.path.join(fake_clips_path, subfolder)
    if os.path.isdir(full_path):
        fake_data.extend(process_folder(full_path, "Fake"))

df_all = pd.DataFrame(real_data + fake_data)
print(df_all.round(2))

# Save dataframe to csv
csv_path = os.path.join(base_dir, "audio_features_dataset.csv")
df_all.to_csv(csv_path, index=False)

# -------------------------------------------------------------------------------------------------------------------
# Stratify sampling of the real and fake audio
# -------------------------------------------------------------------------------------------------------------------

clear_folder(validation_path)
clear_folder(test_path)
stratify_split_fake(fake_clips_path=fake_clips_path, fake_validation_path=validation_path, fake_test_path=test_path)
stratify_split_real(corpus_path, validation_path, test_path)


# -------------------------------------------------------------------------------------------------------------------
# Algorithm Implementation (Non-ML)
# -------------------------------------------------------------------------------------------------------------------

df_all = pd.read_csv(os.path.join(base_dir, 'audio_features_dataset.csv'))
features = [col for col in df_all.columns if col not in ['filename', 'label', 'text']]

# Get filenames in each folder
corpus_files = set(get_filenames(corpus_path))
validation_files = set(get_filenames(validation_path))
test_files = set(get_filenames(test_path))

# Prepare data
df_corpus = df_all[df_all['filename'].isin(corpus_files)].reset_index(drop=True)
df_validation = df_all[df_all['filename'].isin(validation_files)].reset_index(drop=True)
df_test = df_all[df_all['filename'].isin(test_files)].reset_index(drop=True)

X_train = df_corpus[features].values
X_val = df_validation[features].values
y_val = df_validation['label'].values
X_test = df_test[features].values
y_test = df_test['label'].values

# Compute mean and std for each feature from real corpus
feature_means = np.mean(X_train, axis=0)
feature_stds = np.std(X_train, axis=0) + 1e-10  # Avoid division by zero

feature_thresh_values = np.linspace(1, 5.0, 100)  # Feature z-score threshold
abnormal_thresh_values = np.arange(1, len(features) + 1)  # Number of abnormal features

f1_scores = np.zeros((len(feature_thresh_values), len(abnormal_thresh_values)))

# Tune thresholds on validation set
best_f1 = -1
best_feature_thresh = None
best_abnormal_thresh = None

for i, feature_thresh in enumerate(feature_thresh_values):
    for j, abnormal_thresh in enumerate(abnormal_thresh_values):
        val_preds, _ = predict_zscore_anomaly(X_val, feature_means, feature_stds, feature_thresh, abnormal_thresh)
        f1 = classification_report(y_val, val_preds, output_dict=True)['Fake']['f1-score']
        f1_scores[i, j] = f1
        
        if f1 > best_f1:
            best_f1 = f1
            best_feature_thresh = feature_thresh
            best_abnormal_thresh = abnormal_thresh

print(f"Best feature z-threshold: {best_feature_thresh}, best feature count threshold: {best_abnormal_thresh}, best F1: {best_f1}")

plt.figure(figsize=(12,6))
sns.heatmap(f1_scores, annot=False, fmt=".2f", xticklabels=abnormal_thresh_values, yticklabels=np.round(feature_thresh_values,2), cmap="viridis")
plt.xlabel("Abnormal feature count threshold")
plt.ylabel("Feature z-score threshold")
plt.title("F1 Score on Validation Set for Z-Score Thresholds")
plt.show()

# Final prediction on test set
test_preds, test_confidences = predict_zscore_anomaly(X_test, feature_means, feature_stds, best_feature_thresh, best_abnormal_thresh)


# -------------------------------------------------------------------------------------------------------------------
# Algorithm Evaluation
# -------------------------------------------------------------------------------------------------------------------

print("Test Accuracy:", accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))

# -------------------------------------------------------------------------------------------------------------------
# Plot #1 (Confusion Matrix)
# -------------------------------------------------------------------------------------------------------------------

conf_mat = confusion_matrix(y_test, test_preds, labels=['Real', 'Fake'])
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Plot #2 (Histogram of confidence for fake predictions)
# -------------------------------------------------------------------------------------------------------------------
fake_conf = [c for c, p in zip(test_confidences, test_preds) if p == 'Fake']
plt.hist(fake_conf, bins=10, alpha=0.7, color='red')
plt.title("Confidence Histogram for Fake Predictions")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Plot #3 (Histogram of number of samples with number of abnormal features)
# -------------------------------------------------------------------------------------------------------------------

# Calculate abnormal counts for all test samples
abnormal_counts = count_abnormal_features(X_test, feature_means, feature_stds, best_feature_thresh)

# Create DataFrame for plotting
df_plot = pd.DataFrame({
    'abnormal_count': abnormal_counts,
    'label': y_test
})

# Plot histogram
plt.figure(figsize=(12, 6))

# Plot real and fake separately
plt.hist([df_plot[df_plot['label']=='Real']['abnormal_count'], 
          df_plot[df_plot['label']=='Fake']['abnormal_count']],
         bins=range(0, len(features)+2),  # One bin per possible count
         alpha=0.7,
         color=['green', 'red'],
         edgecolor='black',
         label=['Real', 'Fake'],
         stacked=True)

# Add threshold line
plt.axvline(x=best_abnormal_thresh,  # Center between bins
            color='black', 
            linestyle='--',
            linewidth=2,
            label=f'Decision Threshold ({best_abnormal_thresh} abnormal features)')

# Add labels and title
plt.title('Distribution of Abnormal Feature Counts\n(Real vs Fake Samples)')
plt.xlabel('Number of Abnormal Features')
plt.ylabel('Number of Samples')
plt.xticks(range(0, len(features)+1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Plot #4 (Feature z-score plot)
# -------------------------------------------------------------------------------------------------------------------

# Feature details
feature_name = 'mfcc_11_mean'
feature_index = features.index(feature_name)

# Get mean and std of the feature from training corpus
mean_val = feature_means[feature_index]
std_val = feature_stds[feature_index]

# Compute actual threshold value from z-score
z_line_value = mean_val + best_feature_thresh * std_val

# Filter Real and Fake samples in validation set
real_val = df_test[df_test['label'] == 'Real'][feature_name]
fake_val = df_test[df_test['label'] == 'Fake'][feature_name]

# Plot KDEs
plt.figure(figsize=(12, 6))
sns.kdeplot(df_corpus[feature_name], fill=True, label='Corpus (Train)', color='skyblue')
sns.kdeplot(real_val, fill=True, label='Validation Real', color='green', alpha=0.5)
sns.kdeplot(fake_val, fill=True, label='Validation Fake', color='red', alpha=0.5)

# Add z-threshold line
plt.axvline(z_line_value, color='black', linestyle='--', label=f'Z-threshold = {best_feature_thresh:.2f}δ')

plt.title(f"PDF Distributions of '{feature_name}'")
plt.xlabel(feature_name)
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Plot #5 (Feature effectiveness evaluation)
# -------------------------------------------------------------------------------------------------------------------

# Prepare real and fake samples from test set
real_mask = np.array(y_test) == 'Real'
X_real = X_test[real_mask]

fake_mask = np.array(y_test) == 'Fake'
X_fake = X_test[fake_mask]

df_feature_diff = feature_abnormality_diff(X_real, X_fake, feature_means, feature_stds, features, best_feature_thresh)
print(df_feature_diff)

# Plot the results
plot_feature_abnormality_diff(df_feature_diff)


# -------------------------------------------------------------------------------------------------------------------
# Algorithm Implementation (ML)
# -------------------------------------------------------------------------------------------------------------------

# -----------------------
# 1. Scale features
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# 2. Hyperparameter tuning on validation (based on F1)
# -----------------------
# Hyperparameter grid
contamination_values = [0.01, 0.05, 0.1, 0.15]
n_estimators_values = [100, 200, 300]
max_samples_values = [0.6, 0.8, 1.0]
max_features_values = [0.6, 0.8, 1.0]

best_f1 = 0
best_params = None
best_model = None

y_val_true_mapped = np.array([0 if l=="Real" else 1 for l in y_val])

for c in contamination_values:
    for n in n_estimators_values:
        for ms in max_samples_values:
            for mf in max_features_values:
                clf = IsolationForest(
                    contamination=c,
                    n_estimators=n,
                    max_samples=ms,
                    max_features=mf,
                    random_state=42
                )
                clf.fit(X_train_scaled)
                
                y_val_pred = clf.predict(X_val_scaled)
                y_val_pred_mapped = np.array([0 if p==1 else 1 for p in y_val_pred])
                
                f1 = f1_score(y_val_true_mapped, y_val_pred_mapped)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'contamination': c,
                                   'n_estimators': n,
                                   'max_samples': ms,
                                   'max_features': mf}
                    best_model = clf

print(f"\nBest F1 on validation: {best_f1:.3f}")
print("Best params:", best_params)

# -----------------------
# 3. Evaluate on test set
# -----------------------
y_test_true_mapped = np.array([0 if l=="Real" else 1 for l in y_test])
y_test_pred = best_model.predict(X_test_scaled)
y_test_pred_mapped = np.array([0 if p==1 else 1 for p in y_test_pred])

print("\nTest F1 Score:", f1_score(y_test_true_mapped, y_test_pred_mapped))
print("\nClassification Report:\n", classification_report(y_test_true_mapped, y_test_pred_mapped))

# -----------------------
# 4. Confusion Matrix
# -----------------------
cm = confusion_matrix(y_test_true_mapped, y_test_pred_mapped)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Isolation Forest Confusion Matrix")
plt.show()

# -----------------------
# 5. SHAP values
# -----------------------
explainer = shap.Explainer(best_model, X_test_scaled)
shap_values = explainer(X_test_scaled).values  

# True labels mapped: 0=Real, 1=Fake
y_true_mapped = np.array([0 if l=="Real" else 1 for l in y_test])


signed_correct = np.zeros(shap_values.shape[1])

for i in range(shap_values.shape[1]):
    feature_shaps = shap_values[:, i]
    # Real: positive shap = correct, negative = wrong
    # Fake: negative shap = correct, positive = wrong
    signed_correct[i] = np.sum(
        np.where(
            y_true_mapped == 0,  # Real
            feature_shaps,       # keep SHAP as-is
            -feature_shaps       # flip SHAP for Fake
        )
    )

# Create DataFrame and sort
feature_signed = pd.DataFrame({
    'feature': features,
    'signed_correct_shap': signed_correct
}).sort_values(by='signed_correct_shap', ascending=False)

print(feature_signed)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10,6))
colors = ['green' if x > 0 else 'red' for x in feature_signed['signed_correct_shap']]
plt.barh(feature_signed['feature'], feature_signed['signed_correct_shap'], color=colors)
plt.xlabel('Signed Correct SHAP Contribution')
plt.ylabel('Feature')
plt.title('Features Ranked by Correct/Incorrect Contribution')
plt.gca()
plt.show()