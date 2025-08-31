import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import json
import numpy as np
from tqdm import tqdm
import soundfile as sf  # for saving audio
import random
import shutil
from collections import defaultdict
import math  # for ceiling and floor

def load_audio(path, target_sr=16000, offset=0.0, duration=None):
    y, sr = librosa.load(path, sr=target_sr, mono=True, offset=offset, duration=duration)
    return y.astype(np.float32), sr


def convert_mp3_to_wav(source_folder, target_folder):
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

    # MFCC means and stds
    mfcc_means = {f"mfcc_{i+1}_mean": np.mean(mfccs[i]) for i in range(13)}
    mfcc_stds = {f"mfcc_{i+1}_std": np.std(mfccs[i]) for i in range(13)}

    # MFCC deltas (1st derivative)
    mfcc_deltas = librosa.feature.delta(mfccs)
    mfcc_delta_means = {f"mfcc_delta_{i+1}_mean": np.mean(mfcc_deltas[i]) for i in range(13)}
    mfcc_delta_stds = {f"mfcc_delta_{i+1}_std": np.std(mfcc_deltas[i]) for i in range(13)}

    # MFCC delta-deltas (2nd derivative)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfcc_delta2_means = {f"mfcc_delta2_{i+1}_mean": np.mean(mfcc_delta2[i]) for i in range(13)}
    mfcc_delta2_stds = {f"mfcc_delta2_{i+1}_std": np.std(mfcc_delta2[i]) for i in range(13)}

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

    # Add MFCC means/stds
    features.update(mfcc_means)
    features.update(mfcc_stds)

    """    # Add MFCC delta means/stds
    features.update(mfcc_delta_means)
    features.update(mfcc_delta_stds)

    # Add MFCC delta-delta means/stds
    features.update(mfcc_delta2_means)
    features.update(mfcc_delta2_stds)"""

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
    for filename in tqdm(os.listdir(folder_path), desc="Processing real audio"):
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

def clear_folder(folder):
    if os.path.exists(folder):
        # Remove all files and folders inside folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove folder and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder, exist_ok=True)  # Create folder if doesn't exist

def process_segments_from_json(json_path, folder_path, label, save_dir="corpus"):
    clear_folder(save_dir)
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


def stratify_split_fake(fake_clips_path, fake_validation_path, fake_test_path):
    os.makedirs(fake_validation_path, exist_ok=True)
    os.makedirs(fake_test_path, exist_ok=True)

    # Group files by chapter folder name (assumes folder name = chapter name)
    chapter_files = defaultdict(list)
    for subfolder in os.listdir(fake_clips_path):
        full_path = os.path.join(fake_clips_path, subfolder)
        if os.path.isdir(full_path):
            for filename in os.listdir(full_path):
                if filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    chapter_files[subfolder].append(os.path.join(full_path, filename))

    random.seed(21)   # Uncomment this to keep the same seed throughout

    for chapter, files in chapter_files.items():
        random.shuffle(files)
        n = len(files)
        val_count = math.ceil(n / 2)   # larger half to validation
        test_count = n - val_count    # remaining to test

        val_files = files[:val_count]
        test_files = files[val_count:]

        # Copy validation files
        for f in val_files:
            shutil.copy(f, fake_validation_path)
        # Copy test files
        for f in test_files:
            shutil.copy(f, fake_test_path)

        print(f"Chapter {chapter}: copied {len(val_files)} to validation, {len(test_files)} to test")

    print("Stratified split by chapter complete.")

def stratify_split_real(corpus_path, validation_path, test_path):
    random.seed(21)   # Uncomment this to keep the same seed throughout

    # Create target folders if not existing
    for folder in [validation_path, test_path]:
        os.makedirs(folder, exist_ok=True)

    # Organize files by chapter based on filename pattern
    chapter_files = defaultdict(list)
    for filename in os.listdir(corpus_path):
        if filename.lower().endswith(('.wav', '.mp3')):
            # Assuming filenames like "chapter_01_segment_03.wav"
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
            src = os.path.join(corpus_path, f)
            dst = os.path.join(validation_path, f)
            shutil.move(src, dst)

        # Move test files
        for f in test_files:
            src = os.path.join(corpus_path, f)
            dst = os.path.join(test_path, f)
            shutil.move(src, dst)

        print(f"Chapter {chapter}: {len(val_files)} to validation, {len(test_files)} to test, {len(remaining_files)} remain in corpus")

    print("Stratified split complete.")