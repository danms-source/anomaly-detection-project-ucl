import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def get_filenames(folder):
    return [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]

# Function to compute number of abnormal features in a sample
def count_abnormal_features(X, means, stds, feature_z_thresh):
    z_scores = np.abs((X - means) / stds)
    abnormal_counts = np.sum(z_scores > feature_z_thresh, axis=1)
    return abnormal_counts

# Function to make predictions based on number of abnormal features
def predict_zscore_anomaly(X, means, stds, feature_z_thresh, abnormal_feature_thresh):
    abnormal_counts = count_abnormal_features(X, means, stds, feature_z_thresh)
    preds = ['Fake' if count >= abnormal_feature_thresh else 'Real' for count in abnormal_counts]
    
    # Normalize abnormal count to [0, 1] and use 1 - norm as confidence
    max_abnormal = X.shape[1]
    confidences = 1 - (abnormal_counts / max_abnormal)
    return preds, confidences

def feature_abnormality_diff(X_real, X_fake, means, stds, feature_names, feature_z_thresh):
    z_scores = np.abs((X_real - means) / stds)
    abnormal_mask = (z_scores > feature_z_thresh).astype(int)
    feature_counts = abnormal_mask.sum(axis=0)
    freq_real = feature_counts / X_real.shape[0]

    z_scores = np.abs((X_fake - means) / stds)
    abnormal_mask = (z_scores > feature_z_thresh).astype(int)
    feature_counts = abnormal_mask.sum(axis=0)
    freq_fake =  feature_counts / X_fake.shape[0]
    
    
    diff = freq_fake - freq_real
    df = pd.DataFrame({
        'feature': feature_names,
        'abnormal_freq_real': freq_real,
        'abnormal_freq_fake': freq_fake,
        'diff_fake_real': diff
    }).sort_values(by='diff_fake_real', ascending=False)
    
    return df

def plot_feature_abnormality_diff(df_feature_diff, top_n=20):
    # Sort and select top features
    df_sorted = df_feature_diff
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df_sorted['feature'], df_sorted['diff_fake_real'], 
                    color=np.where(df_sorted['diff_fake_real'] > 0, 'green', 'red'))
    
    plt.xlabel('Difference in Abnormal Frequency (Fake - Real)')
    plt.title(f'Features with Largest Abnormal Frequency Differences')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x = width if width > 0 else width - 0.01
        plt.text(label_x, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 va='center', ha='right' if width > 0 else 'left',
                 color='white' if abs(width) > 0.02 else 'black')
    
    plt.tight_layout()
    plt.show()
