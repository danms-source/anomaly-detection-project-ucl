import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import random
from tqdm import tqdm  
from scipy.stats import norm

# Current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

corpus_path = os.path.join(base_dir, 'simulation_corpus')
validation_path = os.path.join(base_dir, 'simulation_validation_final', 'simulation_validation_index')
# Creates list of dataframes
def create_dfs(folder_path):
    samples = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv')])
    samples_df = [pd.read_csv(os.path.join(folder_path, file), encoding='utf-8-sig', engine='python') for file in samples]
    return samples_df, samples

# Extracts specific features from a path
def extract_features(path):
    features = {
        'path_range': float(np.ptp(path)) if len(path) > 0 else 0.0,
        'path_mean': float(np.mean(path)) if len(path) > 0 else 0.0,
        'path_std': float(np.std(path, ddof=1)) if len(path) > 0 else 0.0,
        'max_abs_inc': float(np.max(np.abs(np.diff(path)))) if len(path) > 1 else 0.0,
        'mean_inc': float(np.mean(np.diff(path))) if len(path) > 1 else 0.0,
        'std_inc': float(np.std(np.diff(path), ddof=1)) if len(path) > 1 else 0.0,
        'smoothness_ratio': float(np.std(np.diff(path)) / np.std(path)) if len(path) > 1 else 0.0,
    }

    return features

# Classification algorithm
def classify(path):
    features = extract_features(path)
    if features['std_inc'] > 0.14:
        return "SDE 2"
    
    elif features['std_inc'] < 0.0840:
            return "SDE 3"
        
    else:
        return "SDE 1"

# Computes accuracy of classifier
def evaluate_classifier(evaluation_samples, filenames):
    true_labels = []

    for name in filenames:
        num = int(name.split('_')[1].split('.')[0])

        if num <= 2000:
            true_labels.append('SDE 1')
        elif 2000 < num <= 3000:
            true_labels.append('SDE 3')
        else:  # num >= 3000
            true_labels.append('SDE 2')

    predicted_labels = [classify(sample['x1']) for sample in evaluation_samples]

    accuracy = accuracy_score(true_labels, predicted_labels)
    #print(f"Validation Accuracy: {accuracy:.4f}")
    #print(classification_report(true_labels, predicted_labels))
    return accuracy

def estimate_class_distributions(samples, filenames):
    std_inc_by_class = {'SDE 1': [], 'SDE 2': [], 'SDE 3': []}

    for sample, name in zip(samples, filenames):
        std_inc = extract_features(sample['x1'])['std_inc']
        num = int(name.split('_')[1].split('.')[0])
        if num <= 2000:
            std_inc_by_class['SDE 1'].append(std_inc)
        elif 2000 < num <= 3000:
            std_inc_by_class['SDE 3'].append(std_inc)
        else:
            std_inc_by_class['SDE 2'].append(std_inc)

    class_distributions = {}
    for label, values in std_inc_by_class.items():
        mu = np.mean(values)
        sigma = np.std(values)
        class_distributions[label] = norm(loc=mu, scale=sigma)
    
    return class_distributions

def get_confidence_gaussian(std_inc_value, predicted_label, class_distributions):
    pdfs = {}
    total_pdf = 0

    for label, dist in class_distributions.items():
        pdf = dist.pdf(std_inc_value)
        pdfs[label] = pdf
        total_pdf += pdf

    if total_pdf == 0:
        return 0.0  # To avoid division by zero if PDFs are too low

    confidence = pdfs[predicted_label] / total_pdf
    return round(confidence, 4)

def classify_with_confidence(path):
    std_inc_val = extract_features(path)['std_inc']
    predicted = classify(path)
    confidence = get_confidence_gaussian(std_inc_val, predicted, class_distributions)
    return f"Predicted: {predicted} (Confidence: {confidence:.2%})"


def plot_accuracy_vs_timestamps(samples, filenames, max_timestamps, step):
    timestamp_counts = list(range(1, max_timestamps + 1, step))
    accuracies = []

    for ts_count in tqdm(timestamp_counts, desc="Evaluating"):
        trimmed_samples = [sample.iloc[:ts_count].copy() for sample in samples]
        acc = evaluate_classifier(trimmed_samples, filenames)
        accuracies.append(acc)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamp_counts, accuracies, 'b-o', linewidth=2)
    plt.xlabel("Number of Timestamps Used")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Timestamps")
    plt.grid(True)
    plt.axhline(0.95, color='r', linestyle='--', label='95% Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.show()


# plots accuracy vs sample size
def plot_accuracy_vs_sample_size(samples, filenames, step=10):
        accuracies = []
        sample_sizes = range(1, len(samples) + 1, step) 

        paired = list(zip(samples, filenames))  # zip once outside loop

        for i in tqdm(sample_sizes, desc="Evaluatiing"):
            acc_for_i = []
            for _ in range(10):  # repeat 10 times
                sampled_pairs = random.sample(paired, i)
                sampled_dfs, sampled_filenames = zip(*sampled_pairs)
                sampled_dfs = list(sampled_dfs)
                sampled_filenames = list(sampled_filenames)

                accuracy = evaluate_classifier(sampled_dfs, sampled_filenames)
                acc_for_i.append(accuracy)

            avg_accuracy = np.mean(acc_for_i)
            accuracies.append(avg_accuracy)

        # Plot number of samples vs average accuracy
        plt.figure(figsize=(10,6))
        plt.plot(sample_sizes, accuracies, linestyle='-')
        plt.xlabel("Number of Samples")
        plt.ylabel("Average Classification Accuracy (over 10 runs)")
        plt.title("Sample Size vs Average Classification Accuracy")
        plt.grid(True)
        plt.show()

samples, filenames = create_dfs(validation_path)
class_distributions = estimate_class_distributions(samples, filenames)

# -----------------------------------------------------------------------------------------
# VISUALISATIONS
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# Plot of all paths

"""
plt.figure(figsize=(10, 6))

for i, sample in enumerate(samples):
    if i < 2000:
        color = 'blue'   # SDE 1
    elif i < 3000:
        color = 'red'    # SDE 2
    else:
        color = 'green'  # SDE 3
    plt.plot(sample['x1'], color=color, alpha=0.3, linewidth=0.5)

# Add legend
plt.plot([], [], 'blue', label='SDE 1 (2000 paths)')
plt.plot([], [], 'red', label='SDE 2 (1000 paths)')
plt.plot([], [], 'green', label='SDE 3 (1000 paths)')
plt.legend()

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('All SDE Realizations')
plt.show()"""

# -----------------------------------------------------------------------------------------
# Scatter plot of 2 variables

"""plt.figure(figsize=(8,8))

colors = []
for i in range(len(samples)):
    if i < 2000:        # First 2000 are SDE 1
        colors.append('blue')
    elif i < 3000:      # Next 1000 are SDE 2
        colors.append('red')
    else:               # Last 1000 are SDE 3
        colors.append('green')

# Plot with colored points
for idx, sample in enumerate(samples):
    features_df = extract_features(sample['x1'])
    plt.scatter(features_df['std_inc'], features_df['max_abs_inc'], 
               color=colors[idx], alpha=0.5, s=20)  


legend_elements = [
    mpatches.Patch(color='blue', label='SDE 1 (Inliers)'),
    mpatches.Patch(color='green', label='SDE 2 (High Noise)'),
    mpatches.Patch(color='red', label='SDE 3 (Filtered)')
]
plt.legend(handles=legend_elements)

plt.xlabel('Std of Increments')
plt.ylabel('Maximum Absolute Increment')
plt.tight_layout()
plt.show()
"""
#----------------------------------------------------------------------------------------
# 1 variable histogram plot

std_inc_sde1 = [extract_features(sample['x1'])['std_inc'] for sample in samples[:2000]]
std_inc_sde2 = [extract_features(sample['x1'])['std_inc'] for sample in samples[2000:3000]]
std_inc_sde3 = [extract_features(sample['x1'])['std_inc'] for sample in samples[3000:]]

all_values = std_inc_sde1 + std_inc_sde2 + std_inc_sde3
bins = np.histogram_bin_edges(all_values, bins=100)

# Plot stacked histogram
plt.hist(
    [std_inc_sde1, std_inc_sde3, std_inc_sde2],
    bins=bins,
    stacked=True,
    color=['blue', 'green', 'red'],
    label=['SDE 1 (Inliers)', 'SDE 2 (High Noise)', 'SDE 3 (Filtered)'],
    edgecolor='black',
    alpha=0.7
)

# Add labels and legend
plt.xlabel('Standard Deviation of Increments (σ)')
plt.ylabel('Frequency')
plt.title('Distribution of Std. of Increments by SDE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------
# Sample size vs Accuracy plot & Timestamps vs Accuracy plots (can take a while depending on the variables you input)

"""plot_accuracy_vs_timestamps(samples, filenames, max_timestamps=200, step=5)"""
plot_accuracy_vs_sample_size(samples, filenames, step=100)

#----------------------------------------------------------------------------------------
# Using the classifier

for i in range(10):
    print(classify_with_confidence(samples[i]['x1']))

# Evaluating the classifer
print(f"Accuracy: {evaluate_classifier(samples, filenames)}")
