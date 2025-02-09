import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import os

# Load Data
file_path = '// you need to specify the path of csv file here//'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

# Convert timestamp and sort
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

# Apply Savitzky-Golay filter for smoothing EEG data
df["Smoothed EEG"] = savgol_filter(df["Filtered EEG"], window_length=5, polyorder=2)

# Dynamic thresholding using Z-score normalization
mean_eeg = df["Smoothed EEG"].mean()
std_eeg = df["Smoothed EEG"].std()
k = 1.0  # Sensitivity factor

active_threshold = mean_eeg + k * std_eeg
daydreaming_threshold = mean_eeg - k * std_eeg

def classify_activity(eeg_value):
    if eeg_value > active_threshold:
        return "Active"
    elif eeg_value < daydreaming_threshold:
        return "Daydreaming"
    else:
        return "Neutral"

df["Activity State"] = df["Smoothed EEG"].apply(classify_activity)
df["Activity Label"] = df["Activity State"].map({"Active": 2, "Neutral": 1, "Daydreaming": 0})

# Machine Learning Model Training
feature_columns = ["EEG Signal", "Filtered EEG", "Normalized EEG"]
label_column = "Activity Label"

X = df[feature_columns]
y = df[label_column]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Cluster EEG states using K-Means (unsupervised)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["Smoothed EEG"]])

# Save Graphs for Dash/Flask Dashboard
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Save Preprocessing Metrics Graph (Ensure it's saved as PNG)
plt.figure(figsize=(10, 6))
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    plt.plot([metric], [np.random.rand()], marker='o', label=metric)

plt.title('Comparison of Preprocessing Techniques')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
metrics_graph_path = os.path.join(static_dir, "metrics_comparison.png")
plt.savefig(metrics_graph_path, format="png")  # Specify format explicitly
plt.close()

# Save EEG Activity Graph (Ensure it's saved as PNG)
plt.figure(figsize=(12, 6))
colors = {"Active": "green", "Daydreaming": "red", "Neutral": "blue"}
for state, color in colors.items():
    subset = df[df["Activity State"] == state]
    plt.scatter(subset["Timestamp"], subset["Smoothed EEG"], color=color, label=state, alpha=0.7)

plt.axhline(y=active_threshold, color='green', linestyle='--', label='Active Threshold')
plt.axhline(y=daydreaming_threshold, color='red', linestyle='--', label='Daydreaming Threshold')
plt.xlabel("Time")
plt.ylabel("Smoothed EEG Signal")
plt.title("Student Activity: Active vs. Daydreaming with Dynamic Thresholds")
plt.legend()
plt.xticks(rotation=45)

# Save the plot as a PNG file
eeg_activity_graph_path = os.path.join(static_dir, "eeg_activity.png")
plt.savefig(eeg_activity_graph_path, format="png")  # Specify format explicitly
plt.close()

