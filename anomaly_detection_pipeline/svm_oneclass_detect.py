import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set the root directory containing data files
root_folder = "/media/ming/ExtremeSSD/jiaming/processIccBot/atime"

# Define training and testing years
train_years = []
test_years = []

# Store results and confusion matrices
results = []
confusion_matrices = []

# Load both benign and malware data with labels
def load_data_with_labels(years):
    vectors = []
    labels = []

    for year in years:
        for category in ["benign", "malware"]:
            file_path = os.path.join(root_folder, str(year), category, "benign_clean_max_clean_anomalies.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep=';')
                for apk_name, group in df.groupby('apk_name'):
                    sim_vector = group['max_similarity'].values
                    max_len = 800
                    if len(sim_vector) > max_len:
                        sim_vector = sim_vector[:max_len]
                    elif len(sim_vector) < max_len:
                        sim_vector = np.pad(sim_vector, (0, max_len - len(sim_vector)), mode='constant')
                    vectors.append(sim_vector)
                    labels.append(0 if category == "benign" else 1)

    if not vectors:
        return None, None

    return np.array(vectors), np.array(labels)

# Load only benign data for training
def load_benign_data(years):
    benign_vectors = []

    for year in years:
        file_path = os.path.join(root_folder, str(year), "benign", "benign_clean_max_clean_anomalies.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=';')
            for apk_name, group in df.groupby('apk_name'):
                sim_vector = group['max_similarity'].values
                max_len = 800
                if len(sim_vector) > max_len:
                    sim_vector = sim_vector[:max_len]
                elif len(sim_vector) < max_len:
                    sim_vector = np.pad(sim_vector, (0, max_len - len(sim_vector)), mode='constant')
                benign_vectors.append(sim_vector)

    if not benign_vectors:
        return None

    return np.array(benign_vectors)

# Load training data (only benign samples)
X_train = load_benign_data(train_years)
if X_train is None:
    print("❌ Training data missing. Aborting.")
    exit()

# Train One-Class SVM model
ocsvm_model = OneClassSVM(kernel='linear', nu=0.1)
ocsvm_model.fit(X_train)

# Evaluate on the training set (should all be predicted as normal)
y_train_pred = ocsvm_model.predict(X_train)
y_train_pred = np.where(y_train_pred == -1, 1, 0)
train_labels = np.zeros(len(X_train))  # All benign

train_acc = accuracy_score(train_labels, y_train_pred)
train_report = classification_report(train_labels, y_train_pred, output_dict=True)

results.append({
    "Dataset": "Training",
    "Year": "2011-2017",
    "Precision": train_report["weighted avg"]["precision"],
    "Recall": train_report["weighted avg"]["recall"],
    "F1-score": train_report["weighted avg"]["f1-score"],
    "Accuracy": train_acc
})

# Iterate over test years
for year in test_years:
    X_test, y_test = load_data_with_labels([year])
    if X_test is None or y_test is None:
        print(f"❌ Test data missing for year {year}. Skipping...")
        continue

    y_pred = ocsvm_model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)

    # Evaluation
    test_acc = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Dataset": "Testing",
        "Year": year,
        "Precision": test_report["weighted avg"]["precision"],
        "Recall": test_report["weighted avg"]["recall"],
        "F1-score": test_report["weighted avg"]["f1-score"],
        "Accuracy": test_acc
    })

    confusion_matrices.append({
        "Year": year,
        "TN": int(cm[0][0]) if cm.shape[0] > 0 else 0,
        "FP": int(cm[0][1]) if cm.shape[1] > 1 else 0,
        "FN": int(cm[1][0]) if cm.shape[0] > 1 else 0,
        "TP": int(cm[1][1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    })

# Save evaluation results
output_file = os.path.join(root_folder, "ocsvm_results_full.csv")
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"✅ Results saved to {output_file}")

# Save confusion matrices
cm_file = os.path.join(root_folder, "confusion_matrices_ocsvm.csv")
pd.DataFrame(confusion_matrices).to_csv(cm_file, index=False)
print(f"✅ Confusion matrices saved to {cm_file}")

