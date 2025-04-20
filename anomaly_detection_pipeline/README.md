# Anomaly Detection Pipeline

This package implements an anomaly detection pipeline for Android APK behavior descriptions.
It includes semantic vector computation, similarity-based anomaly detection, and One-Class SVM classification.

## Structure

- `similarity_detect.py`: Detects anomalies based on cosine similarity within clusters.
- `precompute_vectors.py`: Precomputes similarity vectors from LLM-generated behavior descriptions.
- `svm_oneclass_detect.py`: Uses One-Class SVM to detect anomalous APKs over time.
- `README.md`: This documentation.

## Usage

### Precompute similarity vectors
```bash
python precompute_vectors.py --merged_file path/to/merged.csv --cluster_dir path/to/clusters --output_file path/to/output_similarity.csv
```

### Run similarity-based anomaly detection
```bash
python similarity_detect.py
```

### Run One-Class SVM anomaly detection
```bash
python svm_oneclass_detect.py --input_file path/to/similarity.csv 
```

## Requirements

- Python 3.7+
- `sentence-transformers`
- `scikit-learn`
- `pandas`, `numpy`, `tqdm`

Make sure to configure the `HTTP_PROXY` and `HTTPS_PROXY` environment variables if needed.
