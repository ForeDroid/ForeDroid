
# ForeDroid Pipeline

This package implements a functional scenario-based clustering pipeline for Android malware analysis.

## Structure

- `entry_point_filter.py`: Extracts and cleans entry points from APKs, generates semantic vectors.
- `cluster_entry.py`: Performs PCA + MiniBatchKMeans clustering and generates wordclouds.
- `group_by_cluster.py`: Groups original APKs by cluster and exports per-cluster data.
- `api_stats.py`: Computes sensitive API usage statistics per cluster.
- `pipeline.py`: Orchestrates the full end-to-end pipeline.

## Usage

You can run the full pipeline via `pipeline.py`:

```bash
python pipeline.py --input_csv path/to/input.csv --output_dir path/to/output --num_clusters 2000
```

### Arguments

- `--input_csv`: Input CSV file containing at least `id`, `entry_point`, and optionally `entry_method`, `entry_point_actions`, `sensitiveAPI`.
- `--output_dir`: Directory to save the output clustering, models, wordclouds, and merged results.
- `--num_clusters`: Number of clusters (default: 2000).

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Recommended packages include:
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `wordcloud`, `spacy`, `gensim`, `nltk`, `joblib`

Also make sure to download required spaCy and NLTK models:

```bash
python -m spacy download en_core_web_sm
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
```

## Notes

- Requires a pre-trained word2vec model (e.g. `GoogleNews-vectors-negative300.bin`).
- Input CSVs must use `;` as field delimiter for intermediate files.
