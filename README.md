# ForeDroid Analysis Toolkit

This repository contains scripts and tools for extracting scenario information, sensitive API call chains, clustering functional behavior scenarios, and detecting anomalous behaviors in Android APKs. These serve as the static analysis foundation for **ForeDroid**.

---

## Directory Structure

### 1. `extract_Gui/` — GUI Scenario Extraction via Backstage

```
extract_Gui/
├── android-18/             # Android platform JARs
├── jars/                   # JARs for GUI analysis
├── output/                 # Output directory after GUI extraction
│   └── [apk_name]/
│       ├── appSerialized.txt      # Serialized intermediate result
│       └── AndroidManifest.xml    # Manifest file
├── res/                    # GUI layout resources
├── tool/                   # Additional tools (optional)
├── Backstage.jar          # Unified JAR for GUI extraction
├── run_backstage.sh       # GUI extraction shell script
```

### 2. `extract_CallChain/` — Sensitive API Call Chain Extraction via ICCBot

```
extract_CallChain/
├── apk/                   # Input APKs
├── config/                # ICCBot configuration files
├── lib/                   # Required libraries for ICCBot
├── output/                # Output directory for call graphs and reports
├── resources/             # ICCBot internal resource files
├── ICCBot.jar             # Main ICCBot executable
├── runICCBot.sh           # Call chain extraction shell script
```

### 3. `foredroid_static_pipeline.py` — Static Behavior Representation Integration

A unified script that combines:
- ICCBot outputs (`rule_analysis_report.json`)
- Backstage API usage reports
- GUI contextual information  
It outputs a CSV enriched with call chains, entry point semantics, and contextual UI features, generating LLM-based behavior descriptions.

Run:
```bash
python foredroid_static_pipeline.py --input_dir <iccbot_output> --backstage_api_csv <gui_info.csv> --final_csv <output.csv>
```

---

## Module 1: GUI Information Extraction with Backstage

### Step 1: Run GUI Analysis
```bash
bash run_backstage.sh /path/to/apk/dir
```

### Step 2: Extract API Call Info from GUI Context
```bash
java -jar jars/Backstage-APIResultsProcessor.jar -i output -o api-results
```

### Step 3: Extract GUI Scenario Labels
```bash
java -jar jars/Backstage-UIAnalysis.jar output outputBenignUi
```

---

## Module 2: Sensitive API Call Chain Extraction with ICCBot

### Step 1: Run ICCBot
```bash
bash runICCBot.sh
```

### Output:
```
output/[apk_name]/CallGraphInfo/rule_analysis_report.json
```

---

## Module 3: Functional Scenario Clustering (`cluster_pipeline/`)

```
cluster_pipeline/
├── entry_point_filter.py              # Entry point + action cleaning and embedding
├── cluster_entry_vec.py               # PCA + KMeans clustering
├── group_origin_csv_by_cluster.py     # Group CSV by cluster labels
├── cluster_api_stats.py               # Per-cluster API usage statistics
├── run_cluster_pipeline.py            # Main clustering entry script
```

Run:
```bash
python run_cluster_pipeline.py --input_csv <final_csv> --output_dir <output> --num_clusters 2000
```

---

## Module 4: Anomaly Detection (`anomaly_detection_pipeline/`)

```
anomaly_detection_pipeline/
├── generate_embeddings.py             # Use LLM to encode behavior descriptions
├── anomaly_detection.py               # Compute intra-cluster similarity & detect anomalies
├── llm_report_generator.py            # Generate structured explanations for high-risk behaviors
├── run_anomaly_pipeline.py            # Unified anomaly detection script
```

Run:
```bash
python run_anomaly_pipeline.py --clustered_csv <clustered_csv> --llm_model deepseek
```

---

## Output Summary

### From `extract_Gui/`:
- `appSerialized.txt`, `api-results/`, `outputBenignUi/`

### From `extract_CallChain/`:
- `rule_analysis_report.json`

### From `cluster_pipeline/`:
- `entry_point_clustered_with_labels.csv`
- `cluster_id_mapping.csv`
- `cluster_api_stats.json`

### From `anomaly_detection_pipeline/`:
- `anomaly_scores.csv`
- `malware_explanation_report.json`


---

Feel free to open an issue if you encounter any problems!
