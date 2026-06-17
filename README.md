# Padel vs Tennis Network Analysis on Bluesky

This repository contains a Jupyter Notebook project comparing padel and tennis discourse on Bluesky through content analysis and social network analysis.

The project includes the notebook, reusable helper code, the CSV dataset needed to reproduce the analysis, and the final project report.

## Main analyses

- Bluesky post collection and cleaning
- Mention-network construction
- Graph-level network statistics
- Centrality analysis
- Community detection
- Bridge-user detection
- Network visualization
- Text preprocessing
- Sentiment and emotion analysis
- Named Entity Recognition (NER)
- Word clouds and hashtag co-occurrence analysis

## Repository structure

```text
.
├── notebooks/
│   └── padel_vs_tennis_network_analysis_bluesky.ipynb
├── src/
│   └── analysis.py
├── data/
│   ├── raw/
│   │   └── raw_bluesky_posts.csv
│   ├── processed/
│   │   ├── clean_bluesky_posts.csv
│   │   ├── mention_edges.csv
│   │   └── network_nodes.csv
│   ├── results/
│   │   └── generated analysis outputs
│   └── README.md
├── figures/
├── report/
│   └── project_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset included

The uploaded dataset currently contains:

| File | Folder | Rows | Description |
|---|---:|---:|---|
| `raw_bluesky_posts.csv` | `data/raw/` | 1,231 | Raw collected Bluesky posts for the padel/tennis queries |
| `clean_bluesky_posts.csv` | `data/processed/` | 1,231 | Cleaned posts with topic/user classification and engagement |
| `mention_edges.csv` | `data/processed/` | 21 | Weighted mention-network edges |
| `network_nodes.csv` | `data/processed/` | 471 | Mention-network node attributes and degree measures |

Generated outputs, such as `centrality_full.csv`, `communities_louvain.csv`, `bridge_users.csv`, and `network_statistics.csv`, are written to `data/results/` when the notebook is executed.

## Project report

The final written report is included in:

```text
report/project_report.pdf
```

The report summarizes the objective, data collection strategy, methods, results, limitations, and conclusion of the analysis.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The notebook also uses NLTK resources. If they are not already available, run this once in Python:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## How to run

Open the notebook:

```bash
jupyter notebook notebooks/padel_vs_tennis_network_analysis_bluesky.ipynb
```

Then run the cells from top to bottom.

The notebook is configured to read:

- `data/raw/raw_bluesky_posts.csv`
- `data/processed/clean_bluesky_posts.csv`
- `data/processed/mention_edges.csv`
- `data/processed/network_nodes.csv`

## Optional: collect fresh Bluesky data

To collect new posts from Bluesky, set your credentials before running the collection cells:

```bash
export BLUESKY_HANDLE="your-handle.bsky.social"
export BLUESKY_APP_PASSWORD="your-app-password"
```

Do not commit credentials or app passwords to GitHub.

## Notes before publishing

Before pushing publicly, verify that the dataset does not contain private, sensitive, or non-redistributable information. The CSVs contain public social-media text and user handles, so you should confirm that sharing them is acceptable for your course/project rules.

You can also clear notebook outputs to make the repository lighter:

```bash
jupyter nbconvert --clear-output --inplace notebooks/padel_vs_tennis_network_analysis_bluesky.ipynb
```


