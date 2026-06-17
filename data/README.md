# Data folder

This folder contains the CSV files needed to reproduce the project.

## Current files

```text
data/
├── raw/
│   └── raw_bluesky_posts.csv
├── processed/
│   ├── clean_bluesky_posts.csv
│   ├── mention_edges.csv
│   └── network_nodes.csv
└── results/
    └── generated CSV outputs from the notebook
```

## File roles

- `raw/raw_bluesky_posts.csv`: original collected Bluesky posts.
- `processed/clean_bluesky_posts.csv`: cleaned post-level dataset used for content analysis.
- `processed/mention_edges.csv`: weighted directed edges for the mention network.
- `processed/network_nodes.csv`: node metadata and degree measures for network users.
- `results/`: notebook output folder for files such as centrality scores, community assignments, bridge users, network statistics, hashtag outputs, and post-level summaries.

## Privacy note

These files may contain public user handles and post text. Before publishing publicly on GitHub, check your course instructions and confirm that redistributing the data is allowed.
