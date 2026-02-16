# MBKGVISUAL

## Version 1.0.0

### By Skye Lane Goetz

MicrobiomeKG Visualization Tool. MBKGVisual Reads Compressed Graph Data (NODES and EDGES TSV Files), Builds a NetworkX Graph, Samples It Using PageRank-Based Sampling (25% of Nodes), and Generates an Interactive HTML Visualization Using Plotly.

## Usage

Install Dependencies
```bash
pip install plotly networkx polars pyarrow pandas backports.zstd scipy littleballoffur
```

Run Visualization
```bash
python3 ./main.py
```

This Reads `./NODES.tsv.zst` and `./EDGES.tsv.zst` and Outputs `./GRAPH.html`.

## Contributors

[Skye Lane Goetz](mailto:skye.lane.goetz@gmail.com)
