# MBKGVISUAL

## Version 1.0.0

### By Skye Lane Goetz

MicrobiomeKG Visualization Tool. MBKGVisual reads compressed graph data (NODES and EDGES TSV files), builds a NetworkX graph, samples it using degree-based neighbor expansion (0.1% of nodes), and generates an interactive HTML visualization using Pyvis.

[Live Demo](https://skyeav.github.io/MBKGVisual/)

## Usage

Install Dependencies
```bash
pip install pyvis networkx polars pandas pyarrow backports.zstd scipy
```

Run Visualization
```bash
python3 ./main.py
```

This reads `./NODES.tsv.zst` and `./EDGES.tsv.zst` and outputs `./GRAPH.html`.

## Architecture

The visualization pipeline is implemented in `main.py`:

1. **read_tsv** - Decompress zstd TSV files into Polars DataFrames
2. **is_sig** - Filter edges where `significant == "YES"`
3. **sanitize** - Remove invalid nodes (`"1"`, `"2"`, `"3"`, `"4"`) and self-loops
4. **is_directed** - Split edges: `biolink:affects` is directed, others are undirected
5. **mkgraph** - Build NetworkX DiGraph (undirected edges get symmetric pairs)
6. **sampler** - Degree-based neighbor expansion sampling (0.1% of nodes)
7. **mkvis** - Generate interactive HTML via pyvis

## Contributors

[Skye Lane Goetz](mailto:skye.lane.goetz@gmail.com)
