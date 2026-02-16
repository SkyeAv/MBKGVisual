from pyvis.network import Network
from typing import Optional
from backports import zstd
from pathlib import Path
import networkx as nx
import polars as pl
import pandas as pd

def catcolor(cat: str) -> str:
  cmap: dict[str, str] = {
    "biolink:OrganismTaxon": "#2ecc71",
    "biolink:PopulationOfIndividualOrganisms": "#27ae60",
    "biolink:Gene": "#3498db",
    "biolink:GeneFamily": "#2980b9",
    "biolink:Protein": "#1abc9c",
    "biolink:Polypeptide": "#16a085",
    "biolink:ChemicalEntity": "#e67e22",
    "biolink:SmallMolecule": "#d35400",
    "biolink:MolecularEntity": "#f39c12",
    "biolink:BiologicalProcess": "#9b59b6",
    "biolink:MolecularActivity": "#8e44ad",
    "biolink:Phenomenon": "#a569bd",
  }
  return cmap.get(cat, "#95a5a6")

def read_tsv(p: Path) -> pl.DataFrame:
  with zstd.open(p, "rb") as f:
    return pl.read_csv(f.read(), separator="\t", has_header=True, infer_schema=False)

def is_sig(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(pl.col("significant") == "YES")

def sanitize(ndf: pl.DataFrame, edf: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
  ndf = ndf.filter(~pl.col("name").is_in({"4", "3", "2", "1"}))
  edf = edf.filter(~pl.col("subject_name").is_in({"4", "3", "2", "1"}))
  edf = edf.filter(~pl.col("object_name").is_in({"4", "3", "2", "1"}))
  edf = edf.filter(~(pl.col("object_name") == pl.col("subject_name")))
  return (ndf, edf)

def is_directed(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
  directed: pl.DataFrame = df.filter(pl.col("predicate") == "biolink:affects")
  undirected: pl.DataFrame = df.filter(pl.col("predicate") != "biolink:affects")
  return (directed, undirected)

def mkgraph(
  edfd: tuple[pl.DataFrame, pl.DataFrame],
  ndf: pl.DataFrame
) -> nx.Graph:
  G: nx.Graph = nx.MultiDiGraph()

  npd: pd.DataFrame = ndf.to_pandas()
  for _, row in npd.iterrows():
    G.add_node(
      row["name"],
      name=row["name"],
      curie=row["id"],
      category=row["category"]
    )

  edpl, eupl = edfd

  edpd: pd.DataFrame = edpl.to_pandas()
  for _, row in edpd.iterrows():
    G.add_edge(
      row["subject_name"],
      row["object_name"],
      predicate=row["predicate"],
      publication=row["publication"],
      p_value=row["p_value"],
      multiple_testing_correction_method=row["multiple_testing_correction_method"],
      relationship_strength=row["relationship_strength"],
      assertion_method=row["assertion_method"]
    )

  eupd: pd.DataFrame = eupl.to_pandas()
  for _, row in eupd.iterrows():
    G.add_edge(
      row["subject_name"],
      row["object_name"],
      predicate=row["predicate"],
      publication=row["publication"],
      p_value=row["p_value"],
      multiple_testing_correction_method=row["multiple_testing_correction_method"],
      relationship_strength=row["relationship_strength"],
      assertion_method=row["assertion_method"]
    )
    # * Symetrical edge for undirected edges
    G.add_edge(
      row["object_name"],
      row["subject_name"],
      predicate=row["predicate"],
      publication=row["publication"],
      p_value=row["p_value"],
      multiple_testing_correction_method=row["multiple_testing_correction_method"],
      relationship_strength=row["relationship_strength"],
      assertion_method=row["assertion_method"]
    )

  return G

def sampler(G: nx.Graph) -> nx.Graph:
  nnodes: int = round(0.001 * G.number_of_nodes())

  seed: object = max(G.degree(), key=(lambda x: x[1]))[0]  # pyright: ignore
  sampled: set[object] = {seed}

  degrees: dict[object, object] = dict(G.degree())  # pyright: ignore

  while len(sampled) < nnodes:

    neighbors: set[object] = set()
    for node in sampled:
      neighbors |= set(G.successors(node))  # pyright: ignore
      neighbors |= set(G.predecessors(node))  # pyright: ignore

    candidates: Optional[object] = neighbors - sampled
    if not candidates:
      break

    next_node: object = max(candidates, key=(lambda x: degrees[x]))  # pyright: ignore
    sampled |= {next_node}
  
  return G.subgraph(sampled).copy()

def mkvis(G: nx.Graph, out: Path) -> None:
  nt: object = Network(
    height="750px",
    width="100%",
    directed=True,
    bgcolor="#222222",
    font_color="white"  # pyright: ignore
  )

  nt.from_nx(G)
  nt.save_graph(out.as_posix())

def main(
  edges: Path = Path("./EDGES.tsv.zst"),
  nodes: Path = Path("./NODES.tsv.zst"),
  out: Path = Path("./GRAPH.html")
) -> None:
  edf: pl.DataFrame = read_tsv(edges)
  edf = is_sig(edf)

  ndf: pl.DataFrame = read_tsv(nodes)
  ndf, edf = sanitize(ndf, edf)

  edfd: tuple[pl.DataFrame, pl.DataFrame] = is_directed(edf)

  G: nx.Graph = mkgraph(edfd, ndf)
  G = sampler(G)

  mkvis(G, out)

if __name__ == "__main__":
  main()
