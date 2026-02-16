from littleballoffur import PageRankBasedSampler
import plotly.graph_objects as go
from backports import zstd
from pathlib import Path
from typing import Any
import networkx as nx
import polars as pl
import pandas as pd

def read_tsv(p: Path) -> pl.DataFrame:
  with zstd.open(p, "rb") as f:
    return pl.read_csv(f.read(), separator="\t", has_header=True, infer_schema=False)

def is_sig(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(pl.col("significant") == "YES")

def get_undirected(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(pl.col("predicate") != "biolink:affects")

def mkgraph(
  edfu: pl.DataFrame,
  ndf: pl.DataFrame
) -> nx.Graph:
  G: nx.Graph = nx.MultiGraph()

  npd: pd.DataFrame = ndf.to_pandas()
  for _, row in npd.iterrows():
    G.add_node(
      row["id"],
      name=row["name"],
      category=row["category"]
    )

  edpd: pd.DataFrame = edfu.to_pandas()
  for _, row in edpd.iterrows():
    G.add_edge(
      row["subject"],
      row["object"],
      predicate=row["predicate"],
      publication=row["publication"],
      p_value=row["p_value"],
      multiple_testing_correction_method=row["multiple_testing_correction_method"],
      relationship_strength=row["relationship_strength"],
      assertion_method=row["assertion_method"]
    )

  return G

def sampler(G: nx.Graph) -> nx.Graph:
  nnodes: int = round(0.25 * G.number_of_nodes())
  S: object = PageRankBasedSampler(number_of_nodes=nnodes)
  return S.sample(G)

def mkvisual(G: nx.Graph, out: Path) -> None:
  pos: object = nx.spring_layout(G, k=0.5, iterations=20)

  traces: list[object] = []
  for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    attrs: dict[str, Any] = edge[2]
    hover: str = f"<b>{edge[0]} → {edge[1]}</b><br>"

    hover += f"Predicate: {attrs['predicate']}<br>"
    hover += f"Publication: {attrs['publication']}<br>"
    hover += f"P Value: {attrs['p_value']}<br>"
    hover += f"FDR: {attrs['multiple_testing_correction_method']}<br>"
    hover += f"Relationship Strength: {attrs['relationship_strength']}<br>"
    hover += f"Relationship Type: {attrs['assertion_method']}"

    escatter: object = go.Scattergl(
      x=[x0, x1, None],
      y=[y0, y1, None],
      mode="lines",
      line={"color": "#888", "width": 0.5},
      hovertext=hover,
      hoverinfo="text",
      showlegend=False
    ) 
    traces += [escatter]

  ndx: list[float] = [pos[node][0] for node in G.nodes()]
  ndy: list[float] = [pos[node][1] for node in G.nodes()]
  ntext: list[str] = [G.nodes[node]["name"] for node in G.nodes()]
  ncat: list[str] = [G.nodes[node]["category"] for node in G.nodes()]

  nscatter: object = go.Scattergl(
    x=ndx,
    y=ndy,
    text=ntext,
    customdata=ncat,
    hovertemplate="<b>%{text}</b><br>Category: %{customdata}<extra></extra>",
    marker={"size": 6, "color": "lightblue", "line": {"color": "white", "width": 0.5}},
    name="nodes"
  )
  traces += [nscatter]

  fig: object = go.Figure(traces)
  fig.update_layout(
    title="MicrobiomeKG 2.1.0",
    showlegend=False,
    hovermode="closest",
    margin={"b": 0, "l": 0, "r": 0, "t": 40},
    xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
    yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
    plot_bgcolor="white"
  )
  fig.write_html(out)

def main(
  edges: Path = Path("./EDGES.tsv.zst"),
  nodes: Path = Path("./NODES.tsv.zst"),
  out: Path = Path("./GRAPH.html")
) -> None:
  edf: pl.DataFrame = read_tsv(edges)
  edf = is_sig(edf)

  ndf: pl.DataFrame = read_tsv(nodes)

  edfu: pl.DataFrame = get_undirected(edf)

  usubject: pl.Series = edfu.select(pl.col("subject")).to_series()
  uobject: pl.Series = edfu.select(pl.col("object")).to_series()
  unodes: pl.Series = pl.concat((usubject, uobject)).unique()
  ndf = ndf.filter((pl.col("id").is_in(unodes)))

  G: nx.Graph = mkgraph(edfu, ndf)
  G = sampler(G)
  mkvisual(G, out)

if __name__ == "__main__":
  main()
