import plotly.graph_objects as go
from typing import Optional
from backports import zstd
from pathlib import Path
from typing import Any
import networkx as nx
import polars as pl
import pandas as pd
import math

def read_tsv(p: Path) -> pl.DataFrame:
  with zstd.open(p, "rb") as f:
    return pl.read_csv(f.read(), separator="\t", has_header=True, infer_schema=False)

def is_sig(df: pl.DataFrame) -> pl.DataFrame:
  return df.filter(pl.col("significant") == "YES")

def sanitize(ndf: pl.DataFrame, edf: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
  ndf = ndf.filter(~pl.col("name").is_in({"4", "3", "2", "1"}))
  edf = edf.filter(~pl.col("subject_name").is_in({"4", "3", "2", "1"}))
  edf = edf.filter(~pl.col("object_name").is_in({"4", "3", "2", "1"}))
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
      directed=True,
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
      directed=False,
      publication=row["publication"],
      p_value=row["p_value"],
      multiple_testing_correction_method=row["multiple_testing_correction_method"],
      relationship_strength=row["relationship_strength"],
      assertion_method=row["assertion_method"]
    )
    # Symetrical edge for undirected edges
    G.add_edge(
      row["object_name"],
      row["subject_name"],
      predicate=row["predicate"],
      directed=False,
      publication=row["publication"],
      p_value=row["p_value"],
      multiple_testing_correction_method=row["multiple_testing_correction_method"],
      relationship_strength=row["relationship_strength"],
      assertion_method=row["assertion_method"]
    )

  return G

def sampler(G: nx.Graph) -> nx.Graph:
  nnodes: int = round(0.005 * G.number_of_nodes())

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

def mkvisual(G: nx.Graph, out: Path) -> None:
  pos: object = nx.kamada_kawai_layout(G)

  COLORS: dict[str, str] = {
    "bg_deep": "#080b12",
    "bg": "#0f1318",
    "bg_elevated": "#171c24",
    "text_primary": "#f0f4f8",
    "text_secondary": "#a8b5c4",
    "text_muted": "#6b7a8f",
    "edge_causal": "#64b5f6",
    "edge_causal_glow": "rgba(100, 181, 246, 0.3)",
    "edge_correlation": "#37474f",
    "accent": "#4dd0e1",
    "border": "#2a3441",
  }

  CATEGORY_PALETTE: list[dict[str, str]] = [
    {"fill": "#e57373", "stroke": "#c62828"},
    {"fill": "#64b5f6", "stroke": "#1565c0"},
    {"fill": "#81c784", "stroke": "#2e7d32"},
    {"fill": "#ba68c8", "stroke": "#7b1fa2"},
    {"fill": "#ffb74d", "stroke": "#ef6c00"},
    {"fill": "#4dd0e1", "stroke": "#00838f"},
    {"fill": "#f06292", "stroke": "#c2185b"},
    {"fill": "#aed581", "stroke": "#558b2f"},
    {"fill": "#90a4ae", "stroke": "#455a64"},
    {"fill": "#fff176", "stroke": "#f9a825"},
  ]

  FONT: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

  degrees: dict[Any, int] = dict(G.degree())  # pyright: ignore
  max_degree: int = max(degrees.values()) if degrees else 1
  min_degree: int = min(degrees.values()) if degrees else 1

  def scale_size(deg: int) -> float:
    normalized = (deg - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
    return 10 + 22 * math.sqrt(normalized)

  traces: list[object] = []
  directed_edges: list[tuple[float, float, float, float, str, str]] = []
  undirected_edges: list[tuple[float, float, float, float, str]] = []

  for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    attrs: dict[str, Any] = edge[2]
    predicate: str = attrs["predicate"].replace("biolink:", "").replace("_", " ").title()

    hover: str = (
      f"<span style='font-size:11px;color:{COLORS['text_muted']};text-transform:uppercase;"
      f"letter-spacing:0.5px'>Relationship</span><br>"
      f"<b style='font-size:15px;color:{COLORS['text_primary']}'>{edge[0]}</b><br>"
      f"<span style='color:{COLORS['accent']};font-size:12px'>⟶ {predicate}</span><br>"
      f"<b style='font-size:15px;color:{COLORS['text_primary']}'>{edge[1]}</b><br><br>"
      f"<span style='color:{COLORS['text_muted']}'>p-value</span> "
      f"<span style='color:{COLORS['text_secondary']}'>{attrs['p_value']}</span><br>"
      f"<span style='color:{COLORS['text_muted']}'>Strength</span> "
      f"<span style='color:{COLORS['text_secondary']}'>{attrs['relationship_strength']}</span><br>"
      f"<span style='color:{COLORS['text_muted']}'>Method</span> "
      f"<span style='color:{COLORS['text_secondary']}'>{attrs['assertion_method']}</span>"
    )

    if attrs["directed"]:
      directed_edges.append((x0, y0, x1, y1, hover, predicate))
    else:
      undirected_edges.append((x0, y0, x1, y1, hover))

  # Undirected edges
  if undirected_edges:
    ux: list[Optional[float]] = []
    uy: list[Optional[float]] = []
    for x0, y0, x1, y1, _ in undirected_edges:
      ux.extend([x0, x1, None])
      uy.extend([y0, y1, None])

    traces.append(go.Scattergl(
      x=ux, y=uy, mode="lines",
      line={"color": COLORS["edge_correlation"], "width": 0.6},
      hoverinfo="skip", showlegend=True,
      name="○  Correlated With", legendgroup="edges",
      legendgrouptitle={"text": "RELATIONSHIPS", "font": {"size": 10, "color": COLORS["text_muted"]}},
      opacity=0.5
    ))

  # Directed edges with glow
  if directed_edges:
    dx: list[Optional[float]] = []
    dy: list[Optional[float]] = []
    for x0, y0, x1, y1, _, _ in directed_edges:
      dx.extend([x0, x1, None])
      dy.extend([y0, y1, None])

    traces.append(go.Scattergl(
      x=dx, y=dy, mode="lines",
      line={"color": COLORS["edge_causal_glow"], "width": 4},
      hoverinfo="skip", showlegend=False, opacity=0.4
    ))
    traces.append(go.Scattergl(
      x=dx, y=dy, mode="lines",
      line={"color": COLORS["edge_causal"], "width": 1.0},
      hoverinfo="skip", showlegend=True,
      name="●  Affects", legendgroup="edges", opacity=0.85
    ))

  # Edge hover points
  all_edges_hover: list[tuple[float, float, str]] = []
  for x0, y0, x1, y1, hover, *_ in directed_edges:
    all_edges_hover.append(((x0 + x1) / 2, (y0 + y1) / 2, hover))
  for x0, y0, x1, y1, hover in undirected_edges:
    all_edges_hover.append(((x0 + x1) / 2, (y0 + y1) / 2, hover))

  if all_edges_hover:
    traces.append(go.Scattergl(
      x=[e[0] for e in all_edges_hover],
      y=[e[1] for e in all_edges_hover],
      mode="markers", marker={"size": 12, "opacity": 0},
      hovertext=[e[2] for e in all_edges_hover], hoverinfo="text",
      hoverlabel={
        "bgcolor": COLORS["bg_elevated"], "bordercolor": COLORS["border"],
        "font": {"family": FONT, "size": 12, "color": COLORS["text_primary"]}, "align": "left"
      },
      showlegend=False
    ))

  # Nodes by category
  categories: list[str] = [G.nodes[node]["category"] for node in G.nodes()]
  ucategories: list[str] = list(sorted(set(categories)))
  label_annotations: list[dict[str, Any]] = []
  degree_threshold: float = max_degree * 0.6

  for idx, category in enumerate(ucategories):
    cat_nodes = [n for n in G.nodes() if G.nodes[n]["category"] == category]
    palette = CATEGORY_PALETTE[idx % len(CATEGORY_PALETTE)]

    ndx: list[float] = [pos[node][0] for node in cat_nodes]
    ndy: list[float] = [pos[node][1] for node in cat_nodes]
    sizes: list[float] = [scale_size(degrees[node]) for node in cat_nodes]

    ntext: list[str] = []
    for node in cat_nodes:
      name: str = G.nodes[node]["name"]
      curie: str = G.nodes[node]["curie"]
      cat_display: str = G.nodes[node]["category"].replace("biolink:", "").replace("_", " ").title()
      deg: int = degrees[node]

      hover: str = (
        f"<span style='font-size:10px;color:{COLORS['text_muted']};text-transform:uppercase;"
        f"letter-spacing:0.5px'>{cat_display}</span><br>"
        f"<b style='font-size:16px;color:{COLORS['text_primary']}'>{name}</b><br><br>"
        f"<span style='color:{COLORS['text_muted']}'>CURIE</span> "
        f"<span style='color:{COLORS['text_secondary']};font-family:monospace;font-size:11px'>{curie}</span><br>"
        f"<span style='color:{COLORS['text_muted']}'>Connections</span> "
        f"<span style='color:{COLORS['accent']};font-weight:600'>{deg}</span>"
      )
      ntext.append(hover)

      if deg >= degree_threshold:
        label_annotations.append({
          "x": pos[node][0], "y": pos[node][1],
          "text": name[:20] + ("..." if len(name) > 20 else ""),
          "showarrow": False, "opacity": 0.85,
          "font": {"size": 9, "color": COLORS["text_secondary"], "family": FONT},
          "xshift": 0, "yshift": scale_size(deg) / 2 + 8
        })

    legend_name: str = category.replace("biolink:", "").replace("_", " ").title()

    # Glow layer
    traces.append(go.Scattergl(
      x=ndx, y=ndy, mode="markers",
      marker={"size": [s + 8 for s in sizes], "color": palette["fill"], "opacity": 0.15},
      hoverinfo="skip", showlegend=False
    ))

    # Main nodes
    traces.append(go.Scattergl(
      x=ndx, y=ndy, mode="markers",
      hovertext=ntext, hoverinfo="text",
      hoverlabel={
        "bgcolor": COLORS["bg_elevated"], "bordercolor": COLORS["border"],
        "font": {"family": FONT, "size": 12, "color": COLORS["text_primary"]}, "align": "left"
      },
      marker={
        "size": sizes, "color": palette["fill"],
        "line": {"color": palette["stroke"], "width": 1.5}, "opacity": 0.92
      },
      name=f"●  {legend_name}", legendgroup="nodes",
      legendgrouptitle={"text": "NODE TYPES", "font": {"size": 10, "color": COLORS["text_muted"]}}
    ))

  fig: object = go.Figure(traces)

  static_annotations: list[dict[str, Any]] = [
    {
      "text": (
        f"<a href='https://pubmed.ncbi.nlm.nih.gov/39464038/' style='color:{COLORS['accent']};"
        f"text-decoration:none'>Goetz et al. 2024</a>"
        f"<span style='color:{COLORS['text_muted']}'> · Institute for Systems Biology</span>"
      ),
      "showarrow": False, "xref": "paper", "yref": "paper",
      "x": 1.0, "y": -0.01, "xanchor": "right",
      "font": {"size": 10, "color": COLORS["text_muted"], "family": FONT}
    },
    {
      "text": f"<span style='color:{COLORS['text_muted']}'>{G.number_of_nodes()} nodes · {G.number_of_edges()} edges</span>",
      "showarrow": False, "xref": "paper", "yref": "paper",
      "x": 0.0, "y": -0.01, "xanchor": "left",
      "font": {"size": 10, "color": COLORS["text_muted"], "family": FONT}
    }
  ]

  fig.update_layout(
    title={
      "text": (
        f"<b style='font-size:24px;color:{COLORS['text_primary']}'>MicrobiomeKG</b>"
        f"<span style='font-size:14px;color:{COLORS['text_muted']};font-weight:400'> 2.1.0</span>"
        f"<br><span style='font-size:13px;color:{COLORS['text_secondary']}'>Snowball Sample Visualization</span>"
      ),
      "font": {"family": FONT}, "x": 0.5, "y": 0.97, "xanchor": "center", "yanchor": "top"
    },
    showlegend=True,
    legend={
      "orientation": "v", "yanchor": "top", "y": 0.98, "xanchor": "left", "x": 0.01,
      "bgcolor": "rgba(15, 19, 24, 0.95)", "bordercolor": COLORS["border"], "borderwidth": 1,
      "font": {"family": FONT, "size": 11, "color": COLORS["text_secondary"]},
      "itemsizing": "constant", "itemwidth": 30, "tracegroupgap": 12,
      "groupclick": "toggleitem", "indentation": 8
    },
    hovermode="closest", hoverdistance=20,
    margin={"b": 50, "l": 24, "r": 24, "t": 80},
    xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "fixedrange": False, "scaleanchor": "y", "scaleratio": 1},
    yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "fixedrange": False},
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg_deep"],
    annotations=static_annotations + label_annotations,
    dragmode="pan",
    modebar={"bgcolor": "rgba(0,0,0,0)", "color": COLORS["text_muted"], "activecolor": COLORS["accent"], "orientation": "v"}
  )

  config: dict[str, Any] = {
    "scrollZoom": True, "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "modeBarButtonsToAdd": ["toggleSpikelines"],
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "filename": "microbiomekg_visualization", "scale": 2},
    "doubleClick": "reset"
  }

  fig.write_html(out, config=config, include_plotlyjs="cdn", full_html=True, include_mathjax=False)

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
  mkvisual(G, out)

if __name__ == "__main__":
  main()
