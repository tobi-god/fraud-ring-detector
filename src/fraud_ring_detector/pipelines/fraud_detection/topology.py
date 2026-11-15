"""Syndicate topology analysis on the subgraph of flagged transactions.

Addresses three limits of plain Louvain on an undirected graph:
  1. direction is ignored            -> directed, amount-aware graph + directed Louvain
  2. communities cannot overlap      -> edge-based communities (Louvain on the line graph); an account that touches
                                        several edge communities belongs to all of them
  3. no roles / ring shape           -> flow-based roles (originator / mule / collector) + pattern mining (cycles, fans)
Pure functions: no file I/O, easy to unit test.
"""
import logging
from itertools import islice

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities

logger = logging.getLogger(__name__)


def build_flow_graph(flagged: pd.DataFrame) -> nx.DiGraph:
    """Directed graph; parallel transactions are merged (amounts summed, max score kept, count tracked)."""
    G = nx.DiGraph()
    for s, d, a, sc in zip(flagged.src, flagged.dst, flagged.amount, flagged.score):
        s, d = int(s), int(d)
        if G.has_edge(s, d):
            e = G[s][d]
            e["amount"] += float(a); e["score"] = max(e["score"], float(sc)); e["n_tx"] += 1
        else:
            G.add_edge(s, d, amount=float(a), score=float(sc), n_tx=1)
    return G


def directed_communities(G: nx.DiGraph, seed: int, resolution: float = 1.0) -> dict[int, int]:
    """Non-overlapping partition that respects edge direction (directed modularity), weighted by amount."""
    if G.number_of_edges() == 0:
        return {}
    comms = louvain_communities(G, weight="amount", resolution=resolution, seed=seed)
    return {n: i for i, c in enumerate(comms) for n in c}


def overlapping_memberships(G: nx.DiGraph, seed: int, max_line_edges: int = 2_000_000) -> dict[int, set[int]]:
    """Overlapping communities via edge partitioning: cluster EDGES (line graph), then every account joins the
    communities of all its incident edges. Guarded because the line graph grows with sum(degree^2)."""
    U = G.to_undirected(as_view=False)
    size = sum(d * (d - 1) // 2 for _, d in U.degree())
    if U.number_of_edges() == 0 or size > max_line_edges:
        if size > max_line_edges:
            logger.warning("line graph too large (%d edges) -> skipping overlap analysis", size)
        return {}
    L = nx.line_graph(U)
    comms = louvain_communities(L, seed=seed)
    mem: dict[int, set[int]] = {}
    for cid, edges in enumerate(comms):
        for u, v in edges:
            mem.setdefault(u, set()).add(cid); mem.setdefault(v, set()).add(cid)
    return mem


def find_cycles(G: nx.DiGraph, max_len: int = 6, max_cycles: int = 5000) -> list[list[int]]:
    """Directed cycles up to max_len hops (round-tripping is the classic layering signature)."""
    return list(islice(nx.simple_cycles(G, length_bound=max_len), max_cycles))


def assign_roles(G: nx.DiGraph, fan_threshold: int = 4, near_zero: float = 0.2,
                 pass_through: float = 0.6) -> pd.DataFrame:
    """Roles from money flow inside the flagged subgraph.
    originator : sends, (almost) never receives      collector : receives, (almost) never sends
    mule       : receives and forwards a similar amount (pass-through)     mixed : everything else
    Plus shape tags: fan_out / fan_in hubs, and in_cycle (set by caller)."""
    rows = []
    for n in G.nodes:
        inn = sum(d["amount"] for _, _, d in G.in_edges(n, data=True))
        out = sum(d["amount"] for _, _, d in G.out_edges(n, data=True))
        ind, outd = G.in_degree(n), G.out_degree(n)
        if out > 0 and inn <= near_zero * out:
            role = "originator"
        elif inn > 0 and out <= near_zero * inn:
            role = "collector"
        elif inn > 0 and out > 0 and min(inn, out) / max(inn, out) >= pass_through:
            role = "mule"
        else:
            role = "mixed"
        rows.append({"account_idx": n, "role": role, "in_amount": inn, "out_amount": out,
                     "in_degree": ind, "out_degree": outd,
                     "pass_through_ratio": (min(inn, out) / max(inn, out)) if max(inn, out) > 0 else 0.0,
                     "fan_out_hub": outd >= fan_threshold and ind <= 1,
                     "fan_in_hub": ind >= fan_threshold and outd <= 1,
                     "gather_scatter_hub": ind >= fan_threshold and outd >= fan_threshold})
    return pd.DataFrame(rows)
