"""Unit tests for direction-aware communities, overlap, cycles and roles on tiny hand-built graphs."""
import pandas as pd

from fraud_ring_detector.pipelines.fraud_detection.topology import (assign_roles, build_flow_graph, directed_communities,
                                                                   find_cycles, overlapping_memberships)


def frame(edges):
    return pd.DataFrame(edges, columns=["src", "dst", "amount", "score"])


def test_cycle_and_roles_on_layering_chain():
    # 0 -> 1 -> 2 -> 3 (originator -> mule -> mule -> collector) plus a separate 4-cycle 10->11->12->13->10
    chain = [(0, 1, 1000, .9), (1, 2, 950, .9), (2, 3, 900, .9)]
    ring = [(10, 11, 500, .8), (11, 12, 500, .8), (12, 13, 500, .8), (13, 10, 500, .8)]
    G = build_flow_graph(frame(chain + ring))
    cycles = find_cycles(G, max_len=6)
    assert len(cycles) == 1 and set(cycles[0]) == {10, 11, 12, 13}
    roles = assign_roles(G).set_index("account_idx").role
    assert roles[0] == "originator" and roles[3] == "collector"
    assert roles[1] == "mule" and roles[2] == "mule"


def test_direction_matters_for_flow_graph():
    G = build_flow_graph(frame([(0, 1, 100, .9)]))
    assert G.has_edge(0, 1) and not G.has_edge(1, 0)


def test_parallel_transactions_are_merged_not_lost():
    G = build_flow_graph(frame([(0, 1, 100, .5), (0, 1, 200, .9)]))
    assert G[0][1]["amount"] == 300 and G[0][1]["n_tx"] == 2 and G[0][1]["score"] == .9


def test_bridge_account_can_belong_to_two_overlapping_communities():
    # two dense triangles joined by one bridge account (3) that trades with both
    e = [(0, 1, 100, .9), (1, 2, 100, .9), (2, 0, 100, .9), (3, 0, 100, .9), (3, 1, 100, .9),
         (3, 4, 100, .9), (3, 5, 100, .9), (4, 5, 100, .9), (5, 6, 100, .9), (6, 4, 100, .9), (4, 6, 100, .9)]
    G = build_flow_graph(frame(e))
    mem = overlapping_memberships(G, seed=0)
    assert mem and all(len(v) >= 1 for v in mem.values())
    assert directed_communities(G, seed=0)            # non-overlapping partition still available
