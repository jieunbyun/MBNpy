import itertools as it
import networkx as nx
from networkx.algorithms.connectivity import (
    node_connectivity, edge_connectivity,
    minimum_node_cut, minimum_edge_cut,
    local_node_connectivity, local_edge_connectivity
)
from typing import Literal, Dict, Tuple, Any, Optional
import copy

"""
The functions are designed to be compatible with datasets at https://github.com/jieunbyun/network-datasets
"""

def get_G(edges, nodes):
    if all(v["directed"] for e, v in edges.items()):
        G = nx.DiGraph()

        for n, v in nodes.items():
            attrs = {k: val for k, val in v.items() if k not in ("x", "y")}
            G.add_node(n, pos=(v["x"], v["y"]), **attrs)

        for e, v in edges.items():
            attrs = {k: val for k, val in v.items() if k not in ("from", "to", "directed")}
            G.add_edge(v["from"], v["to"], name=e, **attrs)

    elif all(not v["directed"] for e, v in edges.items()):
        G = nx.Graph()
        for n, v in nodes.items():
            attrs = {k: val for k, val in v.items() if k not in ("x", "y")}
            G.add_node(n, pos=(v["x"], v["y"]), **attrs)

        for e, v in edges.items():
            attrs = {k: val for k, val in v.items() if k not in ("from", "to", "directed")}
            G.add_edge(v["from"], v["to"], name=e, **attrs)

    elif any(v["directed"] for e, v in edges.items()):
        G = nx.DiGraph()

        for n, v in nodes.items():
            attrs = {k: val for k, val in v.items() if k not in ("x", "y")}
            G.add_node(n, pos=(v["x"], v["y"]), **attrs)

        for e, v in edges.items():
            attrs = {k: val for k, val in v.items() if k not in ("from", "to", "directed")}
            if v["directed"]:
                G.add_edge(v["from"], v["to"], name=e, **attrs)
            else:
                G.add_edge(v["to"], v["from"], name=e, **attrs)
                G.add_edge(v["from"], v["to"], name=e, **attrs)
    return G

def _is_connected(G):
    """Strongly connected for DiGraph, connected for Graph."""
    return nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)

def global_k_connectivity(G, kind="node"):
    """
    Global k-connectivity of G.
    kind='node' -> vertex connectivity κ(G)
    kind='edge' -> edge connectivity λ(G)
    """
    if kind not in {"node", "edge"}:
        raise ValueError("kind must be 'node' or 'edge'")

    if not _is_connected(G):
        return 0

    return node_connectivity(G) if kind == "node" else edge_connectivity(G)

def global_k_certificate(G, kind="node"):
    """
    Returns (k, s, t, cutset) where:
      k      = global k-connectivity (node or edge)
      (s, t) = a pair of nodes witnessing the minimum connectivity
      cutset = a minimum node/edge cut separating s and t
    """
    if not _is_connected(G):
        # find a disconnected witness quickly
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return 0, None, None, set()
        # pick any pair in different components
        if G.is_directed():
            # weak components may still be >1 even if not strongly connected
            # but k=0 for strong connectivity; just return any ordered pair
            s, t = nodes[0], nodes[1]
        else:
            comps = list(nx.connected_components(G))
            s, t = next(iter(comps[0])), next(iter(comps[-1]))
        return 0, s, t, set()

    # get the global value first
    k = global_k_connectivity(G, kind=kind)

    # search for a pair with local connectivity == k
    # use permutations for directed, combinations for undirected
    node_iter = (it.permutations(G.nodes, 2) if G.is_directed()
                 else it.combinations(G.nodes, 2))

    if kind == "node":
        local_fn = local_node_connectivity
        cut_fn   = minimum_node_cut
    else:
        local_fn = local_edge_connectivity
        cut_fn   = minimum_edge_cut

    for s, t in node_iter:
        if local_fn(G, s, t) == k:
            cutset = cut_fn(G, s, t)
            return k, s, t, set(cutset)

    # Fallback (shouldn’t happen): return k without certificate
    return k, None, None, set()

def net_per_fun_bin(
        comps_st: Dict[Any, int],
        G,
        mode: Literal["global_k_conn"],
        inputs: Dict
    ) -> Tuple[Any, str, Optional[Dict[str, int]]]:
    """
    Computes the system performance function based on the states of components.
    NB This can compute only binary component states.

    Args:
        comps_st (dict): Dictionary {comp_name: state (int)}
        mode (str): ('global k conn').
        inputs (dict): Additional inputs required for certain modes.

    Returns:
        sys_val: Any type representing the system performance value.
        sys_st: Either 's' or 'f' indicating system state.
        min_comps_st (dict or None): Dictionary of components with minimum states to satisfy sys_st.
    """

    if not all(v in (0, 1) for v in comps_st.values()):
        raise ValueError("All component states must be 0 or 1.")
    
    # Remove failed components (either node or edge) from the graph
    G_ = copy.deepcopy(G)
    edge_names = [data["name"] for _, _, data in G.edges(data=True)]
    for x, st in comps_st.items():
        if st == 0:
            if x in edge_names:
                G_.remove_edge(*[(u, v) for u, v, d in G_.edges(data=True) if d["name"] == x][0])
            
            else:
                G_.remove_node(x)

    # Depending on mode, compute the system performance
    if mode == "global_k_conn":
        kind = inputs.get("kind", "node")
        target_k = inputs.get("target_k", 1)
        k, s, t, cutset = global_k_certificate(G_, kind=kind)
        sys_val = k
        if k < target_k:
            sys_st = 'f'
        else:
            sys_st = 's'
        min_comps_st = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return sys_val, sys_st, min_comps_st