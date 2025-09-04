import networkx as nx
import pytest

from mbnpy import network

@pytest.fixture
def network1():
    edges = {
    "e1": {"from": "A", "to": "B", "directed": False},
    "e2": {"from": "B", "to": "C", "directed": False},
    "e3": {"from": "C", "to": "D", "directed": False},
    "e4": {"from": "D", "to": "A", "directed": False},
    }
    nodes = {"A": {"x": 0, "y": 0},
             "B": {"x": 1, "y": 0},
             "C": {"x": 2, "y": 0},
             "D": {"x": 1, "y": 1}}
    
    G = network.get_G(edges, nodes)

    return G

@pytest.fixture
def network2():
    edges = {
        "e1": {"from": "A", "to": "B", "directed": True},
        "e2": {"from": "B", "to": "C", "directed": True},
        "e3": {"from": "C", "to": "A", "directed": True},
    }
    nodes = {"A": {"x": 0, "y": 0},
             "B": {"x": 1, "y": 0},
             "C": {"x": 2, "y": 0}}
    
    G = network.get_G(edges, nodes)

    return G


def test_global_k_certificate1(network1):
    G = network1
    k, s, t, cutset = network.global_k_certificate(G, kind="node")
    assert k == 2

def test_global_k_certificate2(network1):
    G = network1
    k, s, t, cutset = network.global_k_certificate(G, kind="edge")
    assert k == 2

def test_global_k_certificate3(network2):
    G = network2
    k, s, t, cutset = network.global_k_certificate(G, kind="node")
    assert k == 1

def test_global_k_certificate4(network2):
    G = network2
    k, s, t, cutset = network.global_k_certificate(G, kind="edge")
    assert k == 1

def test_net_per_fun_bin1(network1):
    G = network1
    comps_st = {"e1": 1, "e2": 1, "e3": 1, "e4": 1}
    mode = "global_k_conn"
    inputs = {"kind": "edge", "target_k": 2}
    sys_val, sys_st, min_comps_st = network.net_per_fun_bin(comps_st, G, mode, inputs)
    assert sys_val == 2
    assert sys_st == 's'
    assert min_comps_st is None

def test_net_per_fun_bin2(network1):
    G = network1
    comps_st = {"e1": 0, "e2": 1, "e3": 1, "e4": 1}
    mode = "global_k_conn"
    inputs = {"kind": "edge", "target_k": 2}
    sys_val, sys_st, min_comps_st = network.net_per_fun_bin(comps_st, G, mode, inputs)
    assert sys_val == 1
    assert sys_st == 'f'
    assert min_comps_st is None

def test_net_per_fun_bin3(network2):
    G = network2
    comps_st = {"e1": 1, "e2": 0, "e3": 1}
    mode = "global_k_conn"
    inputs = {"kind": "edge", "target_k": 1}
    sys_val, sys_st, min_comps_st = network.net_per_fun_bin(comps_st, G, mode, inputs)
    assert sys_val == 0
    assert sys_st == 'f'
    assert min_comps_st is None
