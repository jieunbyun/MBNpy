from pathlib import Path
import pandas as pd
import time
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import geopandas as gpd
import pickle
import numpy as np
import typer
from collections import namedtuple
import shapely
import polyline
from scipy import stats
from itertools import islice
from typing_extensions import Annotated

from mbnpy import model, config, trans, variable, brc, branch, cpm, inference

app = typer.Typer()

HOME = Path(__file__).parent

DAMAGE_STATES = ['Slight', 'Moderate', 'Extensive', 'Complete']

KEYS = ['LATITUDE', 'LONGITUDE', 'HAZUS_CLASS', 'K3d', 'Kskew', 'Ishape']

GDA94 = 'EPSG:4283'  # GDA94
GDA94_ = 'EPSG:3577'

Route = namedtuple('route', ['path', 'edges', 'weight'])


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def read_file_dmg(file_dmg: str) -> pd.DataFrame:

    # assign cpms given scenario
    if file_dmg.endswith('csv'):
        probs = pd.read_csv(file_dmg, index_col=0)
        probs.index = probs.index.astype('str')

    elif file_dmg.endswith('json'):
        # shakecast output
        with open(file_dmg, 'rb') as f:
            tmp = json.load(f)

        probs = {}
        for item in tmp['features']:
            probs[item['properties']['name']] = {
                    'Slight': item['properties']['shaking']['green'],
                    'Moderate': item['properties']['shaking']['yellow'],
                    'Extensive': item['properties']['shaking']['orange'],
                    'Complete': item['properties']['shaking']['red'],
                    }

        # from percent 
        probs = pd.DataFrame.from_dict(probs).T
        if probs.max().max() > 1:
            print(f'Probability should be 0 to 1 scale: {probs.max().max():.2f}')
            probs *= 0.01

    # failiure defined 
    probs['failure'] = probs['Extensive'] + probs['Complete']
    probs = probs.to_dict('index')

    return probs


def update_nodes_given_dmg(nodes: dict, file_dmg: str, valid_paths=None) -> None:

    # selected nodes
    if valid_paths:
        sel_nodes = [x for _, v in valid_paths.items() for x in v['path']]
        to_be_removed = set(nodes.keys()).difference(sel_nodes)
        [nodes.pop(k) for k in to_be_removed]

    # assign cpms given scenario
    probs = read_file_dmg(file_dmg)

    for k, v in nodes.items():
        try:
            v['failure'] = probs[k]['failure']

        except KeyError:
            v['failure'] = 0.0


def create_shpfile_nodes(nodes: dict, outfile: str) -> None:
    """


    """
    keys = ['pos_x', 'pos_y', 'failure']
    _dic = {}
    for k, v in nodes.items():
        _dic[k] = {d: v[d] for d in keys}

    df = pd.DataFrame.from_dict(_dic).T
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.pos_x, y=df.pos_y), crs=GDA94)
    df.to_file(outfile)
    print(f'{outfile} is written')


def create_shpfile(json_file: str, prob_by_path: dict, outfile: str) -> None:
    """


    """
    with open(json_file, 'r') as f:
        directions_result = json.load(f)

    _dic = {}
    for k, (item, key) in enumerate(zip(directions_result, prob_by_path)):
        poly_line = item['overview_polyline']['points']
        geometry_points = [(x[1], x[0]) for x in polyline.decode(poly_line)]

        distance_km = sum([x['distance']['value']  for x in item['legs']]) / 1000.0
        duration_mins = sum([x['duration']['value']  for x in item['legs']]) / 60.0

        _dic[k] = {'dist_km': distance_km,
                   'duration_m': duration_mins,
                   'line': shapely.LineString(geometry_points),
                   'id': k + 1,
                   'prob': prob_by_path.get(key, 0),
                   }

    df = pd.DataFrame.from_dict(_dic).T
    df = gpd.GeoDataFrame(df, geometry='line', crs= GDA94)
    df.to_file(outfile)
    print(f'{outfile} is written')


def get_kshape(Ishape, sa03, sa10):
    """
    sa03, sa10 can be vector
    """
    if isinstance(sa03, float):
        if Ishape:
            Kshape = np.min([1.0, 2.5*sa10/sa03])
        else:
            Kshape  = 1.0
    else:
        if Ishape:
            Kshape = np.minimum(1.0, 2.5*sa10/sa03)
        else:
            Kshape  = np.ones_like(sa03)
    return Kshape


def compute_pe_by_ds(ps, kshape, Kskew, k3d, sa10):

    factor = {'Slight': kshape,
              'Moderate': Kskew*k3d,
              'Extensive': Kskew*k3d,
              'Complete': Kskew*k3d}

    return pd.Series({ds: stats.lognorm.cdf(sa10, 0.6, scale=factor[ds]*ps[ds])
            for ds in DAMAGE_STATES})


@app.command()
def dmg_bridge(file_bridge: str, file_gm: str):

    # HAZUS Methodology
    bridge_param = pd.read_csv(
        HOME.joinpath('bridge_classification_damage_params.csv'), index_col=0)

    # read sitedb data 
    df_bridge = pd.read_csv(file_bridge, index_col=0)[KEYS].copy()

    # read gm
    gm = pd.read_csv(file_gm, index_col=0, skiprows=1)

    # weighted
    dmg = []
    for i, row in df_bridge.iterrows():
        _df = (gm['lat']-row['LATITUDE'])**2 + (gm['lon']-row['LONGITUDE'])**2
        idx = _df.idxmin()

        if _df.loc[idx] < 0.01:
            row['SA03'] = gm.loc[idx, 'gmv_SA(0.3)']
            row['SA10'] = gm.loc[idx, 'gmv_SA(1.0)']
            row['Kshape'] = get_kshape(row['Ishape'], row['SA03'], row['SA10'])

            df_pe = compute_pe_by_ds(
                bridge_param.loc[row['HAZUS_CLASS']], row['Kshape'], row['Kskew'], row['K3d'], row['SA10'])

            #df_pb = get_pb_from_pe(df_pe)
            #df_pe['Kshape'] = row['Kshape']
            #df_pe['SA10'] = row['SA10']

            dmg.append(df_pe)

        else:
            print('Something wrong {}:{}'.format(i, _df.loc[idx]))

    dmg = pd.DataFrame(dmg)
    dmg.index = df_bridge.index
    #df_bridge = pd.concat([df_bridge, tmp], axis=1)

    # convert to edge prob
    #dmg = convert_dmg(tmp)

    dir_path = Path(file_gm).parent
    file_output = dir_path.joinpath(Path(file_gm).stem + '_dmg.csv')
    dmg.to_csv(file_output)
    print(f'{file_output} saved')

    return dmg


def convert_dmg(dmg):

    cfg = config.Config(HOME.joinpath('./config.json'))

    tmp = {}
    for k, v in cfg.infra['edges'].items():

        try:
            p0 = 1 - dmg.loc[v['origin']]['Extensive']
        except KeyError:
            p0 = 1.0

        try:
            p1 = 1 - dmg.loc[v['destination']]['Extensive']
        except KeyError:
            p0 = 1.0
        finally:
            tmp[k] = {'F': 1 - p0 * p1}

    tmp = pd.DataFrame(tmp).T
    tmp['S'] = 1 - tmp['F']

    return tmp


@app.command()
def plot_alt():

    G = nx.Graph()

    cfg = config.Config(HOME.joinpath('./config.json'))
    #cfg.infra['G'].edges()

    # read routes
    for route_file in Path('../bridge/').glob('route*.txt'):
        #print(_file)

        route = [x.strip() for x in pd.read_csv(route_file, header=None, dtype=str, )[0].to_list()]

        for item in zip(route[:-1], route[1:]):
            try:
                label = cfg.infra['G'].edges[item]['label']
            except KeyError:
                try:
                    label = cfg.infra['G'].edges[item[::-1]]['label']
                except KeyError:
                    print(f'Need to add {item}')
                else:
                    G.add_edge(item[0], item[1], label=label, key=label)
            else:
                G.add_edge(item[0], item[1], label=label, key=label)

            for i in item:
                G.add_node(i, pos=(0, 0), label=i, key=i)

    config.plot_graphviz(G, outfile=HOME.joinpath('wheat_graph_a'))
    # check against model
    print(nx.is_isomorphic(G, cfg.infra['G']))
    a = set([(x[1], x[0]) for x in G.edges]).union(G.edges)
    b = set([(x[1], x[0]) for x in cfg.infra['G'].edges]).union(cfg.infra['G'].edges)
    print(set(a).difference(b))
    print(set(b).difference(a))
    print(len(G.edges), len(cfg.infra['G'].edges))
    print(len(G.nodes), len(cfg.infra['G'].nodes))

    """
    origin = route[0]
    if origin in bridges.index:
        origin = bridges.loc[origin][['lat', 'lng']].values.tolist()

    dest = route[-1]
    if dest in bridges.index:
        dest = bridges.loc[dest][['lat', 'lng']].values.tolist()

    route_btw = [{'lat': bridges.loc[x, 'lat'], 'lng': bridges.loc[x, 'lng'], 'id': x} for x in route[1:-1]]


    # create edges

    # plot
    """


@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))
    trans.plot_graph(cfg.infra['G'], HOME.joinpath('wheat.png'))
    config.plot_graphviz(cfg.infra['G'], outfile=HOME.joinpath('wheat_graph'))


def get_vari_cpm_given_path(route, edge_names, weight):

    name = '_'.join((*od_pair, str(idx)))
    path_names.append(name)

    vari = {name: variable.Variable(name=name, values = [np.inf, weight])}

    n_child = len(path)

    # Event matrix of series system
    # Initialize an array of shape (n+1, n+1) filled with 1
    Cpath = np.ones((n_child + 1, n_child + 1), dtype=int)
    for i in range(1, n_child + 1):
        Cpath[i, 0] = 0
        Cpath[i, i] = 0 # Fill the diagonal below the main diagonal with 0
        Cpath[i, i + 1:] = 2 # Fill the lower triangular part (excluding the diagonal) with 2

    #for i in range(1, n_edge + 1):
    ppath = np.array([1.0]*(n_child + 1))

    cpm = {name: cpm.Cpm(variables = [varis[name]] + [varis[n] for n in path], no_child=1, C=Cpath, p=ppath)}

    return vari, cpm


def sys_fun(comps_st, G, threshold, od_pair, d_time_itc):

    G_tmp = G.copy()

    for br, st in comps_st.items():
        if st == 0:
            for neigh in G.neighbors(br):
                G_tmp[br][neigh]['weight'] = float('inf')

    d_time = nx.shortest_path_length(G_tmp, source=od_pair[0], target=od_pair[1], weight='weight')

    if d_time > threshold*d_time_itc:
        sys_st = 'f'
        min_comps_st = None
    else:
        sys_st = 's'

        path = nx.shortest_path(G_tmp, source=od_pair[0], target=od_pair[1], weight='weight')
        min_comps_st = {node: 1 for node in path if node in comps_st.keys()}

    return d_time, sys_st, min_comps_st


@app.command()
def run_brc(file_dmg: str, key: str='Wooroloo-Merredin') -> None:


    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][key]
    od_name = '_'.join(od_pair)

    probs = read_file_dmg(file_dmg)

    # variables
    cpms, varis = {}, {}
    for k in cfg.infra['nodes'].keys():

        varis[k] = variable.Variable(name=k, values = ['f', 's'])
        try:
            pf = probs[k]['failure']
        except KeyError:
            pf = 0.0
        finally:
            cpms[k] = cpm.Cpm(variables=[varis[k]], no_child=1, C=np.array([[0], [1]]), p=[pf, 1-pf])

    G = cfg.infra['G']
    d_time_itc = nx.shortest_path_length(G, source=od_pair[0], target=od_pair[1], weight='weight')

    sf_brc = lambda comps_st: sys_fun(comps_st, cfg.infra['G'], cfg.data['THRESHOLD'], od_pair, d_time_itc)

    probs_brc = {k: {0: cpms[k].p[0, 0], 1: cpms[k].p[1, 0]} for k in cfg.infra['nodes'].keys()}

    fpath_br = HOME.joinpath(f"./brs_{od_name}.parquet")
    fpath_rule = HOME.joinpath(f"./rules_{od_name}.json")

    if Path(fpath_br).exists():
        # load saved results: brs, rules
        brs = branch.load_brs_from_parquet(fpath_br)
        with open(fpath_rule, 'r') as f:
            rules = json.load(f)

    else:
        # run
        brs, rules, sys_res, monitor = brc.run(probs_brc, sf_brc,
            pf_bnd_wr=0.05, max_rules=20, surv_first=True,
            active_decomp=10, display_freq=5)

        # save rules, brs, sys_res, monitor
        branch.save_brs_to_parquet(brs, fpath_br)

        with open(fpath_rule, "w") as f:
            json.dump(rules, f, indent=4)

        fpath_mon = HOME.joinpath(f"./monitor_{od_name}.json")
        with open(fpath_mon, "w") as f:
            json.dump(monitor, f, indent=4)

        fpath_res = HOME.joinpath(f"./sys_res_{od_name}.json")
        sys_res.to_json(fpath_res, orient='records', lines=True )

    # System's CPM 
    varis['sys'] = variable.Variable(name='sys', values=['f', 's', 'u'])
    comp_names = list(cfg.infra['nodes'].keys())
    Csys = branch.get_cmat(brs, {x: varis[x] for x in comp_names})
    psys = np.ones((Csys.shape[0], 1))

    cpms['sys'] = cpm.Cpm(variables=[varis['sys']] + [varis[x] for x in comp_names], no_child=1, C=Csys, p=psys)

    # paths survival probability
    rule_s_probs = brc.eval_rules_prob(rules['s'], 's', probs_brc)
    rule_s_sort_idx = np.argsort(rule_s_probs)[::-1] # sort by descending order of probabilities

    f_surv_rules = HOME.joinpath('./survival_rules_brc.txt')
    if not f_surv_rules.exists():
        with open(f_surv_rules, 'w') as f:
            f.write("rules_prob, rules\n")
            for i in rule_s_sort_idx:
                f.write(f"{rule_s_probs[i]:.3e}, {rules['s'][i]}\n")

    # system survival probability
    start = time.time()
    Msys = inference.prod_Msys_and_Mcomps(cpms['sys'], [cpms[x] for x in comp_names])
    print(f"Elapsed time for system probability calculation: {time.time()-start:.3f} s")

    pf_sys = sum(p for i, p in enumerate(Msys.p) if Msys.C[i,0]==0)
    if isinstance(pf_sys, int):
        pf_sys = 0.0
    else:
        pf_sys = pf_sys[0]

    ps_sys = sum(p for i, p in enumerate(Msys.p) if Msys.C[i,0]==1)
    if isinstance(ps_sys, int):
        ps_sys = 0.0
    else:
        ps_sys = ps_sys[0]

    print(f"System failure probability bounds: {pf_sys:.3e}, {1.0-ps_sys:.3e}")

    # Bounds are obtained for incomplete BnB
    BMs, CIs = {}, {}
    for idx, comp in enumerate(comp_names):

        P_x0 = cpms[comp].p[0]
        P_x1 = cpms[comp].p[1]

        P_s1_x1 = sum(p for i, p in enumerate(Msys.p) if Msys.C[i,0]==1 and Msys.C[i,idx+1]==1)
        P_s0_x1 = sum(p for i, p in enumerate(Msys.p) if Msys.C[i,0]==1 and Msys.C[i,idx+1]==0)
        P_s1_x0 = sum(p for i, p in enumerate(Msys.p) if Msys.C[i,0]==0 and Msys.C[i,idx+1]==1)
        P_s0_x0 = sum(p for i, p in enumerate(Msys.p) if Msys.C[i,0]==0 and Msys.C[i,idx+1]==0)

        # BM
        P_s1_cond_x1_bnd = (P_s1_x1/P_x1, (1.0-P_s0_x1-P_s1_x0-P_s0_x0)/P_x1)
        P_s1_cond_x0_bnd = (P_s1_x0/P_x0, (1.0-P_s0_x1-P_s1_x0-P_s0_x0)/P_x0)

        BM_bounds = (P_s1_cond_x1_bnd[0]-P_s1_cond_x0_bnd[1], P_s1_cond_x1_bnd[1]-P_s1_cond_x0_bnd[0])
        BMs[comp] = (BM_bounds[0][0], BM_bounds[1][0])

        # CI
        CI_bounds = (BM_bounds[0]*P_x0, BM_bounds[1]*P_x0)
        if not np.isnan(CI_bounds).all():
            CIs[comp] = (CI_bounds[0][0], CI_bounds[1][0])

    sorted_CIs_keys = sorted(CIs.keys(), key=lambda x: CIs[x][0], reverse=True)
    sorted_CIs_keys = sorted_CIs_keys[:10]

    BMs_upper = [BMs[x][1] for x in sorted_CIs_keys]
    CIs_upper = [CIs[x][1] for x in sorted_CIs_keys]

    BMs_lower= [BMs[x][0] for x in sorted_CIs_keys]
    CIs_lower = [CIs[x][0] for x in sorted_CIs_keys]

    BMs_error = np.array([
        [BMs_upper[i] - BMs_lower[i] for i in range(len(BMs_upper))],  # Lower errors
        [0 for _ in range(len(BMs_upper))]  # Upper errors (0 because the bar already reaches BMs_upper)
    ])

    CIs_error = np.array([
        [CIs_upper[i] - CIs_lower[i] for i in range(len(CIs_upper))],  # Lower errors
        [0 for _ in range(len(CIs_upper))]  # Upper errors
    ])

    x = np.arange(len(sorted_CIs_keys))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, CIs_error[0], width, bottom=CIs_lower, yerr=CIs_error, capsize=5, color='tomato', alpha=0.7, label="CI")
    bars2 = ax2.bar(x + width/2, BMs_error[0], width, bottom=BMs_lower, yerr=BMs_error, capsize=5, color='royalblue', alpha=0.7, label="BM")

    # Formatting
    ax1.set_xlabel("Keys")
    ax1.set_ylabel("CI", color='tomato')
    ax1.tick_params(axis='y', labelcolor='tomato')

    ax2.set_ylabel("BM", color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')

    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_CIs_keys, rotation=45)
    plt.title("Keys sorted by lower bound of CI")

    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    fn = HOME.joinpath('./CI_bounds.png')
    fig.savefig(fn)
    plt.close(fig)


def get_selected_paths(cfg, od_pair, route_file):

    G = cfg.infra['G']

    d_time_itc = nx.shortest_path_length(G, source=od_pair[0], target=od_pair[1], weight='weight')

    if route_file:
        with open(route_file, 'r') as f:
            selected_paths = json.load(f)

        selected_paths = [item for item in selected_paths.values()]

    else:
        try:
            no_paths = cfg.data['NO_PATHS']
        except KeyError:
            selected_paths = nx.all_simple_paths(G, od_pair[0], od_pair[1])
        else:
            selected_paths = k_shortest_paths(G, source=od_pair[0], target=od_pair[1], k=no_paths, weight='weight')

    valid_paths = []
    for path in selected_paths:
        # Calculate the total weight of the path
        path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
        path_weight = sum(G[u][v]['weight'] for u, v in path_edges)

        # if takes longer than thres * d_time_itc, we consider the od pair is disconnected; moved to config
        if path_weight < cfg.data['THRESHOLD'] * d_time_itc:

            # Collect the edge names if they exist, otherwise just use the edge tuple
            edge_names = [G[u][v].get('key', (u, v)) for u, v in path_edges]
            valid_paths.append((path, edge_names, path_weight))

    # Sort valid paths by weight
    valid_paths = sorted(valid_paths, key=lambda x: x[2])

    keys = ['path', 'edges', 'weight']
    valid_paths = {'_'.join((*od_pair, str(i))): dict(zip(keys, item)) for i, item in enumerate(valid_paths)}

    return valid_paths


@app.command()
def setup_model(file_dmg: str,
                od_name: str='Wooroloo_Merredin',
                route_file: str=None) -> str:
    """
    key:
    route_file: a json file where each OD is defined with a list of nodes
    """
    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][od_name]

    valid_paths = get_selected_paths(cfg, od_pair, route_file)
    path_names = list(valid_paths.keys())

    update_nodes_given_dmg(cfg.infra['nodes'], file_dmg, valid_paths)

    varis = {}
    cpms = {}

    # nodes
    for k, v in cfg.infra['nodes'].items():
        varis[k] = variable.Variable(name=k, values = ['f', 's'])
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                          C = np.array([0, 1]).T, p = [v['failure'], 1 - v['failure']])

    # path
    for name, route in valid_paths.items():

        varis[name] = variable.Variable(name=name, values = [np.inf, route['weight']])

        n_child_p1 = len(route['path']) + 1
        # Event matrix of series system
        # Initialize an array of shape (n+1, n+1) filled with 1
        Cpath = np.ones((n_child_p1, n_child_p1), dtype=int)
        for i in range(1, n_child_p1):
            Cpath[i, 0] = 0
            Cpath[i, i] = 0 # Fill the diagonal below the main diagonal with 0
            Cpath[i, i + 1:] = 2 # Fill the lower triangular part (excluding the diagonal) with 2

        #for i in range(1, n_edge + 1):
        ppath = np.ones(n_child_p1)
        cpms[name] = cpm.Cpm(variables = [varis[name]] + [varis[n] for n in route['path']],
                             no_child=1, C=Cpath, p=ppath)

    # system (od_pair)
    varis[od_name] = variable.Variable(
            name=od_name,
            values=[np.inf] + [varis[p].values[1] for p in path_names[::-1]])

    n_path = len(path_names)
    Csys = np.zeros((n_path + 1, n_path + 1), dtype=int)
    for i in range(n_path):
        Csys[i, 0] = n_path - i
        Csys[i, i + 1] = 1
        Csys[i, i + 2:] = 2
    psys = np.ones(n_path + 1)
    cpms[od_name] = cpm.Cpm(variables = [varis[od_name]] + [varis[p] for p in path_names], no_child=1, C=Csys, p=psys)

    # save
    output_model = cfg.output_path.joinpath(f'model_{od_name}_{len(valid_paths)}_{Path(file_dmg).stem}.pk')
    with open(output_model, 'wb') as f:
        dump = {'cpms': cpms,
                'varis': varis,
                'valid_paths': valid_paths,
                'cfg': cfg}
        pickle.dump(dump, f)
    print(f'{output_model} saved')

    # route_file
    if not route_file:
        route_for_dictions = {k: v['path'] for k, v in valid_paths.items()}
        file_output = cfg.output_path.joinpath(f'model_{od_name}_{len(valid_paths)}.json')
        with open(file_output, 'w') as f:
            json.dump(route_for_dictions, f, indent=4)
        print(f'{file_output} saved')

    return output_model


@app.command()
def single(file_dmg: str,
           od_name: str='Wooroloo_Merredin',
           route_file: str=None) -> None:
    """
    file_dmg:
    od_name: 'York_Merredin'
    route_file:
    """
    file_model = setup_model(file_dmg=file_dmg, od_name=od_name, route_file=route_file)

    run_survivability(file_model=file_model)

    run_inference(file_model=file_model)


@app.command()
def batch(file_dmg: str):

    cfg = config.Config(HOME.joinpath('./config.json'))

    for key in cfg.infra['ODs']:
        single(file_dmg, key)


@app.command()
def run_inference(file_model: str) -> None:

    with open(file_model, 'rb') as f:
        dump = pickle.load(f)

    cfg = dump['cfg']
    cpms = dump['cpms']
    varis = dump['varis']
    valid_paths = dump['valid_paths']
    path_names = list(valid_paths.keys())
    no_paths = len(path_names)

    od_name = '_'.join(path_names[0].split('_')[:-1])
    VE_ord = list(cfg.infra['nodes'].keys()) + path_names
    vars_inf = inference.get_inf_vars(cpms, od_name, VE_ord)

    Mod = inference.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v != od_name])
    #Mod.sort()

    plt.figure()
    p_flat = Mod.p.flatten()
    elapsed = [varis[od_name].values[int(x[0])] for x in Mod.C]

    elapsed_in_mins = [f'{(x/60):.1f}' for x in elapsed]
    elapsed_in_mins = [f'{no_paths-x[0]}: {y}' if x[0] else y for x, y in zip(Mod.C, elapsed_in_mins)]
    plt.bar(range(len(p_flat)), p_flat, tick_label=elapsed_in_mins)

    plt.xlabel("Travel time (mins)")
    plt.ylabel("Probability")
    file_output = cfg.output_path.joinpath(f'{Path(file_model).stem}_travel.png')
    plt.savefig(file_output, dpi=100)
    print(f'{file_output} saved')

    prob_by_path = {f'{od_name}_{no_paths-x[0]}': y for x, y in zip(Mod.C, p_flat) if x[0]}

    # create shp file
    json_file = Path(file_model).parent.joinpath(Path(file_model).stem + '_paths.json')
    if json_file.exists():
        outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_direction.shp')
        create_shpfile(json_file, prob_by_path, outfile)

    # create shp file for node failure
    outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_nodes.shp')
    create_shpfile_nodes(cfg.infra['nodes'], outfile)


@app.command()
def run_survivability(file_model: str) -> None:
    """
    compute the likelihood of routes functioning after the event
    file_model:
    file_dmg

    """

    with open(file_model, 'rb') as f:
        dump = pickle.load(f)

    cfg = dump['cfg']
    cpms = dump['cpms']
    varis = dump['varis']
    valid_paths = dump['valid_paths']
    path_names = list(valid_paths.keys())
    no_paths = len(path_names)
    od_name = '_'.join(path_names[0].split('_')[:-1])

    # computing survivability (P(path=1))
    paths_rel = {}
    VE_ord = list(cfg.infra['nodes'].keys()) + path_names
    for path in path_names:
        vars_inf = inference.get_inf_vars(cpms, path, VE_ord)
        Mpath = inference.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v!=path])
        paths_rel[path] = Mpath.p[Mpath.C==1][0]

    # FIXME: computing P(path=1|S)
    pdb.set_trace()
    vars_inf = inference.get_inf_vars(cpms, od_name, VE_ord)
    Mobs = inference.condition([cpms[v] for v in vars_inf], [path_names[0]], [1])
    # P(sys, path)
    Mod = inference.variable_elim(Mobs, [v for v in vars_inf if v != od_name])
    # P(path=1|S=1) = P(p,S) / P(S)

    fig, ax = plt.subplots()
    ax.bar(range(len(paths_rel)), paths_rel.values(), tick_label=list(paths_rel.keys()))
    fig.autofmt_xdate()
    ax.set_xlabel(f"Route: {od_name}")
    ax.set_ylabel("Reliability")

    file_output = cfg.output_path.joinpath(f'{Path(file_model).stem}_routes.png')
    fig.savefig(file_output, dpi=100)
    print(f'{file_output} saved')

    # create shp file
    json_file = Path(file_model).parent.joinpath(Path(file_model).stem + '_paths.json')
    if json_file.exists():
        outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_routes.shp')
        create_shpfile(json_file, paths_rel, outfile)

        # nodes
        outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_nodes.csv')
        df_pf = pd.Series({x: v['failure'] for x, v in cfg.infra['nodes'].items()}, name='failure')
        df_pf.sort_values(ascending=False).to_csv(outfile, float_format='%.2f')


if __name__=='__main__':

    app()

