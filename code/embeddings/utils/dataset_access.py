import collections
import functools
import json
import numpy as np
import pandas as pd
import os
import random

from . import ast_utils
from . import neighbors
from . import graphs

from code.utils.utils import EMB_DIR, STU_DIR

RAWDATA_DIR = STU_DIR
GENERATED_DIR = EMB_DIR 

def ast_counts(dataset_id, json_available=True):
    counts_file = os.path.join(RAWDATA_DIR, dataset_id, 'asts', 'counts.txt')
    df = pd.read_csv(
        counts_file,
        sep='\t',
        dtype={'astId': str, 'count': int },
    )

    counts = collections.defaultdict(int)
    for i, row in df.iterrows():
        counts[row['astId']] = row['count' if dataset_id == 'hoc18' else 'counts']
    return counts


@functools.lru_cache(maxsize=None)
def load_annotations(dataset_id, seed=None):
    print('Loading annotations')

    annotations_file = os.path.join(
        RAWDATA_DIR,
        dataset_id,
        'groundTruth',
        'groundTruth.txt',
    )

    df = pd.read_csv(
        annotations_file,
        sep=',',
        dtype={'astId': str, 'nextId': str },
    )

    counts = ast_counts(dataset_id)

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    annotated_counts = [ counts[str(row['astId'])] for _, row in df.iterrows() ]
    annotated_counts.sort(reverse=True)
    head_tail_size = 40
    head = annotated_counts[:head_tail_size]
    tail = annotated_counts[-head_tail_size:]

    annotations_val = dict()
    annotations_test = dict()
    for annotations in (annotations_val, annotations_test):
        for key in ['all', 'head', 'tail']:
            annotations[key] = collections.defaultdict(list)

    validation_split = 0.0

    for i, row in df_shuffled.iterrows():
        if i < validation_split * len(df):
            annotations = annotations_val
        else:
            annotations = annotations_test
        astId = str(row['astId'])
        count = counts[astId]
        nextId = str(row['nextId'])
        ast = load_ast(astId, dataset_id)
        if ast is None:
            continue
        next = load_ast(nextId, dataset_id)
        if next is None:
            continue
        if not neighbors.check_size_and_depth(ast) or not graphs.constraints(ast):
            continue
        if not neighbors.check_size_and_depth(next) or not graphs.constraints(next):
            continue
        if next not in graphs.default_neighbors(ast, dataset_id):
            continue
        annotations['all'][ast.to_int()].append(next.to_int())
        if count in head:
            annotations['head'][ast.to_int()].append(next.to_int())
        if count in tail:
            annotations['tail'][ast.to_int()].append(next.to_int())
    print('DONE')
    return {
        'test': annotations_test,
        'validate': annotations_val,
    }


def get_state_dirs(dataset_id):
    assert(dataset_id in ['hoc18', 'hoc4'])
    dirs = [
        os.path.join(RAWDATA_DIR, dataset_id, 'asts'),
    ]
    if dataset_id == 'hoc18':
        dirs.append(
            os.path.join(RAWDATA_DIR, dataset_id, 'unseen'),
        )
    return dirs


@functools.lru_cache(maxsize=1000000)
def load_ast(ast_id, dataset_id):
    state_dirs = get_state_dirs(dataset_id)
    for dir in state_dirs:
        json_path = os.path.join(dir, ast_id + '.json')
        try:
            with open(json_path, 'r') as opened:
                ast_json = json.load(opened)
                try:
                    return ast_utils.ASTNode.from_json(ast_json)
                except:
                    return None
        except:
            pass
    return None


def load_goal_ast(dataset_id):
    return load_ast('0', dataset_id)


@functools.lru_cache(maxsize=10)
def get_successful_interpolated_trajs(dataset_id):
    print('Loading trajectories')

    cached_trajs_file = os.path.join(GENERATED_DIR, 'trajectories_{}.json'.format(dataset_id))
    print('Checking {} for previously loaded trajectories'.format(cached_trajs_file))
    try:
        with open(cached_trajs_file, 'r') as f:
            trajs = json.load(f)
            print('DONE, loaded from existing file')
            return trajs
    except:
        print('failed, file does not exist')
        print('reading trajectories')

    traj_dir = os.path.join(RAWDATA_DIR, dataset_id, 'interpolated')

    counts_file = os.path.join(traj_dir, 'counts.txt')
    df = pd.read_csv(
        counts_file,
        sep='\t',
        dtype={'interpId': str, 'count': int },
    )

    trajs = []

    for traj_file in os.listdir(traj_dir):

        filename = os.fsdecode(traj_file)
        if filename.endswith('counts.txt') or filename.endswith('idMap.txt'):
            continue

        with open(os.path.join(traj_dir, traj_file), 'r') as opened:
            traj = []
            for line in opened:
                traj.append(line.rstrip('\n'))

        if not traj or traj[-1] != '0':
            continue

        traj_id = filename[:-len('.txt')]
        count = df[df['interpId'] == traj_id]['count'].tolist()[0]

        trajs.append((traj, count))

    trajs.sort(key=(lambda x : x[1]), reverse=True)

    with open(cached_trajs_file, 'w') as f:
        json.dump(trajs, f)

    print('DONE')
    return trajs


def sample_trajectories(n, dataset_id, seed=None):
    if seed is not None:
        random.seed(seed)

    if n == 0:
        return []

    trajs = get_successful_interpolated_trajs(dataset_id)

    ids = []
    for i, (traj, count) in enumerate(trajs):
        ids.extend([i] * count)
    sampled_ids = random.sample(ids, n)

    sampled_counts = collections.defaultdict(int)
    for i in sampled_ids:
        sampled_counts[i] += 1
    unique_ids = list(sampled_counts.keys())

    sampled_trajs = [
        (fix_traj(trajs[i][0], dataset_id), sampled_counts[i]) for i in unique_ids
    ]
    return sampled_trajs


def read_traj(traj, dataset_id):
    pairs = []
    prev = load_ast(traj[0], dataset_id)
    for next_id in traj[1:]:
        next = load_ast(next_id, dataset_id)
        if prev is not None and next is not None and next in graphs.default_neighbors(prev, dataset_id):
            pairs.append((prev, next))
        prev = next
    return pairs


def weight_states_from_traj_sample(trajs, dataset_id, repeated=False):
    empty_id = '34' if dataset_id == 'hoc18' else '51'
    weights = collections.defaultdict(int)
    for traj, count in trajs:
        if repeated:
            state_ids = traj
        else:
            state_ids = list(set(traj))

        for i, state_id in enumerate(state_ids):
            weights[state_id] += count

    weights[empty_id] = 1 if len(trajs) else 0
    return weights


def weight_edges_from_traj_sample(trajs, dataset_id):
    weights = collections.defaultdict(int)
    for traj, count in trajs:
        for prev, next in zip(traj, traj[1:]):
            weights[(prev, next)] += count
    return weights


def fix_traj(traj, dataset_id):
    if dataset_id != 'hoc18':
        return traj
    prev_id = traj[0]
    fixed_traj = [ prev_id ]
    goal = load_ast('0', dataset_id)
    for next_id in traj[1:]:
        if next_id == '7':
            prev = load_ast(prev_id, dataset_id)
            if prev is not None and goal in graphs.default_neighbors(prev, dataset_id):
                fixed_traj.append('0')
                return fixed_traj
        fixed_traj.append(next_id)
        prev_id = next_id
    return fixed_traj


def get_graph_path(dataset_id):
    graph_path = os.path.join(
        GENERATED_DIR,
        'graph_'+dataset_id+'.json'
    )
    return graph_path


def load_graph(dataset_id):
    graph_path = get_graph_path(dataset_id)
    if not os.path.isfile(graph_path):
        return None
    with open(graph_path, 'r') as opened:
        g_str = json.load(opened)
        g = { int(u): [ int(v) for v in gu ] for u, gu in g_str.items() }
        return g


def get_states_path(dataset_id):
    states_path = os.path.join(
        GENERATED_DIR,
        'nodes_'+dataset_id+'.json'
    )
    return states_path

def generate_states_with_distances_to_goal(dataset_id):
    path = get_states_path(dataset_id)

    g = load_graph(dataset_id)
    assert(g is not None) #Generate graph first'

    ds_to_goal = graphs.distances_bfs(graphs.transpose(g), load_goal_ast(dataset_id))[0]

    with open(path, 'w') as f:
        json.dump(ds_to_goal, f)
    return ds_to_goal

@functools.lru_cache(maxsize=10)
def load_states_with_distances_to_goal(dataset_id):
    path = get_states_path(dataset_id)
    if not os.path.isfile(path):
        generate_states_with_distances_to_goal(dataset_id)

    with open(path, 'r') as opened:
        d_str = json.load(opened)
        d = { int(u): int(v) for u, v in d_str.items() }
        return d


def random_ast(dataset_id, seed=None):
    if seed is not None:
        random.seed(seed)

    state_ints = load_states_with_distances_to_goal(dataset_id)

    ast_int = np.random.choice(list(state_ints.keys()))
    return ast_utils.ASTNode.from_int(ast_int)


def distance_to_goal(ast, dataset_id):
    ds = load_states_with_distances_to_goal(dataset_id)
    return ds[ast.to_int()]

