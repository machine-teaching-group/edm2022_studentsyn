from collections import deque
import heapq
import json
import sys 
from . import ast_utils
from . import dataset_access
from . import neighbors


def full_graph(
    node,
    dataset_id,
    constraints=None,
    max_dist=None,
    max_size=neighbors.MAX_SIZE,
    max_depth=neighbors.MAX_DEPTH,
    all_edges=False,
    verbose=False
    ):
    
    if max_dist is None:
        max_dist = max_size + 1

    g = dict()

    q = deque()
    q.append((node, 0))

    def get_neighbors(u):
        return neighbors.constrained_generate_neighbors(u, dataset_id, constraints, max_size, max_depth)

    seen_states_count = 1
        
    while q:
        u, du = q.popleft()
        u_neighbors = set(get_neighbors(u))
        g[u.to_int()] = [ neighbor.to_int() for neighbor in u_neighbors ]
        if verbose and (seen_states_count % 1000 == 0):
            print('Total states discovered: ', seen_states_count)

        dv = du + 1
        for v in (neighbor for neighbor in u_neighbors if neighbor.to_int() not in g):      
            seen_states_count += 1
            if dv < max_dist:
                q.append((v, dv))
                g[v.to_int()] = []
            elif all_edges:
                v_neighbors = set(neighbor for neighbor in get_neighbors(v) if neighbor.to_int() in g)
                g[v.to_int()] = [ neighbor.to_int() for neighbor in v_neighbors ]

    return g


def transpose(g):
    result = dict()
    for u, neighbors in g.items():
        for v in neighbors:
            if not v in result:
                result[v] = set()
            result[v].add(u)
    return result



def distances_bfs(g, node):
    if node is None : 
        print('student data not in data/student/')  
        raise Exception

    if not isinstance(node, int):
        node_int = node.to_int()
    else:
        node_int = node
    q = deque()
    q.append(node_int)

    d = dict()
    d[node_int] = 0

    p = dict()
    p[node_int] = None
    
    while q:
        u_int = q.popleft()
        du = d[u_int]
        for v_int in g.get(u_int, []):
            if v_int in d:
                continue
            d[v_int] = du + 1
            p[v_int] = u_int
            q.append(v_int)

    return d, p


def distances_dijkstra(g, node, edge_weight):
    q = [(0, node.to_int())]

    d = dict()
    d[node.to_int()] = 0

    p = dict()
    p[node.to_int()] = None
    
    visited = set()

    while q:
        du, u_int = heapq.heappop(q)
        if u_int in visited:
            continue
        for v_int in g.get(u_int, []):
            if v_int in visited:
                continue
            w = edge_weight(u_int, v_int)
            new_w = du + w
            old_w = d.get(v_int, float('Inf'))
            if new_w < old_w:
                d[v_int] = new_w
                p[v_int] = u_int
                heapq.heappush(q, (new_w, v_int))

    return d, p


def constraints(v):
    if v._n_if_else + v._n_forever > 2:
        return False
    else:
        return True
constraints.max_n_if_else_and_forever = 2


def default_neighbors(
    node: ast_utils.ASTNode,
    dataset_id: str,
):
    yield from neighbors.constrained_generate_neighbors(
        node,
        dataset_id,
        constraints=constraints,
    )


def generate_graph(dataset_id):
    save_path = dataset_access.get_graph_path(dataset_id)

    empty_state = ast_utils.create_empty_node()
    print('Computing full graph, will save to {}'.format(save_path))
    g = full_graph(
        empty_state,
        dataset_id,
        constraints=constraints,
        max_size=neighbors.MAX_SIZE,
        max_depth=neighbors.MAX_DEPTH,
        all_edges=True,
        verbose=True,
    )

    print('Number of nodes: {}'.format(len(g)))
    import os
    cwd = os.getcwd()
    print(cwd)
    with open(save_path, 'w+') as save_opened:
        json.dump(g, save_opened)
    return g


if __name__ == '__main__':
    dataset_id = sys.argv[1]
    generate_graph(dataset_id)
