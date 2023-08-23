import numpy as np
import random
from PIL import Image
import networkx as nx
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# %matplotlib widget

from cubic_spline import CubicSpline2D
from cubic_polynomial_fit import CubicPolynomial2D, cubic_func
from extract_network import find_color, extract_network, render_skeleton, render_network

def image_to_graph(img: Image, simplify: bool=True) -> nx.Graph:
    rgb = (255, 255, 255)
    px = find_color(img, rgb).T
    return extract_network(px, min_distance=4, simplify=simplify)

def graph_to_polylines(g: nx.Graph, simplify: bool=True) -> list:
    polylines = []
    for (n1, n2, k) in g.edges(keys=True):
        edge = g[n1][n2][k]
        path = edge['path']
        if simplify:
            coords = np.array(path.coords)
        else:
            coords = np.array(path)
        polylines.append(coords)
    return polylines


def correct_path_direction(path: list[tuple], n1: tuple, n2: tuple) -> list[tuple]:
    if n1 == path[0]:
        return path
        # print('forward path list')
    elif n1 == path[-1]:
        return path[::-1]
        # print('reverse path list')
    else:
        print(f'cannot find node = {n1}, in path start = {path[0]}, end = {path[-1]}')
        return []


def connect_small_gaps(graph: nx.Graph, nodes: list[tuple], thresh: int=4) -> nx.Graph:
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes[i+1:]):
            dist = np.hypot(n1[0] - n2[0], n1[1] - n2[1])
            if dist <= thresh:
                n1, n1_neighbour, k = list(graph.edges(n1, keys=True))[0]
                n2, n2_neighbour, k = list(graph.edges(n2, keys=True))[0]
                
                e1 = graph[n1][n1_neighbour][k]
                e1_path = e1['path']
                e2 = graph[n2][n2_neighbour][k]
                e2_path = e2['path']

                new_path = e1_path + e2_path
                graph.remove_edge(n1, n1_neighbour)
                graph.remove_edge(n2, n2_neighbour)
                graph.add_edge(n1, n2_neighbour, path=new_path, d=len(new_path) - 1)
                break

    return graph


def find_terminal_nodes(graph: nx.Graph) -> list[tuple]:
    nodes_terminal = [(node[0], node[1]) for (node, degree) in graph.degree if degree == 1]
    print(f'Found {len(nodes_terminal)} terminal nodes (nodes with only 1 degree connection)')

    return nodes_terminal


def find_branching_nodes(graph: nx.Graph, nodes_terminal: list[tuple]) -> list[tuple]:
    nodes_branching = []
    for n1 in nodes_terminal:
        n1, n1_neighbour, k = list(graph.edges(n1, keys=True))[0]
        nodes_branching.append(n1_neighbour)
    
    return nodes_branching


def normalize_dx_dy(dx, dy) -> tuple:
    norm = np.hypot(dx, dy)
    dx = dx/norm
    dy = dy/norm
    
    return dx, dy
    

def find_node_directions(graph: nx.Graph, nodes_terminal: list[tuple], nodes_branching: list[tuple], img_color: Image) -> list[tuple]:
    directed_terminals = []
    directed_branching = []
    for n1, n1b in zip(nodes_terminal, nodes_branching):
        n1, n1_neighbour, k = list(graph.edges(n1, keys=True))[0]
        dx, dy = normalize_dx_dy(n1_neighbour[0] - n1[0], n1_neighbour[1] - n1[1])
        node_angle = np.rad2deg(np.arctan2(dy, dx))

        n1_color = img_color.getpixel(n1)
        color_dx, color_dy = normalize_dx_dy(n1_color[0] - 128, 128 - n1_color[1])
        color_angle = np.rad2deg(np.arctan2(color_dy, color_dx))

        angle_diff = np.fabs(color_angle - node_angle)
        direction = 1 # inlet
        if angle_diff > 90.0: # if direction difference is smaller than 90 degrees
            direction = 0 # outlet
            dx = -dx
            dy = -dy

        # print(f'at node {n1}, node_angle = {node_angle:.1f}, color_angle = {color_angle:.1f}, angle_diff = {angle_diff:.1f}, direction = {direction}')
        directed_terminals.append((n1[0], n1[1], dx, dy, color_dx, color_dy, direction))
        directed_branching.append((n1b[0], n1b[1], dx, dy, color_dx, color_dy, direction))

    return directed_terminals, directed_branching


def track_path(path: list) -> np.ndarray:
    waypoints = []
    for i in range(len(path) - 1):
        n1 = path[i]
        n2 = path[i+1]
        e = graph[n1][n2][0]
        points = e['path']
        waypoints = waypoints + correct_path_direction(points, n1, n2)

    return waypoints


def fit_cubic_polynomial(xs: np.ndarray, ys: np.ndarray):
    cubic_spline = CubicPolynomial2D(xs, ys)
    s = np.arange(0, cubic_spline.s[-1], 1)
    ref_xy = [cubic_spline.calc_position(i_s) for i_s in s]
    ref_yaw = [cubic_spline.calc_yaw(i_s) for i_s in s]
    ref_rk = [cubic_spline.calc_curvature(i_s) for i_s in s]

    return cubic_spline, np.column_stack((ref_xy, ref_yaw, ref_rk))


def downsample_path(path: np.ndarray, ratio: int=2) -> np.ndarray:
    if path.shape[0] > ratio:
        new_path = path[::ratio]
        if np.hypot(new_path[-1, 0] - path[-1, 0], new_path[-1, 1] - path[-1, 1]) > 0.001:
            # new_path = np.delete(new_path, -1)
            new_path = np.insert(new_path, -1, path[-1], axis=0)
        return new_path
    elif path.shape[0] == 0:
        return np.array([])
    else:
        return np.take(path, [1, -1], axis=0)


def path_is_smooth(path: np.ndarray, thresh: float=60.0) -> bool:
    _, idx = np.unique(path, return_index=True, axis=0)
    path = path[np.sort(idx)]

    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    ds = np.hypot(dx, dy)
    yaw = np.arctan2(dy, dx)
    yaw_diff = np.rad2deg(np.diff(yaw))
    yaw_rate = yaw_diff / ds[:-1]
    # yaw_rate_valid = np.fabs(yaw_rate) <= thresh

    if np.max(yaw_rate) > thresh:
        # print(f'max yaw_rate: {np.max(yaw_rate):.1f}, threshold: {thresh}')
        return False
    else:
        return True


def path_is_valid(path: list) -> bool:
    # path = np.array(path)
    # for segment in path:
        
    return True


def find_paths_among_terminals(graph: nx.Graph, inlets: np.ndarray, outlets: np.ndarray, thresh: int=4) -> tuple[list[list]]:
    inlets_T = inlets.T.astype(int)
    outlets_T = outlets.T.astype(int)
    inlets = list(zip(inlets_T[0], inlets_T[1]))
    outlets = list(zip(outlets_T[0], outlets_T[1]))

    paths = []
    waypoints_all = []
    for n1 in inlets:
        for n2 in outlets:
            if nx.has_path(graph, source=n1, target=n2):
                path = nx.shortest_path(graph, n1, n2, weight='d', method='dijkstra')
                # print(f'From {n1} to {n2}: {path}')
                if path_is_valid(path):
                    waypoints_all.append(track_path(path))
                    paths.append(path)

    print(f'Found {len(paths)} paths')
    return paths, waypoints_all


def estimate_path_end_yaw(path: list[tuple], local_length: int=10) -> tuple:
    path_np = np.array(path)
    if path_np.shape[0] > local_length:
        front_path = path_np[:local_length]
        rear_path = path_np[-local_length:][::-1]
    else:
        front_path = path_np
        rear_path = path_np[::-1]
    front_cubic_poly, front_curve = fit_cubic_polynomial(front_path[:, 0], front_path[:, 1])
    rear_cubic_poly, rear_curve = fit_cubic_polynomial(rear_path[:, 0], rear_path[:, 1])
    front_yaw = front_curve[0, 2]
    rear_yaw = rear_curve[0, 2]

    return front_yaw, rear_yaw

def reduce_graph(graph: nx.Graph) -> tuple:
    for n0, degree in graph.degree:
        if 'type' in graph.nodes[n0]:
            node_type = graph.nodes[n0]['type']
        else:
            node_type = 'unknown'

        if degree < 2 or node_type == 'branch':
            continue

        # Find all edges connected at n0, and their yaw angles (pointing away from n0)
        edges = list(graph.edges(n0, keys=True))
        yaws = []
        paths = []
        nodes = []
        for n, n1, k in edges:
            if n != n0:
                e1 = graph[n][n0][k]
                print(f'edge start from node: {n0}, but from: {n}')
            else:
                e1 = graph[n0][n1][k]

            e1_path = e1['path']
            e1_path = correct_path_direction(e1_path, n0, n1) # paths pointing away from n0
            if e1_path:
                n0_yaw, n1_yaw = estimate_path_end_yaw(e1_path, 20)
                yaws.append(n0_yaw)
                paths.append(e1_path)
                nodes.append(n1)

        # Among all edges at n0, vote for the best matches
        votes = np.zeros(len(nodes), dtype=int)
        connect_matrix = np.zeros((len(nodes), len(nodes)), dtype=bool)
        for i, yaw1 in enumerate(yaws):
            diffs = []
            for j, yaw2 in enumerate(yaws):
                if i == j:
                    diffs.append(2*np.pi)
                else:
                    diffs.append(yaw1 + yaw2) # use plus here, since both angles are pointing away from n0
            
            # Pair the best matches
            min_id = np.argmin(np.fabs(diffs)) # find the minimum angle difference
            print(f'diffs: {diffs}, min_id: {min_id}')
            votes[min_id] = votes[min_id] + 1
            connect_matrix[i, min_id] = True
            connect_matrix[min_id, i] = True

        # Reconnect graph based on the vote result
        branch_ids = [i for (i, vote) in enumerate(votes) if vote > 1]
        passer_ids = [i for i in range(len(nodes)) if i not in branch_ids]

        for i in branch_ids:
            n0_new = paths[i][1]
            graph.add_node(n0_new, type='branch')
            new_path = paths[i][1:]
            graph.add_edge(n0_new, nodes[i], path=new_path, d=len(new_path) - 1)

            js = [j for (j, val) in enumerate(connect_matrix[i]) if val == True]
            for j in js:
                if j in passer_ids:
                    passer_ids.remove(j)
                new_path = [n0_new] + paths[j]
                graph.add_edge(n0_new, nodes[j], path=new_path, d=len(new_path) - 1)

        for i in passer_ids:
            # print(f'i = {i}, nodes length = {len(nodes)}, passer_ids = {passer_ids}')
            n1 = nodes[i]
            js = [j for (j, val) in enumerate(connect_matrix[i]) if val == True]
            for j in js:
                n2 = nodes[j]
                new_path = paths[i][::-1] + paths[i][1:]
                graph.add_edge(n1, n2, path=new_path, d=len(new_path) - 1)
            
        graph.remove_node(n0)
        return graph, True

    return graph, False


def extract_polylines_from_graph(graph: nx.Graph) -> np.ndarray:

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    axes[0].imshow(skel.T, cmap='gray')
    axes[1].imshow(skel.T, cmap='gray')
    axes[2].imshow(img_gray)
    axes[3].imshow(img_gray)
    axes[4].imshow(img_gray)
    axes[5].imshow(img_gray)
    axes[6].imshow(img_gray)
    axes[7].imshow(img_gray)
    axes[8].imshow(img_gray)

    # Visualize the nodes, colored by degrees of connectivity
    nodes = np.array([node for (node, degree) in graph.degree()], dtype=float)
    degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
    axes[0].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=1)
    print(f'Orignal graph has {nodes.shape[0]} nodes, degrees: {degrees}')

    # Fix small gaps in the orignal graph
    nodes_1_degree = find_terminal_nodes(graph)
    graph = connect_small_gaps(graph, nodes_1_degree, thresh=8)
    nodes_1_degree_np = np.array(nodes_1_degree)
    axes[1].scatter(nodes_1_degree_np[:, 0], nodes_1_degree_np[:, 1], c='red', s=3)

    # Simplify the graph
    changed = True
    count = 0
    while changed:
        graph, changed = reduce_graph(graph)
        count  = count + 1
    print(f'Reduced graph after {count} iterations')
    nodes = np.array([node for (node, degree) in graph.degree()], dtype=float)
    degrees = np.array([degree for (node, degree) in graph.degree()], dtype=float)
    axes[2].scatter(nodes[:, 0], nodes[:, 1], c=degrees, s=1)
    print(f'Reduced graph has {nodes.shape[0]} nodes, degrees: {degrees}')


    # Find terminals
    nodes_terminal = find_terminal_nodes(graph)
    nodes_branching = find_branching_nodes(graph, nodes_terminal)
    nodes_terminal, nodes_branching = find_node_directions(graph, nodes_terminal, nodes_branching, img_color)
    
    nodes_terminal_np = np.array(nodes_terminal)
    nodes_branching_np = np.array(nodes_branching)
    inlets = nodes_terminal_np[nodes_terminal_np[:, -1] > 0.5]
    outlets = nodes_terminal_np[nodes_terminal_np[:, -1] < 0.5]
    # inlets = nodes_branching_np[nodes_branching_np[:, -1] > 0.5]
    # outlets = nodes_branching_np[nodes_branching_np[:, -1] < 0.5]
    print(f'found {inlets.shape[0]} inlets, {outlets.shape[0]} outlets')

    axes[3].quiver(inlets[:, 0], inlets[:, 1], inlets[:, 2], inlets[:, 3], 
                   color='r', angles='xy', scale_units='xy', scale=0.1)
    axes[3].quiver(outlets[:, 0], outlets[:, 1], outlets[:, 2], outlets[:, 3], 
                   color='g', angles='xy', scale_units='xy', scale=0.1)
    # axes[3].quiver(nodes_directed_np[:, 0], nodes_directed_np[:, 1], nodes_directed_np[:, 4], nodes_directed_np[:, 5], 
    #                color='b', angles='xy', scale_units='xy', scale=0.1)
    paths, path_waypoints = find_paths_among_terminals(graph, inlets, outlets)

    for path, waypoints in zip(paths, path_waypoints):
        path = np.array(path, dtype=float)
        xs = path[:, 0]
        ys = path[:, 1]
        axes[4].plot(xs, ys)

        dx = np.diff(path[:, 0])
        dy = np.diff(path[:, 1])
        dx, dy = normalize_dx_dy(dx, dy)
        axes[5].quiver(xs[:-1], ys[:-1], dx, dy, color='g', angles='xy', scale_units='xy', scale=0.1)
        
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    axes[2].set_aspect('equal')
    axes[3].set_aspect('equal')
    axes[4].set_aspect('equal')
    axes[5].set_aspect('equal')

    plt.show()
    
    return graph


if __name__ == '__main__':
    img_id = 1
    gray_file = f'samples/gt/{img_id}_gray.png'
    color_file = f'samples/gt/{img_id}_scatter.png'
    img_gray = Image.open(gray_file)
    img_color = Image.open(color_file)

    simplify = False
    skel, graph = image_to_graph(img_gray, simplify=simplify)
    polylines = graph_to_polylines(graph, simplify=simplify)

    # connect_terminal_nodes(graph)
    extract_polylines_from_graph(graph)
