# %%

import numpy as np
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt


def alphashape(pos, alpha=1):
    """
    Plots the alpha shape for the input positions and alpha value. Note,
    argument alpha is actually 1/alpha, as is standard for most implementations
    of alphashape algorithms.

    The alphashape function was reimplemented instead of using the existing
    alphashape library on PyPI. The library on PyPI is inefficient for a
    strictly 2D alpha-shape implementation. For the county graph, this takes
    953 us compared to alphashape 22 ms. Therefore this function represents a
    x23 improvement. It has yet to be tested if this efficiency improvement
    carriers over to larger datasets. The improvement should be even greater,
    hopefully reducing alpha-shape computing times from minutes to seconds.

    For the block graph, this function takes 4.39 s. The PyPI library takes
    67 seconds. This represents a speed up of x15.26.

    """
    ### Edges in Delaunay shape are guaranteed to be superset of those
    ### in alpha shape
    tri = Delaunay(pos)
    all_tri = pos[tri.vertices]

    ### The edges that can be in alphashape are those where the circumradius
    ###   is smaller than the alpha value

    ### Circumcenter calculation from wikipedia
    a, b, c = all_tri[:, 0], all_tri[:, 1], all_tri[:, 2]
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ay_by = ay - by
    by_cy = by - cy
    cy_ay = cy - ay
    ax_cx = ax - cx
    bx_ax = bx - ax
    cx_bx = cx - bx

    dinv = 1 / (2 * (ax * by_cy + bx * cy_ay + cx * ay_by))
    ux = dinv * (a2 * by_cy + b2 * cy_ay + c2 * ay_by)
    uy = dinv * (a2 * cx_bx + b2 * ax_cx + c2 * bx_ax)
    circumcenter = np.hstack([ux[:, None], uy[:, None]])
    circumradii = np.linalg.norm(all_tri[:, 0] - circumcenter, axis=-1)
    keep_idx = np.where(circumradii < (1.0 / alpha))[0]
    keep_tri = tri.vertices[keep_idx]
    keep_tri = np.sort(keep_tri, axis=-1)

    ### Edges on outside only appear once, so remove interior edges by
    ###   finding edges that appear twice
    count_edges = {}
    delete_edges = {}
    all_edges = np.vstack(
        [
            keep_tri[:, [0, 1]],
            keep_tri[:, [0, 2]],
            keep_tri[:, [1, 2]],
        ]
    )
    all_edges = list(map(tuple, all_edges))
    for edge in all_edges:
        if edge in count_edges:
            delete_edges[edge] = True
        else:
            count_edges[edge] = True

    keep_edges = []
    link_edges_dict = {}
    for edge in count_edges:
        if edge in delete_edges:
            continue
        else:
            keep_edges.append(edge)
            if edge[0] not in link_edges_dict:
                link_edges_dict[edge[0]] = []
            if edge[1] not in link_edges_dict:
                link_edges_dict[edge[1]] = []
            link_edges_dict[edge[0]].append(edge[1])
            link_edges_dict[edge[1]].append(edge[0])

    points = np.unique(np.hstack(keep_edges))

    start_point = list(link_edges_dict.keys())[0]
    edges = [start_point, link_edges_dict[start_point][0]]
    for i in range(len(points) - 1):
        edges += [x for x in link_edges_dict[edges[-1]] if x != edges[-2]]

    return points, edges


def get_graph_point_clouds(g, pos, attribute="county"):
    point_clouds_idx = {}
    for idx, node in enumerate(g.nodes):
        temp_attr = g.nodes[node][attribute]
        if temp_attr not in point_clouds_idx:
            point_clouds_idx[temp_attr] = []
        point_clouds_idx[temp_attr].append(idx)
    point_clouds = {}
    for key, idx in point_clouds_idx.items():
        point_clouds[key] = pos[idx]
    return point_clouds, point_clouds_idx


def get_graph_alphashapes(
    g,
    pos,
    attribute="county",
    alpha=1,
):
    point_clouds, _ = get_graph_point_clouds(g, pos)
    pos_dict = {}
    points_dict = {}
    edges_dict = {}
    for key, temp_pos in point_clouds.items():
        alpha_points, alpha_edges = alphashape(temp_pos, alpha)
        pos_dict[key] = temp_pos
        points_dict[key] = alpha_points
        edges_dict[key] = alpha_edges
    return pos_dict, points_dict, edges_dict


def plot_alphashape(
    pos,
    points,
    edges,
    ax=None,
    plot_pos=False,
    plt_kw={"color": "k"},
    scatter_kw={"color": "k"},
    pos_kw={"color": "tab:blue"},
):
    if ax == None:
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)
    plot_points = pos[points]
    plot_edges = pos[edges]
    if plot_pos:
        ax.scatter(pos[:, 0], pos[:, 1], **pos_kw)
    ax.scatter(plot_points[:, 0], plot_points[:, 1], **plt_kw)
    ax.plot(plot_edges[:, 0], plot_edges[:, 1], **scatter_kw)
    ax.set_aspect("equal")

    return ax


def plot_point_cloud_alphashape(
    pos_dict,
    points_dict,
    edges_dict,
    ax=None,
    plot_pos=False,
    plt_kw={"color": "k"},
    scatter_kw={"color": "k"},
    pos_kw={"color": "tab:blue"},
):
    if ax == None:
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)

    for key in points_dict:
        plot_alphashape(
            pos_dict[key],
            points_dict[key],
            edges_dict[key],
            ax=ax,
            plot_pos=plot_pos,
            plt_kw=plt_kw,
            scatter_kw=scatter_kw,
            pos_kw=pos_kw,
        )

    return ax


def get_pos(g):
    return np.vstack(
        [[g.nodes[x]["latitude"], g.nodes[x]["longitude"]] for x in g.nodes]
    )


# %%

if __name__ == "__main__":
    import pickle5

    ### Plotting simple outline of state from county centers using alpha-shape
    with open("county_graph.pickle", "rb") as pickle_file:
        g = pickle5.load(pickle_file)
    pos = get_pos(g)
    points, edges = alphashape(pos, alpha=1)
    plot_alphashape(pos, points, edges, plot_pos=True)
    plt.show()

    ### Plotting county lines of entire state from block_graph using alpha-shapes
    with open("block_graph.pickle", "rb") as pickle_file:
        g = pickle5.load(pickle_file)
    pos = get_pos(g)
    ### Change pos x,y for some reason...
    pos[:, [0, 1]] = pos[:, [1, 0]]
    pos_dict, points_dict, edges_dict = get_graph_alphashapes(g, pos, alpha=20)
    plot_point_cloud_alphashape(pos_dict, points_dict, edges_dict)
    plt.show()
