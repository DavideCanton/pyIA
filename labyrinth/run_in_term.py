__author__ = 'davide'

import labyrinth as lab_module
import numpy as np
import searching
import time
import itertools as it

# file path
LAB_PATH = "D:/labyrinth/lab2.bmp"
#LAB_PATH = "D:/labyrinth/map/ost000a.map"


def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def reconstruct_path(path, labyrinth):
    """
    Reconstructs the complete path inserting nodes which
    were "jumped" by the generating function. This is possible
    because JPS generates jump only in straight or diagonal lines.

    For example:
    [(0,0),(4,4)] -> [(0,0),(1,1),(2,2),(3,3),(4,4)].
    """
    expanded_path = [path[0]]
    for cur_node, next_node in pairwise(path):
        cur_node = np.array(cur_node)
        next_node = np.array(next_node)
        direction = lab_module.normalize(next_node - cur_node)
        while not np.array_equal(cur_node, next_node):
            cur_node = tuple(int(x) for x in cur_node + direction)
            expanded_path.append(cur_node)
    return expanded_path


def load_lab(path):
    if path.endswith("map"):
        return lab_module.load_from_map_file(path, img=False)[0]
    else:
        return lab_module.load_from_img(path)[0]


def compute_path(labyrinth):
    children_gen = lab_module.NeighborsGenerator(labyrinth)
    heur_goal = lab_module.heur_diag(labyrinth.goal, labyrinth.start)

    eq_to_goal = lambda p: p == labyrinth.goal
    cur_time = time.perf_counter()
    path, *_, info = searching.a_star(labyrinth.start, eq_to_goal,
                                      heur_goal, children_gen)
    cur_time = time.perf_counter() - cur_time

    if path:
        path = reconstruct_path(path, labyrinth)

    return path, cur_time, info


if __name__ == "__main__":
    labyrinth = load_lab(LAB_PATH)

    print("Start detected:\t{}".format(labyrinth.start))
    print("Goal detected:\t{}".format(labyrinth.goal))
    print("Starting search...")

    path, cur_time, info = compute_path(labyrinth)

    print("Search ended")
    print("Time:", round(cur_time, 2), "s")
    print("Nodes searched:", info.nodes)
    print("Maximum list size:", info.maxl)

    if path is None:
        print("Path not found")
    else:
        print("Found path of", len(path), "nodes")
        print("Path generation completed")
        print("*" * 100)
