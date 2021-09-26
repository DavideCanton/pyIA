__author__ = 'Kami'

import itertools as it
import time
from typing import Tuple, List, TypeVar, Iterable, Iterator

import numpy as np

import labyrinth as lab_module
import searching

# file path
LAB_PATH = "img/lab2.bmp"

# LAB_PATH = "img/map/ost100d.map"

PathEntry = Tuple[int, int]
Path = List[PathEntry]
T = TypeVar('T')


def pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def reconstruct_path(path: Path) -> Path:
    """
    Reconstructs the complete path inserting nodes which
    were "jumped" by the generating function. This is possible
    because JPS generates jump only in straight or diagonal lines.

    For example:
    [(0,0),(4,4)] -> [(0,0),(1,1),(2,2),(3,3),(4,4)].
    """
    expanded_path: Path = [path[0]]
    for cur_node, next_node in pairwise(path):
        cur_node_a = np.array(cur_node)
        next_node_a = np.array(next_node)
        direction = lab_module.normalize(next_node_a - cur_node_a)
        while not np.array_equal(cur_node_a, next_node_a):
            cur_node_a = cur_node_a + direction
            cur_node_n = (int(cur_node_a[0]), int(cur_node_a[1]))
            expanded_path.append(cur_node_n)
    return expanded_path


def load_lab(path: str) -> lab_module.Labyrinth:
    if path.endswith("map"):
        return lab_module.load_from_map_file(path, img=False)[0]
    else:
        return lab_module.load_from_img(path)[0]


def compute_path(labyrinth: lab_module.Labyrinth) -> Tuple[Path, float, searching.Info]:
    children_gen = lab_module.NeighborsGeneratorJPS(labyrinth)
    heur_goal = lab_module.heur_diag(labyrinth.goal, labyrinth.start)

    cur_time = time.perf_counter()
    path, *_, info = searching.a_star(labyrinth.start,
                                      lambda p: p == labyrinth.goal,
                                      heur_goal,
                                      children_gen)
    cur_time = time.perf_counter() - cur_time

    if path:
        path = reconstruct_path(path)

    return path, cur_time, info


def main() -> None:
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


if __name__ == '__main__':
    main()
