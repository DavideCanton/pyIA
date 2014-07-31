__author__ = 'davide'

from random import shuffle
from unionfind import UnionFind
from labyrinth import Labyrinth, lab_to_im


def unflatten(i, w, h):
    return divmod(i, w)


def flatten(i, j, w, h):
    return i * w + j


def maze(w, h):
    uf = UnionFind(w * h)
    lab = Labyrinth(2 * w - 1, 2 * h - 1)

    for x in range(lab.w):
        for y in range(lab.h):
            lab[x, y] = 0

    edges = []
    for i in range(h - 1):
        for j in range(w - 1):
            f = flatten(i, j, w, h)
            edges.append((f, f + 1))  # right
            edges.append((f, f + w))  # down

    for i in range(h - 1):
        f = flatten(i, w - 1, w, h)
        edges.append((f, f + w))  # down

    for j in range(w - 1):
        f = flatten(h - 1, j, w, h)
        edges.append((f, f + 1))  # right

    shuffle(edges)

    while len(uf) > 1:
        u, v = edges.pop()
        y1, x1 = unflatten(u, w, h)
        y2, x2 = unflatten(v, w, h)
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            if x2 - x1 == 1:
                lab[2 * x1, 2 * y1] = True
                lab[2 * x1 + 1, 2 * y1] = True
                lab[2 * x1 + 2, 2 * y1] = True
            else:
                lab[2 * x1, 2 * y1] = True
                lab[2 * x1, 2 * y1 + 1] = True
                lab[2 * x1, 2 * y1 + 2] = True

    lab[0, 0] = 1
    lab.start = 0, 0
    lab[lab.w - 2, lab.h - 2] = 1
    lab.goal = lab.w - 2, lab.h - 2

    return lab


def empty_maze(w, h):
    lab = Labyrinth(2 * w - 1, 2 * h - 1)
    lab[0, 0] = 1
    lab.start = 0, 0
    lab[lab.w - 2, lab.h - 2] = 1
    lab.goal = lab.w - 2, lab.h - 2
    return lab


if __name__ == "__main__":
    w, h = 3, 5

    e = maze(w, h)
    im = lab_to_im(e)
    im.show()
