__author__ = 'davide'

from random import shuffle
from typing import Tuple

from labyrinth import Labyrinth, lab_to_im
from unionfind import UnionFind


def reverse_flatten(i: int, w: int) -> Tuple[int, int]:
    return divmod(i, w)


def flatten(i: int, j: int, w: int, h: int) -> int:
    return i * w + j


def maze(w: int, h: int, size: int = 2):
    def conv_size(n: int) -> int:
        return (n - 1) // size + 1

    nw, nh = conv_size(w), conv_size(h)
    ns = size // 2 - 1
    uf = UnionFind(nw * nh)
    lab = Labyrinth(w, h)

    for x in range(w):
        for y in range(h):
            lab[x, y] = 0

    edges = []
    for i in range(nh - 1):
        for j in range(nw - 1):
            f = flatten(i, j, nw, nh)
            edges.append((f, f + 1))  # right
            edges.append((f, f + nw))  # down

    for i in range(nh - 1):
        f = flatten(i, nw - 1, nw, nh)
        edges.append((f, f + nw))  # down

    for j in range(nw - 1):
        f = flatten(nh - 1, j, nw, nh)
        edges.append((f, f + 1))  # right

    shuffle(edges)

    while len(uf) > 1:
        u, v = edges.pop()
        y1, x1 = reverse_flatten(u, nw)
        y2, x2 = reverse_flatten(v, nw)
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            if x2 - x1 == 1:
                for i in range(size + 1):
                    for j in range(1, ns + 1):
                        ny = size * y1 - j
                        if ny >= 0:
                            lab[size * x1 + i, ny] = True
                        else:
                            break
                    lab[size * x1 + i, size * y1] = True
                    for j in range(1, ns + 1):
                        ny = size * y1 + j
                        if ny < h:
                            lab[size * x1 + i, ny] = True
                        else:
                            break
            else:
                for i in range(size + 1):
                    for j in range(1, ns + 1):
                        nx = size * x1 - j
                        if nx >= 0:
                            lab[nx, size * y1 + i] = True
                        else:
                            break
                    lab[size * x1, size * y1 + i] = True
                    for j in range(1, ns + 1):
                        nx = size * x1 + j
                        if nx < w:
                            lab[nx, size * y1 + i] = True
                        else:
                            break

    lab[0, 0] = 1
    lab.start = 0, 0
    lab[lab.w - 2, lab.h - 2] = 1
    lab.goal = lab.w - 2, lab.h - 2

    return lab


def empty_maze(w: int, h: int) -> Labyrinth:
    lab = Labyrinth(w, h)
    lab[0, 0] = 1
    lab.start = 0, 0
    lab[w - 2, h - 2] = 1
    lab.goal = w - 2, h - 2
    return lab


def main() -> None:
    w, h = 3, 5

    e = maze(w, h)
    im = lab_to_im(e)
    im.show()


if __name__ == '__main__':
    main()
