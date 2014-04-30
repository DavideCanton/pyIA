__author__ = 'davide'

from labyrinth import Labyrinth, NeighborsGenerator
import numpy as np

if __name__ == "__main__":
    lab = Labyrinth(6, 5)
    lab.start = 0, 0
    lab.goal = 5, 4
    lab.walls = {(2, 2), (3, 2), (4, 2), (5, 2)}

    n = NeighborsGenerator(lab)
    current = np.array([1, 2])
    direction = np.array([1, 1])

    # the correct result is [3,4], but the third call
    # returns None
    print(n.jump_rec(current, direction, lab.goal))
    print(n.jump_it_1(current, direction, lab.goal))
    print(n.jump_it_2(current, direction, lab.goal))

    for i in range(lab.w):
        print("|", end="")
        for j in range(lab.h):
            if (i, j) == lab.start:
                print("S|", end="")
            elif (i, j) == lab.goal:
                print("G|", end="")
            else:
                print(" " if lab[i, j] else "X", end="|")
        print()
