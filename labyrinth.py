from math import sqrt
import numpy as np
from collections import defaultdict, namedtuple
from PIL import Image

DIRS = U, L, D, R, UL, UR, DL, DR = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, -1),
                                     (1, -1), (-1, 1), (1, 1)]

def dist_2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return sqrt(dx * dx + dy * dy)


def vsum(x, y):
    return tuple([a + b for a, b in zip(x, y)])


class NeighboursGenerator:
    def __init__(self, labyrinth):
        self.labyrinth = labyrinth

    def _U(self, x, y):
        return y > 0 and self.labyrinth[x, y - 1] == 1

    def _D(self, x, y):
        return y < self.labyrinth.h - 1 and self.labyrinth[x, y + 1] == 1

    def _L(self, x, y):
        return x > 0 and self.labyrinth[x - 1, y] == 1

    def _R(self, x, y):
        return x < self.labyrinth.w - 1 and self.labyrinth[x + 1, y] == 1

    def __call__(self, n, dir=None):
        x, y = n
        if self._U(x, y):
            yield vsum(n, U), 1
        if self._D(x, y):
            yield vsum(n, D), 1
        if self._L(x, y):
            yield vsum(n, L), 1
        if self._R(x, y):
            yield vsum(n, R), 1


SQRT_2 = sqrt(2)
MAX_ALIVE = 10000


class NeighboursGeneratorDiag(NeighboursGenerator):
    def __init__(self, labyrinth):
        NeighboursGenerator.__init__(self, labyrinth)

    def __call__(self, n, dir=None):
        x, y = n
        if self._U(x, y):
            yield vsum(n, U), 1
        if self._D(x, y):
            yield vsum(n, D), 1
        if self._L(x, y):
            yield vsum(n, L), 1
        if self._R(x, y):
            yield vsum(n, R), 1
        if (self._U(x, y) and self._L(x, y - 1) or
            self._L(x, y) and self._U(x - 1, y)):
            yield vsum(n, UL), SQRT_2
        if (self._U(x, y) and self._R(x, y - 1) or
            self._R(x, y) and self._U(x + 1, y)):
            yield vsum(n, UR), SQRT_2
        if (self._D(x, y) and self._L(x, y + 1) or
            self._L(x, y) and self._D(x - 1, y)):
            yield vsum(n, DL), SQRT_2
        if (self._D(x, y) and self._R(x, y + 1) or
            self._R(x, y) and self._D(x + 1, y)):
            yield vsum(n, DR), SQRT_2


class NeighborsGeneratorPruning(NeighboursGeneratorDiag):
    def __init__(self, labyrinth):
        NeighboursGeneratorDiag.__init__(self, labyrinth)

    def __call__(self, current, parent=None):
        neighbors = NeighboursGeneratorDiag.__call__(self, current)
        if parent is None:
            yield from neighbors
        else:
            current = np.array(current)
            neighbors = [np.array(n[0]) for n in neighbors]
            parent = np.array(parent)
            move = current - parent
            move = normalize(move)

            if move.all(): #se nessuno e' 0 allora e' una mossa diagonale
                neighbors = self._pruneDiag(neighbors, current, move)
            else:
                neighbors = self._pruneStraight(neighbors, current, move)
            act_neighbors = []
            for n in neighbors:
                n = self._jump(current, n - current, self.labyrinth.goal)
                if n is not None:
                    t = tuple(int(x) for x in n)
                    act_neighbors.append((t, dist_2(current, n)))
            yield from act_neighbors

    def compute_forcedStraight(self, n, move):
        pruned = []
        for direct in orthogonal(move):
            dirt = n + direct
            if dirt in self.labyrinth and self.labyrinth[dirt] == 0:
                pruned.append(dirt + move)
        return pruned

    def compute_forcedDiag(self, parent, move):
        pruned = []
        for c in components(move):
            ob = parent + c
            if ob in self.labyrinth and self.labyrinth[ob] == 0:
                pruned.append(ob + c)
        return pruned

    def _pruneStraight(self, neighbors, n, move):
        pruned = [n + move]
        pruned.extend(self.compute_forcedStraight(n, move))
        return [p for p in pruned if any(np.array_equal(p, x) for x in neighbors)]

    def _pruneDiag(self, neighbors, n, move):
        pruned = [n + d for d in components(move)]
        #if all(self.labyrinth[x] == 1 for x in pruned):
        pruned.append(n + move)
        parent = n - move
        pruned.extend(self.compute_forcedDiag(parent, move))
        return [p for p in pruned if any(np.array_equal(p, x) for x in neighbors)]

    def _jump(self, current, direction, goal):
        next = current + direction
        if not self.labyrinth[next] or next not in self.labyrinth:
            return None
        if np.array_equal(next, goal):
            return next
        isDiag = direction.all()
        if isDiag:
            if all(not self.labyrinth[current + dirs]
                for dirs in components(direction)):
                return None
            forced = self.compute_forcedDiag(current, direction)
        else:
            forced = self.compute_forcedStraight(next, direction)
        if any(self.labyrinth[f] for f in forced):
            return next

        if isDiag:
            for dirt in components(direction):
                if self._jump(next, dirt, goal) is not None:
                    return next
        return self._jump(next, direction, goal)

    def _jumpi(self, current, direction, goal):
        retval = None
        stack = [Snapshot(current, direction, goal, None, None, 0)]

        while stack:
            el = stack.pop()
            if el.stage == 0:
                next = el.current + el.direction
                if not self.labyrinth[next] or next not in self.labyrinth:
                    retval = None
                    continue
                if np.array_equal(next, el.goal):
                    retval = next
                    continue
                isDiag = el.direction.all()
                if isDiag:
                    if all(not self.labyrinth[el.current + dirs]
                        for dirs in components(direction)):
                        retval = None
                        continue
                    forced = self.compute_forcedDiag(el.current, el.direction)
                else:
                    forced = self.compute_forcedStraight(next, el.direction)
                if any(self.labyrinth[f] for f in forced):
                    retval = next
                    continue
                if isDiag:
                    el.stage = 1
                    el.next = next
                    stack.append(el)
                    dirs = list(components(direction))
                    el.dirs = dirs
                    snapshot = Snapshot(next, dirs[0], el.goal, next, dirs, 0)
                    stack.append(snapshot)
                    continue
                else:
                    snapshot = Snapshot(next, el.direction, el.goal, None, None, 0)
                    stack.append(snapshot)
                    continue
            elif el.stage == 1:
                r1 = retval
                if r1 is not None:
                    retval = el.next
                    continue
                el.stage = 2
                stack.append(el)
                snapshot = Snapshot(el.next, el.dirs[1], el.goal, el.next, el.dirs, 0)
                stack.append(snapshot)
                continue
            elif el.stage == 2:
                r2 = retval
                if r2 is not None:
                    retval = el.next
                    continue
                snapshot = Snapshot(el.next, el.direction, el.goal, None, None, 0)
                stack.append(snapshot)
                continue
        return retval

    def _jumpi2(self, current, direction, goal):
        stack = [(current, direction, goal)]
        while stack:
            current, direction, goal = stack.pop()
            next = current + direction
            if not self.labyrinth[next] or next not in self.labyrinth:
                return None
            if np.array_equal(next, goal):
                return next  # assuming n cannot be None
            isDiag = direction.all()
            if isDiag:
                if all(not self.labyrinth[current + dirs]
                    for dirs in components(direction)):
                    return None
                forced = self.compute_forcedDiag(current, direction)
            else:
                forced = self.compute_forcedStraight(next, direction)
            if any(self.labyrinth[f] for f in forced):
                return next

            if isDiag:
                stack.extend((next, di, goal)
                    for di in components(direction))
            else:
                stack.append((next, direction, goal))


class Snapshot:
    def __init__(self, current, direction, goal, next, dirs, stage):
        self.current = current
        self.direction = direction
        self.goal = goal
        self.next = next
        self.dirs = dirs
        self.stage = stage

    def __str__(self):
        return str(self.__dict__)


class Labyrinth:
    def __init__(self, w, h):
        self.labyrinth = defaultdict(int)
        self.w = w
        self.h = h
        self.start = None
        self.goal = None

    def __getitem__(self, item):
        return self.labyrinth[tuple(item)]

    def __contains__(self, pos):
        return 0 <= pos[0] < self.w and 0 <= pos[1] < self.h

    def __setitem__(self, key, value):
        self.labyrinth[tuple(key)] = value


def orthogonal(move):
    move = move.copy()
    move[[0, 1]] = move[[1, 0]]
    yield move
    yield -move


def components(move, vert=True):
    move = move.copy()
    indexes = (1, 0) if vert else (0, 1)
    for ind in indexes:
        d1 = move.copy()
        d1[ind] = 0
        yield d1


def normalize(move):
    f = move[0] if move[0] else move[1]
    return move / abs(f)


def load_from_img(imgpath):
    im = Image.open(imgpath)
    pix = im.load()
    h, w = im.size
    labyrinth = Labyrinth(w, h)

    for i in range(w):
        for j in range(h):
            #avoid alpha
            pixel = pix[j, i][:3]
            if pixel == (255, 255, 255):
                labyrinth[i, j] = 1
            elif pixel == (255, 0, 0):
                labyrinth[i, j] = 1
                labyrinth.start = i, j
            elif pixel == (0, 255, 0):
                labyrinth[i, j] = 1
                labyrinth.goal = i, j

    return labyrinth, im


def load_from_map_file(filepath):
    i, w, h = 0, 0, 0
    map_started = False
    with open(filepath) as map_file:
        for line in map_file:
            if line.startswith("height"):
                w = int(line.split()[1])
            elif line.startswith("width"):
                h = int(line.split()[1])
            elif line.startswith("map"):
                labyrinth = Labyrinth(w, h)
                map_started = True
            elif map_started:
                for j, c in enumerate(line):
                    if c in ".G":
                        labyrinth[i, j] = 1
                    elif c == "X":
                        labyrinth[i, j] = 1
                        labyrinth.start = ((i, j))
                    elif c == "Y":
                        labyrinth[i, j] = 1
                        labyrinth.goal = ((i, j))
                    else:
                        labyrinth[i, j] = 0
                i += 1

    im = lab_to_im(labyrinth)
    return labyrinth, im


def lab_to_im(labyrinth):
    im = Image.new("RGB", (labyrinth.h, labyrinth.w))
    pix = im.load()
    for i in range(labyrinth.w):
        for j in range(labyrinth.h):
            v = labyrinth[i, j]
            pix[j, i] = (v * 255, v * 255, v * 255)
    start = labyrinth.start
    pix[start[1], start[0]] = (255, 0, 0)
    goal = labyrinth.goal
    pix[goal[1], goal[0]] = (0, 255, 0)
    return im


if __name__ == "__main__":
    imgpath = r"D:\labyrinth\lab4.bmp"
    #imgpath = r"D:\labyrinth\map\arena.map"
    print("Reading labyrinth from {}...".format(imgpath))
    labyrinth, _ = load_from_img(imgpath)
    print("Read")
    gen = NeighborsGeneratorPruning(labyrinth)
    for g in gen((2, 17), parent=(1, 16)):
        print(g)
