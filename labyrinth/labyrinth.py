from math import sqrt
import numpy as np
from collections import defaultdict
from PIL import Image

U, L, D, R, UL, UR, DL, DR = list(map(np.array, [(0, -1), (-1, 0), (0, 1),
                                                 (1, 0), (-1, -1), (1, -1),
                                                 (-1, 1), (1, 1)]))
SQRT_2 = sqrt(2)


def array_to_tuple(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return [(tuple(node), weight) for node, weight in ret]

    return wrapper


class NeighborsGenerator:
    """
    Generates moves towards the 8 directions.
    __call__ method yields a tuple (n,w), n is the successor
    node, w is the weight of the move. Uses the JPS
    (Jump Point Search) in order to make the successor generation
    smarter and pruning the search graph.
    A straight move costs 1 unit, while a diagonal one costs
    sqrt(2) units.
    """

    def __init__(self, labyrinth):
        self.labyrinth = labyrinth

    def up_free(self, x, y):
        return y > 0 and self.labyrinth[x, y - 1]

    def down_free(self, x, y):
        return y < self.labyrinth.h - 1 and self.labyrinth[x, y + 1]

    def left_free(self, x, y):
        return x > 0 and self.labyrinth[x - 1, y]

    def right_free(self, x, y):
        return x < self.labyrinth.w - 1 and self.labyrinth[x + 1, y]

    def natural_neighbors(self, pos):
        x, y = pos
        pos = np.array(pos)
        neighbors = []

        if self.up_free(x, y):
            neighbors.append((U + pos, 1))

        if self.down_free(x, y):
            neighbors.append((D + pos, 1))

        if self.left_free(x, y):
            neighbors.append((L + pos, 1))

        if self.right_free(x, y):
            neighbors.append((R + pos, 1))

        if (self.up_free(x, y) and self.left_free(x, y - 1) or
            self.left_free(x, y) and self.up_free(x - 1, y)):
            neighbors.append((pos + UL, SQRT_2))

        if (self.up_free(x, y) and self.right_free(x, y - 1) or
            self.right_free(x, y) and self.up_free(x + 1, y)):
            neighbors.append((pos + UR, SQRT_2))

        if (self.down_free(x, y) and self.left_free(x, y + 1) or
            self.left_free(x, y) and self.down_free(x - 1, y)):
            neighbors.append((pos + DL, SQRT_2))

        if (self.down_free(x, y) and self.right_free(x, y + 1) or
            self.right_free(x, y) and self.down_free(x + 1, y)):
            neighbors.append((pos + DR, SQRT_2))

        return neighbors

    @array_to_tuple
    def __call__(self, current, parent=None):
        """
        Generates the neighbors of current, if coming from parent.
        If parent is None, then current is the start node.
        """
        natural_neighbors = self.natural_neighbors(current)

        if parent is None:
            # I'm at the start node, all neighbors are OK
            return natural_neighbors

        else:
            #map tuples to numpy arrays
            current = np.array(current)
            natural_neighbors = [np.array(node[0])
                                 for node in natural_neighbors]
            parent = np.array(parent)
            move = normalize(current - parent)

            # pruning of neighbors
            if move.all():  # if move[0] != 0 and move[1] != 0 is diagonal
                natural_neighbors = self.prune_diag(natural_neighbors,
                                                    current, move)
            else:
                natural_neighbors = self.prune_straight(natural_neighbors,
                                                        current, move)

            #jumping
            real_neighbors = []
            for node in natural_neighbors:
                #print("Called jump from", current, "towards", node - current)
                jumped_to = self.jump_it_1(current, node - current,
                                           self.labyrinth.goal)
                #print("Returned", jumped_to)
                if jumped_to is not None:
                    distance = np.linalg.norm(jumped_to - current)
                    real_neighbors.append((jumped_to, distance))
            return real_neighbors

    def compute_forced_straight(self, node, move):
        """
        Compute forced straight neighbors. Example:
        | 1 | X | 3 |
        | 4 | O | 5 |
        | 6 | 7 | 8 |

        If going in O from 4, 5 is a natural neighbor, 3 is a forced neighbor
        """
        pruned = []
        for dir in orthogonal(move):
            next = node + dir
            if next in self.labyrinth and not self.labyrinth[next]:
                pruned.append(next + move)
        return pruned

    def compute_forced_diag(self, node, move):
        """
        Compute forced diagonal neighbors. Example:
        | 1 | 2 | 3 |
        | X | O | 5 |
        | 6 | 7 | 8 |

        If going in O from 6, [2,3,5] are
        natural neighbors, 1 is a forced neighbor
        """
        pruned = []
        for comp in components(move):
            next = node + comp
            if next in self.labyrinth and not self.labyrinth[next]:
                pruned.append(next + comp)
        return pruned

    def prune_straight(self, neighbors, node, move):
        """
        Returns the valid neighbors of node towards move, i.e.
        straight natural neighbors + straight forced neighbors
        """
        pruned_list = [node + move]
        pruned_list.extend(self.compute_forced_straight(node, move))
        return [node for node in pruned_list
                if any(np.array_equal(node, neighbor)
            for neighbor in neighbors)]

    def prune_diag(self, neighbors, node, move):
        """
        Returns the valid neighbors of node towards move, i.e.
        diagonal natural neighbors + diagonal forced neighbors
        """
        pruned_list = [node + dir for dir in components(move)]
        pruned_list.append(node + move)
        pruned_list.extend(self.compute_forced_diag(node - move, move))
        return [node for node in pruned_list
                if any(np.array_equal(node, neighbor)
            for neighbor in neighbors)]

    def jump_rec(self, current, direction, goal):
        """
        Recursive jump function
        """
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
            forced = self.compute_forced_diag(current, direction)
        else:
            forced = self.compute_forced_straight(next, direction)

        if any(self.labyrinth[f] for f in forced):
            return next

        if isDiag:
            for dirt in components(direction):
                if self.jump_rec(next, dirt, goal) is not None:
                    return next
        return self.jump_rec(next, direction, goal)

    def jump_it_1(self, current, direction, goal):
        """
        My iterative jump function
        """
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
                    forced = self.compute_forced_diag(el.current, el.direction)
                else:
                    forced = self.compute_forced_straight(next, el.direction)
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
                    snapshot = Snapshot(next, el.direction,
                                        el.goal, None, None, 0)
                    stack.append(snapshot)
                    continue
            elif el.stage == 1:
                r1 = retval
                if r1 is not None:
                    retval = el.next
                    continue
                el.stage = 2
                stack.append(el)
                snapshot = Snapshot(el.next, el.dirs[1],
                                    el.goal, el.next, el.dirs, 0)
                stack.append(snapshot)
                continue
            elif el.stage == 2:
                r2 = retval
                if r2 is not None:
                    retval = el.next
                    continue
                snapshot = Snapshot(el.next, el.direction,
                                    el.goal, None, None, 0)
                stack.append(snapshot)
                continue
        return retval

    def jump_it_2(self, current, direction, goal):
        """
        riko's iterative jump function
        """
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
                forced = self.compute_forced_diag(current, direction)
            else:
                forced = self.compute_forced_straight(next, direction)
            if any(self.labyrinth[f] for f in forced):
                return next

            if isDiag:
                stack.extend((next, di, goal)
                    for di in components(direction))
            else:
                stack.append((next, direction, goal))


class Snapshot:
    """
    This class models a snapshot of the local stack when jump_it_1 is called.
    """

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
    """
    Labyrinth class. Every position is stored as a binary number. lab[i,j] is
    0 if in the cell (i,j) there is an obstacle, else is 1.
    Implemented as a sparse matrix.
    """

    def __init__(self, w, h):
        self.labyrinth = defaultdict(int)
        self.w = w
        self.h = h
        self.start = None
        self.goal = None

    def __getitem__(self, item):
        return self.labyrinth[tuple(item)]

    def __contains__(self, pos):
        """
        Contains checks if pos is inside the labyrinth.
        """
        return 0 <= pos[0] < self.w and 0 <= pos[1] < self.h

    def __setitem__(self, key, value):
        self.labyrinth[tuple(key)] = value


def orthogonal(v):
    """
    Given a vector v parallel to a coordinate axis, computes
    the two vectors orthogonal to v. Example:
    >>> import numpy as np
    >>> orthogonal(np.array([1,0]))
    [array([0, 1]), array([ 0, -1])]
    """
    v = v.copy()
    v[[0, 1]] = v[[1, 0]]
    return [v, -v]


def components(v):
    """
    Given a diagonal vector v, returns the two components of v. Example:
    >>> import numpy as np
    >>> components(np.array([1,-1]))
    [array([1, 0]), array([ 0, -1])]
    """
    d1, d2 = v.copy(), v.copy()
    d1[1] = d2[0] = 0
    return [d1, d2]


def normalize(v):
    """
    Scales the vector v such that every element is 1,0,-1.
    >>> normalize([2,2])
    [1,1]
    >>> normalize([3,0])
    [1,0]
    """
    return v / abs(v[v != 0][0])


def load_from_img(imgpath):
    """
    Loads a labyrinth object from an image.
    A red pixel is interpreted as the start point, a green one as the goal.
    Black pixels represents obstacles, while white ones are free ground.
    """
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
    """
    Loads a labyrinth object from a map file.
    """
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
    """
    Converts a labyrinth to a PIL image (useful for the GUI).
    """
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
