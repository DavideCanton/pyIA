from math import sqrt
from typing import Tuple, Callable, List

import numpy as np
from PIL import Image

U = np.array([0, -1])
L = np.array([-1, 0])
D = -U
R = -L
UL, UR = U + L, U + R
DL, DR = D + L, D + R

SQRT_2 = sqrt(2)

Node = Tuple[int, int]


def manhattan(goal: Node, _: Node) -> Callable[[Node], float]:
    """
    Computes the Manhattan distance between the node and the goal.
    Example:

    | O |   |   |
    |   |   |   |
    |   |   | X |

    The distance is 4, because the shortest paths between
    (0,0) and (2,2) (like [(0,0)->(0,1)->(0,2)->(1,2)->(2,2)])
    have a cost of 4.
    """

    def wrapper(node):
        return (abs(node[0] - goal[0]) +
                abs(node[1] - goal[1])) * 1.001  # break ties

    return wrapper


def heur_diag(goal: Node, _: Node) -> Callable[[Node], float]:
    """
    Computes the minimum distance between the node
    and the goal considering diagonal moves.

    For example:

    | O |   |   |   |
    |   |   |   |   |
    |   |   |   | X |

    The distance is 2 * sqrt(2) + 1, because the shortest path between
    (0,0) and (2,3) (like [(0,0)->(1,1)->(2,2)->(2,3)]) has a cost of 4.
    """

    def wrapper(node):
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        m, n = dx, dy
        if m > n:
            m, n = n, m
        return (m * (SQRT_2 - 1) + n) * 1.001  # break ties

    return wrapper


def array_to_tuple(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return [(tuple(node), weight) for node, weight in ret]

    return wrapper


class Labyrinth:
    """
    Labyrinth class. Every position is stored as a binary number. lab[i,j] is
    0 if in the cell (i,j) there is an obstacle, else is 1.
    Implemented as a set.
    """

    def __init__(self, w: int, h: int, check=False) -> None:
        self.walls = set()
        self.w = w
        self.h = h
        self.start = None
        self.goal = None
        self.check = check

    def __getitem__(self, item: Node) -> int:
        if self.check and item not in self:
            raise ValueError("{} not contained in labyrinth".format(item))

        return int(tuple(item) not in self.walls)

    def __contains__(self, pos: Node) -> bool:
        """
        Contains checks if pos is inside the labyrinth.
        """
        return 0 <= pos[0] < self.w and 0 <= pos[1] < self.h

    def __setitem__(self, key: Node, value: bool | int) -> None:
        if self.check and key not in self:
            raise ValueError("{} not contained in labyrinth".format(key))

        key = tuple(key)
        if value:
            self.walls.discard(key)
        else:
            self.walls.add(key)

    def end_add(self) -> None:
        pass


class NeighborsGenerator:
    """
    Generates moves towards the 8 directions.
    __call__ method yields a tuple (n,w), n is the successor
    node, w is the weight of the move.
    A straight move costs 1 unit, while a diagonal one costs
    sqrt(2) units.
    """

    def __init__(self, labyrinth: Labyrinth, incr=1):
        """
        Creates the labyrinth.
        @param labyrinth: the labyrinth
        @type labyrinth: Labyrinth
        @param incr: the increment (default 1)
        @type incr: int
        """
        self.labyrinth = labyrinth
        self.incr = incr

    def up_free(self, x: int, y: int) -> bool:
        """
        Checks if up_move can be done
        @param x: the x coord
        @type x: int
        @param y: the y coord
        @type y: int
        @return: bool
        """
        return y > (self.incr - 1) and self.labyrinth[x, y - self.incr]

    def down_free(self, x: int, y: int) -> bool:
        """
        Checks if down_move can be done
        @param x: the x coord
        @type x: int
        @param y: the y coord
        @type y: int
        @return: bool
        """
        return (y < self.labyrinth.h - self.incr and
                self.labyrinth[x, y + self.incr])

    def left_free(self, x: int, y: int) -> bool:
        """
        Checks if left can be done
        @param x: the x coord
        @type x: int
        @param y: the y coord
        @type y: int
        @return: bool
        """
        return x > (self.incr - 1) and self.labyrinth[x - self.incr, y]

    def right_free(self, x: int, y: int) -> bool:
        """
        Checks if right_move can be done
        @param x: the x coord
        @type x: int
        @param y: the y coord
        @type y: int
        @return: bool
        """
        return (x < self.labyrinth.w - self.incr and
                self.labyrinth[x + self.incr, y])

    def natural_neighbors(self, pos: Node) -> List[Node]:
        """
        Computes the natural neighbors of pos.
        @param pos: the position.
        @type pos: tuple[int]
        @return: the list of natural neighbors of pos
        """
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

        p1 = self.up_free(x, y) and self.left_free(x, y - self.incr)
        p2 = self.left_free(x, y) and self.up_free(x - self.incr, y)
        if p1 or p2:
            neighbors.append((pos + UL * self.incr, SQRT_2 * self.incr))

        p1 = self.up_free(x, y) and self.right_free(x, y - self.incr)
        p2 = self.right_free(x, y) and self.up_free(x + self.incr, y)
        if p1 or p2:
            neighbors.append((pos + UR * self.incr, SQRT_2 * self.incr))

        p1 = self.down_free(x, y) and self.left_free(x, y + self.incr)
        p2 = self.left_free(x, y) and self.down_free(x - self.incr, y)
        if p1 or p2:
            neighbors.append((pos + DL * self.incr, SQRT_2 * self.incr))

        p1 = self.down_free(x, y) and self.right_free(x, y + self.incr)
        p2 = self.right_free(x, y) and self.down_free(x + self.incr, y)
        if p1 or p2:
            neighbors.append((pos + DR * self.incr, SQRT_2 * self.incr))

        return neighbors

    @array_to_tuple
    def __call__(self, current, parent=None):
        return self.natural_neighbors(current)


class NeighborsGeneratorJPS(NeighborsGenerator):
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
        """
        Creates the labyrinth.
        @param labyrinth: the labyrinth
        @type labyrinth: Labyrinth
        """
        super().__init__(labyrinth)

    @array_to_tuple
    def __call__(self, current, parent=None):
        """
        Generates the neighbors of current, if coming from parent.
        If parent is None, then current is the start node.

        @param current: the current position
        @type current: tuple[int]
        @param parent: the parent position
        @type parent: tuple[int] | None
        """
        natural_neighbors = self.natural_neighbors(current)

        if parent is None:
            # I'm at the start node, all neighbors are OK
            return natural_neighbors

        else:
            # map tuples to numpy arrays
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

            # jumping
            real_neighbors = []
            for node in natural_neighbors:
                # print("Called jump from", current, "towards", node - current)
                jumped_to = self.jump_it_1(current, node - current,
                                           self.labyrinth.goal)
                # print("Returned", jumped_to)
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

        @param node: the current node
        @type node: np.array
        @param move: the incoming move
        @type move: np.array
        """
        pruned = []
        for direction in orthogonal(move):
            next_node = node + direction
            if next_node in self.labyrinth and not self.labyrinth[next_node]:
                pruned.append(next_node + move)
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
            next_node = node + comp
            if next_node in self.labyrinth and not self.labyrinth[next_node]:
                pruned.append(next_node + comp)
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
        pruned_list = [node + dir_ for dir_ in components(move)]
        pruned_list.append(node + move)
        pruned_list.extend(self.compute_forced_diag(node - move, move))
        return [node for node in pruned_list
                if any(np.array_equal(node, neighbor)
                       for neighbor in neighbors)]

    def jump_rec(self, current, direction, goal):
        """
        Recursive jump function
        """
        next_node = current + direction
        if next_node not in self.labyrinth or not self.labyrinth[next_node]:
            return None
        if np.array_equal(next_node, goal):
            return next_node
        is_diag = direction.all()
        if is_diag:
            if all(not self.labyrinth[current + dirs]
                   for dirs in components(direction)):
                return None
            forced = self.compute_forced_diag(current, direction)
        else:
            forced = self.compute_forced_straight(next_node, direction)

        if any(self.labyrinth[f] for f in forced if f in self.labyrinth):
            return next_node

        if is_diag:
            for dirt in components(direction):
                if self.jump_rec(next_node, dirt, goal) is not None:
                    return next_node
        return self.jump_rec(next_node, direction, goal)

    def jump_it_1(self, current, direction, goal):
        """
        My iterative jump function
        """
        retval = None
        stack = [Snapshot(current, direction, goal, None, None, 0)]

        while stack:
            el = stack.pop()
            if el.stage == 0:
                next_node = el.current + el.direction
                if (next_node not in self.labyrinth or
                        not self.labyrinth[next_node]):
                    retval = None
                    continue
                if np.array_equal(next_node, el.goal):
                    retval = next_node
                    continue
                is_diag = el.direction.all()
                if is_diag:
                    if all(not self.labyrinth[el.current + dirs]
                           for dirs in components(direction)):
                        retval = None
                        continue
                    forced = self.compute_forced_diag(el.current, el.direction)
                else:
                    forced = self.compute_forced_straight(next_node,
                                                          el.direction)
                if any(self.labyrinth[f] for f in forced
                       if f in self.labyrinth):
                    retval = next_node
                    continue
                if is_diag:
                    el.stage = 1
                    el.next = next_node
                    stack.append(el)
                    dirs = list(components(direction))
                    el.dirs = dirs
                    snapshot = Snapshot(next_node, dirs[0],
                                        el.goal, next_node, dirs, 0)
                    stack.append(snapshot)
                    continue
                else:
                    snapshot = Snapshot(next_node, el.direction,
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
            next_node = current + direction
            if next_node not in self.labyrinth or not self.labyrinth[next_node]:
                return None
            if np.array_equal(next_node, goal):
                return next_node  # assuming n cannot be None
            is_diag = direction.all()
            if is_diag:
                if all(not self.labyrinth[current + dirs]
                       for dirs in components(direction)):
                    return None
                forced = self.compute_forced_diag(current, direction)
            else:
                forced = self.compute_forced_straight(next_node, direction)
            if any(self.labyrinth[f] for f in forced if f in self.labyrinth):
                return next_node

            if is_diag:
                stack.extend((next_node, di, goal)
                             for di in components(direction))
            else:
                stack.append((next_node, direction, goal))


class Snapshot:
    """
    This class models a snapshot of the local stack when jump_it_1 is called.
    """

    def __init__(self, current, direction, goal, next_, dirs, stage):
        self.current = current
        self.direction = direction
        self.goal = goal
        self.next = next_
        self.dirs = dirs
        self.stage = stage

    def __str__(self):
        return str(self.__dict__)


def orthogonal(v):
    """
    Given a vector v parallel to a coordinate axis, computes
    the two vectors orthogonal to v. Example:
    >>> import numpy as np
    >>> orthogonal(np.array([1,0]))
    [array([0, 1]), array([ 0, -1])]
    """
    v = v[::-1].copy()
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
    >>> import numpy as np
    >>> normalize(np.array([2,2]))
    [1,1]
    >>> normalize(np.array([3,0]))
    [1,0]
    """
    return v / abs(v[v != 0][0])


def load_from_map_file(filepath, img=True):
    """
    Loads a labyrinth object from a map file.
    """
    i, w, h = 0, 0, 0
    map_started = False
    with open(filepath) as map_file:
        for line in map_file:
            if line.startswith("height"):
                w = int(line.split()[1]) + 1
            elif line.startswith("width"):
                h = int(line.split()[1]) + 1
            elif line.startswith("map"):
                labyrinth = Labyrinth(w, h)
                map_started = True
            elif map_started:
                for j, c in enumerate(line):
                    if c in ".G":
                        labyrinth[j, i] = 1
                    elif c == "X":
                        labyrinth[j, i] = 1
                        labyrinth.start = j, i
                    elif c == "Y":
                        labyrinth[j, i] = 1
                        labyrinth.goal = j, i
                    else:
                        labyrinth[j, i] = 0
                i += 1

    im = lab_to_im(labyrinth) if img else None
    labyrinth.end_add()
    return labyrinth, im


def load_from_img(imgpath):
    """
    Loads a labyrinth object from an image.
    A red pixel is interpreted as the start point, a green one as the goal.
    Black pixels represents obstacles, while white ones are free ground.
    """
    im = Image.open(imgpath)
    pix = im.load()
    w, h = im.size
    labyrinth = Labyrinth(w, h)

    for x in range(w):
        for y in range(h):
            # avoid alpha
            match pix[x, y][:3]:
                case (255, 255, 255):
                    labyrinth[x, y] = 1
                case (255, 0, 0):
                    labyrinth[x, y] = 1
                    labyrinth.start = x, y
                case (0, 255, 0):
                    labyrinth[x, y] = 1
                    labyrinth.goal = x, y
                case _:
                    labyrinth[x, y] = 0

    labyrinth.end_add()
    return labyrinth, im


def lab_to_im(labyrinth):
    """
    Converts a labyrinth to a PIL image (useful for the GUI).
    """
    im = Image.new("RGB", (labyrinth.w, labyrinth.h))
    pix = im.load()
    for x in range(labyrinth.w):
        for y in range(labyrinth.h):
            v = labyrinth[x, y]
            pix[x, y] = (v * 255, v * 255, v * 255)
    if labyrinth.start:
        start = labyrinth.start
        pix[start[0], start[1]] = (255, 0, 0)
    if labyrinth.goal:
        goal = labyrinth.goal
        pix[goal[0], goal[1]] = (0, 255, 0)
    return im


def main():
    imgpath = r"img\lab4.bmp"
    # imgpath = r"D:\labyrinth\map\arena.map"
    print("Reading labyrinth from {}...".format(imgpath))
    labyrinth, _ = load_from_img(imgpath)
    print("Read")
    gen = NeighborsGeneratorJPS(labyrinth)
    for g in gen((2, 17), parent=(1, 16)):
        print(g)


if __name__ == '__main__':
    main()
