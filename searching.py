import heapq as hq
import numpy as np
from collections import deque
from functools import total_ordering
from math import exp
from random import random, choice

__all__ = ["Info", "a_star", "best_first", "breadth_first", "depth_first",
           "generic_search", "hill_climbing", "ida_star",
           "iterative_broadening", "iterative_deepening", "simulated_annealing"]

# infinite represented by -1
INF = -1


@total_ordering
class Node:
    def __init__(self, content, parent=None, value=0, depth=0):
        self.content = content
        self.value = value
        self.depth = depth
        self.parent = parent

    def __eq__(self, val):
        return self.value == val.value

    def __lt__(self, val):
        return self.value < val.value

    def __repr__(self):
        return ("<Node, cnt: {}, val: {}, dpt: {}>"
                .format(self.content, self.value, self.depth))


class Info:
    def __init__(self, maxl=0, nodes=0):
        self.maxl = maxl
        self.nodes = nodes

    def __repr__(self):
        return ("Info[Max Length: {}, Nodes processed: {}]"
                .format(self.maxl, self.nodes))


def simulated_annealing(start, goal, h, gen_children, schedule, callback=None):
    best = None
    n = Node(start)
    n.value = h(start)
    L = [n]
    info = Info()
    visited = set()
    t = 0
    current = n

    while True:
        if callback:
            callback(L)

        if best is None or current.value > best.value:
            best = current

        T = schedule(t)
        t += 1

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        if abs(T) < 1E-10 or goal(current.content):
            n = current
            pc = []
            while n:
                pc.append(n.content)
                n = n.parent
            pc.reverse()
            return pc, None, info

        sons = list(gen_children(current.content))
        nc = choice(sons)
        #while nc in visited:
        #    nc = choice(sons)
        next = Node(nc)
        next.value = h(nc)
        next.depth = current.depth + 1
        next.parent = current
        delta = float(current.value - next.value)
        if delta > 0 or random() < exp(delta / T):
            visited.add(current.content)
            current = next
            L.append(next)
        elif delta < 0 and random() < 0.7:
            current = best
            t = 0

    return None, None, info


def a_star(start, goal, h, gen_children, callback=None):
    g = {start: 0}
    c_from = {}
    L = [(h(start), 0, start)]

    visited = set()
    info = Info()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        current = hq.heappop(L)[2]
        visited.add(current)

        if goal(current):
            n = current
            pc = []
            while n in c_from:
                pc.append(n)
                n = c_from[n]
            pc.reverse()
            return pc, visited, info

        for i, child in enumerate(gen_children(current, c_from.get(current))):
            child, w = child
            tg = g[current] + w
            if child in visited:
                if tg >= g[child]:
                    continue
            if child not in g or tg < g[child]:
                c_from[child] = current
                g[child] = tg
                vh = h(child)
                f = g[current] + vh
                hq.heappush(L, (f, vh + i, child))

    return None, None, info


def best_first(start, goal, h, gen_children, callback=None):
    c_from = {}
    L = [(h(start), start)]
    visited = set()
    info = Info()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        _, current = hq.heappop(L)
        visited.add(current)

        if goal(current):
            n = current
            pc = []
            while n in c_from:
                pc.append(n)
                n = c_from[n]
            pc.reverse()
            return pc, visited, info

        for i, child in enumerate(gen_children(current, c_from.get(current))):
            child, w = child
            if child in visited:
                continue
            c_from[child] = current
            vh = h(child)
            hq.heappush(L, (vh, child))

    return None, None, info


def hill_climbing(start, goal, h, gen_children, callback=None):
    c_from = {}
    L = [(h(start), start)]
    visited = set()
    info = Info()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        _, current = hq.heappop(L)
        visited.add(current)

        if goal(current):
            n = current
            pc = []
            while n in c_from:
                pc.append(n)
                n = c_from[n]
            pc.reverse()
            return pc, visited, info

        sons = []
        for i, child in enumerate(gen_children(current, c_from.get(current))):
            child, w = child
            if child in visited:
                continue
            c_from[child] = current
            vh = h(child)
            sons.append((vh, child))
        sons.sort(reverse=True)
        L.extend(sons)

    return None, None, info


def generic_search(start, is_goal, gen_children,
                   L, get_func, put_func, callback=None):
    put_func(L, Node(start))
    info = Info()
    visited = set()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        current = get_func(L)

        if is_goal(current.content):
            n = current
            pc = []
            while n:
                pc.append(n.content)
                n = n.parent
            pc.reverse()
            return pc, info

        visited.add(current.content)

        for child in gen_children(current.content):
            if child in visited:
                continue
            x = Node(child, current)
            x.depth = current.depth + 1
            put_func(L, x)

    return None, info


def depth_first(start, is_goal, gen_children, left=False, callback=None):
    if left:
        gen_children = Reversed(gen_children)
    return generic_search(start, is_goal, gen_children,
                          deque(), deque.popleft, deque.appendleft, callback)


def breadth_first(start, is_goal, gen_children, callback=None):
    return generic_search(start, is_goal, gen_children,
                          deque(), deque.popleft, deque.append, callback)


def iterative_deepening(start, is_goal, gen_children, callback=None):
    c = 0
    info = Info()

    while True:
        L = [Node(start)]
        visited = set()
        maxdepth = 0

        while L:
            if callback:
                callback(L)

            info.maxl = max(len(L), info.maxl)
            info.nodes += 1

            current = L.pop()
            maxdepth = max(maxdepth, current.depth)

            if is_goal(current.content):
                n = current
                pc = []
                while n:
                    pc.append(n.content)
                    n = n.parent
                pc.reverse()
                return pc, info

            visited.add(current.content)

            if current.depth < c:
                for child in gen_children(current.content):
                    if child in visited:
                        continue
                    x = Node(child, current)
                    x.depth = current.depth + 1
                    L.append(x)

        if maxdepth < c:
            break
        c += 1

    return None, info


def iterative_broadening(start, is_goal, gen_children, bmax, callback=None):
    info = Info()

    for c in range(1, bmax + 1):
        L = [Node(start)]
        visited = set()

        while L:
            if callback:
                callback(L)

            info.maxl = max(len(L), info.maxl)
            info.nodes += 1

            current = L.pop()

            if is_goal(current.content):
                n = current
                pc = []
                while n:
                    pc.append(n.content)
                    n = n.parent
                pc.reverse()
                return pc, info

            visited.add(current.content)

            for i, child in enumerate(gen_children(current.content)):
                if i == c:
                    break
                if child in visited:
                    continue
                x = Node(child, current)
                x.depth = current.depth + 1
                L.append(x)

    return None, info


def ida_star(start, is_goal, h, gen_children, callback=None):
    c = 1
    info = Info()
    c_from = {}
    g = {start: 0}

    while True:
        n = h(start), start
        L = [n]
        cp = INF
        visited = {}

        if callback:
            callback(L)

        while L:
            if callback:
                callback(L)
            info.maxl = max(len(L), info.maxl)
            info.nodes += 1

            val, current = L.pop()

            if is_goal(current):
                n = current
                pc = []
                while n in c_from:
                    pc.append(n)
                    n = c_from[n]
                pc.reverse()
                return pc, visited, info

            visited[current] = val

            for child in gen_children(current, c_from.get(current)):
                child, w = child
                g[child] = g[current] + w
                x = h(child) + g[child], child

                if child in visited:
                    if x[0] < visited[child]:
                        visited[child] = x[0]
                    else:
                        continue
                else:
                    visited[child] = x[0]

                if x[0] <= c:
                    L.append(x)
                else:
                    cp = x[0] if cp == INF else min(cp, x[0])

        if cp == INF:
            break

        c = cp

    return None, None, info
