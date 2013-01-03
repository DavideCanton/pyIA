import heapq as hq
from collections import deque
from functools import total_ordering

__all__ = ["Info", "a_star", "best_first", "breadth_first", "depth_first",
           "generic_search", "hill_climbing", "ida_star",
           "iterative_broadening", "iterative_deepening"]

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


def a_star(start, goal, h, gen_children, callback=None):
    n = Node(start)
    n.value = h(start)
    L = [n]
    visited = set()
    info = Info()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        current = hq.heappop(L)

        if goal(current.content):
            n = current
            pc = []
            while n:
                pc.append(n.content)
                n = n.parent
            pc.reverse()
            return pc, visited, info

        if current.content in visited:
            continue
        visited.add(current.content)

        for child in gen_children(current.content):
            if child in visited:
                continue
            x = Node(child, current)
            x.depth = current.depth + 1
            x.value = h(child) + x.depth
            hq.heappush(L, x)

    return None, None, info


def best_first(start, goal, h, gen_children, callback=None):
    n = Node(start)
    n.value = h(start)
    L = [n]
    visited = set()
    info = Info()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        current = hq.heappop(L)

        if goal(current.content):
            n = current
            pc = []
            while n:
                pc.append(n.content)
                n = n.parent
            pc.reverse()
            return pc, visited, info

        if current.content in visited:
            continue
        visited.add(current.content)

        for child in gen_children(current.content):
            if child in visited:
                continue
            x = Node(child, current)
            x.depth = current.depth + 1
            x.value = h(child)
            hq.heappush(L, x)

    return None, None, info


def hill_climbing(start, goal, h, gen_children, callback=None):
    n = Node(start)
    n.value = h(start)
    L = [n]
    visited = set()
    info = Info()

    while L:
        if callback:
            callback(L)

        info.maxl = max(len(L), info.maxl)
        info.nodes += 1

        current = L.pop()

        if goal(current.content):
            n = current
            pc = []
            while n:
                pc.append(n.content)
                n = n.parent
            pc.reverse()
            return pc, visited, info

        if current.content in visited:
            continue
        visited.add(current.content)

        sons = []
        for child in gen_children(current.content):
            if child in visited:
                continue
            x = Node(child, current)
            x.depth = current.depth + 1
            x.value = h(child)
            sons.append(x)
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


class Reversed:
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, *a, **kw):
        for el in reversed(list(self.gen(*a, **kw))):
            yield el


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

    while True:
        n = Node(start)
        n.value = h(start)
        L = [n]
        cp = INF
        visited = {}

        if callback:
            callback(L, reset=True)

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
                return pc, set(visited.keys()), info

            visited[current.content] = current.value

            for child in gen_children(current.content):

                x = Node(child, current)
                x.depth = current.depth + 1
                x.value = h(child) + x.depth

                if child in visited:
                    if x.value < visited[child]:
                        visited[child] = x.value
                    else:
                        continue
                else:
                    visited[child] = x.value

                if x.value <= c:
                    L.append(x)
                else:
                    cp = x.value if cp == INF else min(cp, x.value)

        if cp == INF:
            break

        c = cp

    return None, None, info


if __name__ == '__main__':
    import dis
    dis.dis(a_star)
