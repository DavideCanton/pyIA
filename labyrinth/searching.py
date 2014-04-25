import heapq as hq
from collections import namedtuple


Info = namedtuple("Info", "maxl nodes")


def a_star(start, goal, h, gen_children, callback=None):
    """
    A* algorithm, receives the start node, the goal node,
    the heuristic function h(n), a callable generating node
    successors and an optional callback.

    @param start: the start node
    @param goal: the goal node
    @param h: the heuristic function
    @param gen_children: the generation callable
    @param callback: a callback
    @return: the optimum path from start to goal
    """

    depth = {start: 0}
    parents = {}
    queue = [(h(start), 0, start)]
    visited = set()
    info = Info(maxl=0, nodes=0)

    while queue:
        if callback:
            callback(queue)

        info = info._replace(maxl=max(len(queue), info.maxl),
                             nodes=info.nodes + 1)

        current = hq.heappop(queue)[2]
        visited.add(current)

        if goal(current):
            node = current
            path = []
            while node in parents:
                path.append(node)
                node = parents[node]
            path.reverse()
            return path, visited, info

        parent = parents.get(current)

        for index, generated in enumerate(gen_children(current, parent)):
            successor, weight = generated
            successor_depth = depth[current] + weight
            if successor not in visited or successor_depth < depth[successor]:
                parents[successor] = current
                depth[successor] = successor_depth
                h_score = h(successor)
                f_score = depth[current] + h_score
                hq.heappush(queue, (f_score, h_score + index, successor))

    return None, None, info
