import itertools as it

import logic
import sat


def build_formula_independent_set(edges):
    return " ^ ".join("!{} V !{}".format(*e) for e in edges)


def build_formula_vertex_cover(edges):
    return " ^ ".join("{} V {}".format(*e) for e in edges)


def build_formula_edge_cover(edges):
    nodes = set(it.chain(*edges))
    f = lambda n: sorted(e for e in edges if e[0] == n or e[1] == n)
    return " ^ ".join(" V ".join("".join(e) for e in f(n)) for n in nodes)


def build_formula_clique(edges):
    nodes = set(it.chain(*edges))
    not_edges = ((a, b) for a, b in it.combinations(nodes, 2)
                 if (a, b) not in edges and (b, a) not in edges)
    formulas = ([f"\xac{n} V {n}" for n in nodes] +
                ["\xac{} V \xac{}".format(*e) for e in not_edges])
    return " ^ ".join(formulas)


def main():
    edges = "ab|bc|bd|ae|be|ac|ce"
    edges = [tuple(e) for e in edges.split("|")]
    formula_s = build_formula_vertex_cover(edges)
    formula = logic.Parser().build_formula(formula_s)
    clauses = formula.clauses
    print(" ^ ".join(map(str, clauses)))
    print(formula.is_horn)

    iterations = 100000
    for _ in range(iterations):
        # if formula.is_horn:
        #    sol = formula.minimal_model
        # else:
        sol = sat.wsat_solve(formula)
        if len(sol) < 6:
            print(sol)
            exit()
    print("Nessuna soluzione trovata")


if __name__ == '__main__':
    main()
