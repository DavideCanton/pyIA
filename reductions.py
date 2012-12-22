from pyIA import sat, logic
import itertools as it


def buildFormulaIS(edges):
    return " ^ ".join(["!{} V !{}".format(*e) for e in edges])


def buildFormulaVC(edges):
    return " ^ ".join(["{} V {}".format(*e) for e in edges])


def buildFormulaClique(edges):
    nodes = set(it.chain(*edges))
    not_edges = [(a, b) for a, b in it.combinations(nodes, 2)
                 if (a, b) not in edges and (b, a) not in edges]
    formulas = (["\xac{0} V {0}".format(n) for n in nodes] +
                ["\xac{} V \xac{}".format(*e) for e in not_edges])
    return " ^ ".join(formulas)


if __name__ == '__main__':
    edges = "ab|bc|bd|ae|be|ac|ce"
    edges = [tuple(e) for e in edges.split("|")]
    formulaS = buildFormulaIS(edges)
    formula = logic.build_formula(formulaS)
    clauses, vars = formula.clauses, formula.vars
    print(" ^ ".join(map(str, clauses)))

    iterations = 1000
    sol = []
    ok = lambda sol: len(sol) >= 2
    for _ in range(iterations):
        if ok(sol):
            print([var.name for var in sol])
            exit()
        sol = sat.wsat_solve(formula)
    print("Nessuna soluzione trovata")
