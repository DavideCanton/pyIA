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
    formulaS = buildFormulaVC(edges)
    formula = logic.Parser().build_formula(formulaS)
    clauses, vars = formula.clauses, formula.vars
    print(" ^ ".join(map(str, clauses)))
    print(" ^ ".join(cl.imply_form for cl in clauses))
    print(formula.is_horn)

    iterations = 1000
    sol = []
    ok = lambda sol: True
    for _ in range(iterations):
        if formula.is_horn:
            sol = formula.minimal_model
        else:
            sol = sat.wsat_solve(formula)
        if ok(sol):
            print([var.name for var in sol])
            exit()
    print("Nessuna soluzione trovata")
