import sat, logic
import itertools as it


def buildFormulaIS(edges):
    return " ^ ".join("!{} V !{}".format(*e) for e in edges)


def buildFormulaVC(edges):
    return " ^ ".join("{} V {}".format(*e) for e in edges)


def buildFormulaEC(edges):
    nodes = set(it.chain(*edges))
    f = lambda n: sorted(e for e in edges if e[0] == n or e[1] == n)
    return " ^ ".join(" V ".join("".join(e) for e in f(n)) for n in nodes)


def buildFormulaClique(edges):
    nodes = set(it.chain(*edges))
    not_edges = ((a, b) for a, b in it.combinations(nodes, 2)
                 if (a, b) not in edges and (b, a) not in edges)
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
    print(formula.is_horn)

    iterations = 100000
    for _ in range(iterations):
        #if formula.is_horn:
        #    sol = formula.minimal_model
        #else:
        sol = sat.wsat_solve(formula)
        if len(sol) < 6:
            print(sol)
            exit()
    print("Nessuna soluzione trovata")
