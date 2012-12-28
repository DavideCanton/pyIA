from pyIA import logic
from random import random, choice, randint, sample

__all__ = ["gsat_solve", "wsat_solve"]


def gsat_solve(formula, maxtries=1000):
    """Tries to solve a formula, resetting
    at most maxtries times using GSAT (greedy SAT) heuristic algorithm."""
    for _ in range(maxtries):
        solution = _tryToSolveG(formula)
        if solution is not None:
            return solution


def _tryToSolveG(formula):
    clauses, vars = formula.clauses, formula.vars
    # compute random initial assignment
    assignment = sample(vars, randint(0, len(vars)))
    cur = formula.clause_eval(assignment)
    while True:
        # if reached a model, returns it
        if cur == len(clauses):
            return assignment
        max_v = 0
        next_best = None
        # all possible value flips for variables
        for next in _nextAssignment(assignment, vars):
            # evaluates the new assignment
            v = formula.clause_eval(next)
            # if found, exit
            if v == len(clauses):
                return next
            # if better than current, store it
            if v > max_v:
                next_best, max_v = next, v
        # if max_v <= current value,
        # we've reached a local maximum (not good)
        # equal is used to avoid plateau (and thus loop)
        if max_v <= cur:
            return None
        assignment = next_best
        cur = max_v


# iterator flipping one variable per iteration
def _nextAssignment(assignment, vars):
    for v in vars:
        next_assignment = assignment[:]
        if v in assignment:
            next_assignment.remove(v)
        else:
            next_assignment.append(v)
        yield next_assignment


def wsat_solve(formula, maxtries=1000, p=.9):
    """Tries to solve a formula, resetting
    at most maxtries times with the WSAT (walking SAT) heuristic algorithm.
    The parameter p represent the probability of executing a GSAT iteration."""
    for _ in range(maxtries):
        solution = _tryToSolveW(formula, p)
        if solution is not None:
            return solution


def _tryToSolveW(formula, p):
    clauses, vars = formula.clauses, formula.vars
    # compute random initial assignment
    assignment = sample(vars, randint(0, len(vars)))
    cur = formula.clause_eval(assignment)
    while True:
        # reached a model for the formula, return it!
        if cur == len(clauses):
            return assignment
        if random() < p:
            # do GSAT move
            max_v = 0
            next_best = None
            # all possible value flips for variables
            for next in _nextAssignment(assignment, vars):
                # evaluates the new assignment
                v = formula.clause_eval(next)
                # if found, exit
                if v == len(clauses):
                    return next
                # if better than current, store it
                if v > max_v:
                    next_best, max_v = next, v
            # if max_v <= current value,
            # we've reached a local maximum (not good)
            # equal is used to avoid plateau (and thus loop)
            if max_v <= cur:
                return None
            assignment = next_best
            cur = max_v
        else:
            # compute unsatisfied clauses
            cl = choice([cl for cl in clauses
                            if not cl.satisfied(assignment)])
            # choose a random element
            var = choice(list(cl.var_list))
            # flips value
            if var in assignment:
                assignment.remove(var)
            else:
                assignment.append(var)
            # store new value
            cur = formula.clause_eval(assignment)


if __name__ == '__main__':
    maxtries = 5000
    formula = logic.randomFormula(nvar=3, nclauses=12)
    print(" ^ ".join(map(str, formula.clauses)))
    print("# vars:", len(formula.vars))
    print("# clauses:", len(formula.clauses))
    print("WSAT")
    wmodel = wsat_solve(formula)
    if wmodel is not None:
        wmodel = set(wmodel)
    print(wmodel)
    print("GSAT")
    gmodel = gsat_solve(formula)
    if gmodel is not None:
        gmodel = set(gmodel)
    print(gmodel)
    print("Horn formula?", ('yes' if formula.is_horn else 'no'))
    mmodel = formula.minimal_model
    print("Minimal model:", mmodel)
