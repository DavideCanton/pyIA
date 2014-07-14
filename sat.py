import logic
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
    clauses, vars_ = formula.clauses, formula.vars
    # compute random initial assignment
    assignment = sample(vars_, randint(0, len(vars_)))
    cur = formula.clause_eval(assignment)
    while True:
        # if reached a model, returns it
        if cur == len(clauses):
            return assignment
        max_v = 0
        next_best = None
        # all possible value flips for variables
        for next_ in _nextAssignment(assignment, vars_):
            # evaluates the new assignment
            v = formula.clause_eval(next_)
            # if found, exit
            if v == len(clauses):
                return next_
            # if better than current, store it
            if v > max_v:
                next_best, max_v = next_, v
        # if max_v <= current value,
        # we've reached a local maximum (not good)
        # equal is used to avoid plateau (and thus loop)
        if max_v <= cur:
            return None
        assignment = next_best
        cur = max_v


# iterator flipping one variable per iteration
def _nextAssignment(assignment, vars_):
    for v in vars_:
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
    clauses, vars_ = formula.clauses, formula.vars
    # compute random initial assignment
    assignment = sample(vars_, randint(0, len(vars_)))
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
            for next_ in _nextAssignment(assignment, vars_):
                # evaluates the new assignment
                v = formula.clause_eval(next_)
                # if found, exit
                if v == len(clauses):
                    return next_
                # if better than current, store it
                if v > max_v:
                    next_best, max_v = next_, v
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
    formula = logic.randomHornFormula(nvar=3, nclauses=5)
    print(formula)
    print(formula.imply_form)
    print("# vars:", len(formula.vars))
    print("# clauses:", len(formula.clauses))
    print("WSAT")
    wmodel = wsat_solve(formula)
    if wmodel is not None:
        wmodel = set(wmodel)
        print(wmodel)
        print(formula.satisfied(wmodel))
    print("GSAT")
    gmodel = gsat_solve(formula)
    if gmodel is not None:
        gmodel = set(gmodel)
        print(gmodel)
        print(formula.satisfied(gmodel))
    print("Horn formula?", ('yes' if formula.is_horn else 'no'))
    mmodel = formula.minimal_model
    print("Minimal model:", mmodel)
    if wmodel is not None:
        print(mmodel <= wmodel)
    if gmodel is not None:
        print(mmodel <= gmodel)
