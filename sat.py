from random import randint, sample, random, choice
from functools import total_ordering
import itertools as it
import re


@total_ordering
class Variable:
    "Class representing a variable"
    # private id of every variable
    _id = 0

    def __init__(self, name=None):
        """Creates a variable named name, if name is not None,
        else names it "v" + str(id)"""
        self._id = Variable._id
        self.name = name if name is not None else "v{}".format(self._id)
        Variable._id += 1

    @property
    def id(self):
        return self.id

    __repr__ = lambda self: self.name
    __hash__ = lambda self: self._id
    __eq__ = lambda self, v: self.name == v.name
    __lt__ = lambda self, v: self.name < v.name


class Formula:
    "Class representing a formula"
    def __init__(self, vars, clauses=[]):
        "Build a formula from vars list and clause list"
        self.vars = frozenset(vars)
        self.clauses = clauses

    def clause_eval(self, assignment):
        "Returns the number of clauses satisfied by assignment"
        return sum(c.satisfied(assignment) for c in self.clauses)

    def satisfied(self, assignment):
        "Returns true iff all clauses are satisfied by assignment"
        return all(c.satisfied(assignment) for c in self.clauses)

    @property
    def models(self):
        "Generator yielding all models for the formula"
        for assignment in _generateSubsets(self.vars):
            if self.satisfied(assignment):
                yield assignment


class Clause:
    "Class representing a disjunction of literals, a.k.a. clause"
    # separator of clauses
    or_sep = "V"

    def __init__(self, pos, neg):
        """Creates a disjunction with every variable in pos occurring
        positively and every variable in neg occurring negatively"""
        if not pos and not neg:
            raise ValueError("Empty pos and neg")
        self.pos_list = frozenset(pos)
        self.neg_list = frozenset(neg)

    @property
    def var_list(self):
        "Returns the set of the variables occurring in the clause"
        return self.pos_list | self.neg_list

    def satisfied(self, assignment):
        """Receives an interpretation in the form of a sequence of true
        variables and returns true iff it is a model for the clause"""
        for pos in self.pos_list:
            if pos in assignment:
                return True
        for neg in self.neg_list:
            if neg not in assignment:
                return True
        return False

    def __str__(self):
        o = " " + Clause.or_sep + " "
        sp = o.join("{}".format(var.name) for var in self.pos_list)
        sn = o.join("\xac{}".format(var.name) for var in self.neg_list)
        if sp and sn:
            # join the two pieces
            s = sp + o + sn
        else:
            # choose the one not empty
            s = sp or sn
        return "(" + s + ")"

    def __eq__(self, cl):
        return (self.pos_list == cl.pos_list and
                self.neg_list == cl.neg_list)

    def __hash__(self):
        return 13 * hash(self.var_list) + 17 * hash(self.neg)


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


def randomFormula(nvar=3, nclauses=5):
    """Creates a random formula with nvar variables and
    a number of clauses uniformally distributed in [1, nclauses]"""
    nclauses = randint(1, nclauses + 1)
    vars = [Variable() for _ in range(nvar)]
    cl = set()
    for _ in range(nclauses):
        vars_c = sample(vars, randint(1, nvar))
        neg = sample(vars_c, randint(0, len(vars_c)))
        cl.add(Clause(vars_c, neg))
    return Formula(vars=vars, clauses=cl)


def build_formula(expression, or_expr="[ V|]",
                  and_expr="[\^&]", not_expr="[!\xac]"):
    """Build a sequence of clauses from the expression given, using
    the regex and_expr for dividing clauses, or_expr for getting
    the literals and not_expr for not symbols,
    returns the formula. Ignores parenthesis in expression."""
    # removes parenthesis
    expression = re.sub("[)(]", "", expression)
    # variables encountered during processing (String -> Variable)
    vars = {}
    # clauses built
    clauses = []
    # positive variables of the current building clauses
    pos = []
    # negative variables of the current building clauses
    negs = []
    # splits tokens (dividing literals but leaving and symbols)
    for token in filter(None, re.split(or_expr, expression)):
        token = token.strip()
        # if it's an and_token builds the clause
        if re.match(and_expr, token):
            clauses.append(Clause(pos, negs))
            pos[:] = []
            negs[:] = []
        else:
            # check if it's negative
            neg = False
            if re.match(not_expr, token[0]):
                token = token[1:]
                neg = True
            # update dict
            if token in vars:
                var = vars[token]
            else:
                var = Variable(token)
                vars[token] = var
            # appends the variable in the right list
            (negs if neg else pos).append(var)
    clauses.append(Clause(pos, negs))
    # returns the vars and the clauses
    return Formula(vars=list(vars.values()), clauses=clauses)


def _generateSubsets(vars):
    vars = sorted(vars)
    yield []
    for r in range(1, len(vars)):
        for subset in it.combinations(vars, r):
            yield list(subset)
    yield vars


def _solveAndFormat(solver, formula, maxtries=1000):
    vars = formula.vars
    sol = solver(formula, maxtries=maxtries)
    if sol is None:
        return None
    else:
        not_ch = lambda x: "\xac" if x not in sol else ""
        return (" ^ ".join("{}{}".format(not_ch(var), var.name)
                           for var in vars))


if __name__ == '__main__':
    maxtries = 5000
    formula = build_formula("a ^ b V a ^ c V !b")
    print(list(formula.models))
    # vars, cl = randomFormula(nvar=3, nclauses=12)
    print(" ^ ".join(map(str, formula.clauses)))
    print("# vars:", len(formula.vars))
    print("# clauses:", len(formula.clauses))
    print("WSAT")
    print(_solveAndFormat(wsat_solve, formula, maxtries))
    print("GSAT")
    print(_solveAndFormat(gsat_solve, formula, maxtries))
