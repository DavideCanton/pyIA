from functools import total_ordering
import itertools as it
from random import randint, sample
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
        return self._id

    __repr__ = lambda self: self.name
    __hash__ = lambda self: self._id
    __eq__ = lambda self, v: self.name == v.name
    __lt__ = lambda self, v: self.name < v.name


class Formula:
    "Class representing a formula"
    def __init__(self, vars, clauses=[]):
        "Build a formula from vars list and clause list"
        self.vars = set(vars)
        self.clauses = clauses

    def clause_eval(self, assignment):
        "Returns the number of clauses satisfied by assignment"
        return sum(c.satisfied(assignment) for c in self.clauses)

    def satisfied(self, assignment):
        "Returns true iff all clauses are satisfied by assignment"
        return all(c.satisfied(assignment) for c in self.clauses)

    def __str__(self):
        return " ^ ".join(map(str, self.clauses))

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
        self.pos_list = set(pos)
        self.neg_list = set(neg)

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


def _generateSubsets(vars):
    vars = sorted(vars)
    yield []
    for r in range(1, len(vars)):
        for subset in it.combinations(vars, r):
            yield list(subset)
    yield vars


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
    # returns the formula
    return Formula(vars=list(vars.values()), clauses=clauses)
