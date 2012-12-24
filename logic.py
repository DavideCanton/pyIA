from functools import total_ordering, reduce
from operator import and_
import itertools as it
from random import randint, choice, sample, random
import re
from collections import defaultdict

__all__ = ["Clause", "Formula", "Variable", "build_formula",
           "randomFormula", "randomHornFormula"]


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
    def imply_form(self):
        return " ^ ".join(cl.imply_form for cl in self.clauses)

    @property
    def is_horn(self):
        return all(c.is_horn for c in self.clauses)

    @property
    def minimal_model(self):
        if self.is_horn:
            return self._hornsat()
        else:
            return reduce(and_, map(set, self.models), set(self.vars))

    def _hornsat(self):
        var_link = defaultdict(list)
        for i, cl in enumerate(self.clauses):
            for var in cl.var_list:
                var_link[var].append(i)
        model = set()
        f = self.copy()
        cl_list = f.clauses
        while True:
            unit_clause = self._find_unit_clause(cl_list)
            if not unit_clause:
                break
            l, *_ = unit_clause.pos_list
            model.add(l)
            for i in var_link[l][:]:
                cl = cl_list[i]
                if cl and l in cl.pos_list:
                    var_link[l].remove(i)
                    cl_list[i] = None
                elif cl and l in cl.neg_list:
                    if not cl.pos_list and len(cl.neg_list) == 1:
                        return set()
                    cl.neg_list.remove(l)
        return model

    def _find_unit_clause(self, clauses):
        for cl in clauses:
            if cl and len(cl.pos_list) == 1 and not cl.neg_list:
                return cl
        return None

    @property
    def models(self):
        "Generator yielding all models for the formula"
        for assignment in _generateSubsets(self.vars):
            if self.satisfied(assignment):
                yield assignment

    def copy(self):
        cl = [c.copy() for c in self.clauses]
        return Formula(vars=self.vars, clauses=cl)


class Clause:
    "Class representing a disjunction of literals, a.k.a. clause"
    # separator of clauses
    or_sep = " V "
    and_sep = " ^ "
    imply = " -> "

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

    @property
    def is_horn(self):
        return len(self.pos_list) <= 1

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
        sp = Clause.or_sep.join(var.name
                                for var in self.pos_list)
        sn = Clause.or_sep.join("\xac" + var.name
                                for var in self.neg_list)
        if sp and sn:
            # join the two pieces
            s = Clause.or_sep.join((sp, sn))
        else:
            # choose the one not empty
            s = sp or sn
        return "(" + s + ")"

    @property
    def imply_form(self):
        head = Clause.or_sep.join(var.name for var in self.pos_list)
        tail = Clause.and_sep.join(var.name for var in self.neg_list)
        return "(" + Clause.imply.join((tail, head)) + ")"

    def __eq__(self, cl):
        return (self.pos_list == cl.pos_list and
                self.neg_list == cl.neg_list)

    def __hash__(self):
        return 13 * hash(self.pos_list) + 17 * hash(self.neg_list)

    def copy(self):
        return Clause(pos=set(self.pos_list), neg=set(self.neg_list))


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
    cl = []
    while len(cl) != nclauses:
        try:
            vars_c = sample(vars, randint(0, nvar))
            neg = sample(vars, randint(0, nvar))
            cl.append(Clause(vars_c, neg))
        except ValueError:
            pass
    return Formula(vars=vars, clauses=cl)


def randomHornFormula(nvar=3, nclauses=5):
    """Creates a random Horn formula with nvar variables and
    a number of clauses uniformally distributed in [1, nclauses]"""
    nclauses = randint(1, nclauses + 1)
    vars = [Variable() for _ in range(nvar)]
    cl = []
    while len(cl) != nclauses:
        try:
            if random() < 0.5:
                vars_c = [choice(vars)]
            else:
                vars_c = []
            neg = sample(vars, randint(0, nvar))
            cl.append(Clause(vars_c, neg))
        except ValueError:
            pass
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

if __name__ == '__main__':
    f = randomFormula(30, 100)
    print(f)
    print(f.imply_form)
    print(f.is_horn)
    print(f.minimal_model)
    print(list(f.models))
