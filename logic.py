import itertools as it
import re
import string
from collections import defaultdict, deque
from functools import total_ordering
from random import randint, choice, sample, random

__all__ = ["Clause", "Formula", "Variable", "Parser",
           "random_formula", "random_horn_formula"]


@total_ordering
class Variable:
    """Class representing a variable"""
    # private id of every variable
    _id = 0

    def __init__(self, name=None):
        """Creates a variable named name, if name is not None,
        else names it "v" + str(id)"""
        self._id = Variable._id
        self.name = name or "v{}".format(self._id)
        Variable._id += 1

    @property
    def id(self):
        return self._id

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, v):
        return self.name == v.name

    def __lt__(self, v):
        return self.name < v.name


class Formula:
    """Class representing a formula"""

    def __init__(self, variables, clauses=None, heads=None):
        """Build a formula from variables list and clause list"""
        self.variables = set(variables)
        self.clauses = clauses if clauses is not None else []
        if heads is not None:
            self.set_heads(heads)
        else:
            self.heads = []

    def set_heads(self, heads):
        if len(heads) != len(self.clauses):
            raise ValueError("Lunghezze non compatibili")
        self.heads = []
        for h in heads:
            ll = []
            for s in h:
                if hasattr(s, "name"):
                    ll.append(s)
                else:
                    for v in self.variables:
                        if v.name == s:
                            ll.append(v)
            self.heads.append(set(ll))  # TODO

    def clause_eval(self, assignment):
        """Returns the number of clauses satisfied by assignment"""
        return sum(c.satisfied(assignment) for c in self.clauses)

    def satisfied(self, assignment):
        """Returns true iff all clauses are satisfied by assignment"""
        return all(c.satisfied(assignment) for c in self.clauses)

    def __str__(self):
        return " ^ ".join("(" + str(cl) + ")" for cl in self.clauses)

    def is_stable_model(self, model):
        if self.is_horn:
            return model == self.minimal_model
        else:
            reduct = self._reduce_to(model)
            assert reduct.is_horn
            return model == reduct.minimal_model

    @property
    def stable_models(self):
        for model in self.models:
            if self.is_stable_model(model):
                yield model

    def _reduce_to(self, model):
        ll = []
        for cl, h in zip(self.clauses, self.heads):
            cop = cl.copy()
            to_remove = False
            for v in cl.pos_list:
                if v in h:
                    continue
                if v not in model:
                    cop.pos_list.remove(v)
                else:
                    to_remove = True
                    break
            if not to_remove:
                ll.append(cop)
        return Formula(self.variables, ll)

    @property
    def imply_form(self):
        return (" ^ ".join("(" + cl.imply_form(h) + ")"
                           for cl, h in zip(self.clauses, self.heads)))

    @property
    def is_horn(self):
        return all(c.is_horn for c in self.clauses)

    @property
    def minimal_model(self):
        if self.is_horn:
            return self._hornsat()
        else:
            raise ValueError("Formula non-Horn")

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
                        return None
                    cl.neg_list.remove(l)
        return model

    def _find_unit_clause(self, clauses):
        for cl in clauses:
            if cl and len(cl.pos_list) == 1 and not cl.neg_list:
                return cl
        return None

    @property
    def models(self):
        """Generator yielding all models for the formula"""
        for assignment in _generate_subsets(self.variables):
            if self.satisfied(assignment):
                yield set(assignment)

    def copy(self):
        cl = [c.copy() for c in self.clauses]
        return Formula(variables=self.variables, clauses=cl)


class Clause:
    """Class representing a disjunction of literals, a.k.a. clause"""
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
        """Returns the set of the variables occurring in the clause"""
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
        sp = Clause.or_sep.join(var.name for var in self.pos_list)
        sn = Clause.or_sep.join("\xac" + var.name for var in self.neg_list)
        if sp and sn:
            # join the two pieces
            s = Clause.or_sep.join((sp, sn))
        else:
            # choose the one not empty
            s = sp or sn
        return s

    def to_rule(self, head):
        bn = self.pos_list - head
        return Rule(head=head, body_pos=self.neg_list, body_neg=bn)

    def imply_form(self, head):
        return str(self.to_rule(head))

    def __eq__(self, cl):
        return (self.pos_list == cl.pos_list and
                self.neg_list == cl.neg_list)

    def __hash__(self):
        return 13 * hash(self.pos_list) + 17 * hash(self.neg_list)

    def copy(self):
        return Clause(pos=set(self.pos_list), neg=set(self.neg_list))


class ImplyFormula:
    """Class representing a formula"""

    def __init__(self, variables, rules=None):
        """Build a formula from variables list and rules list"""
        self.variables = set(variables)
        self.rules = rules if rules is not None else []

    def clause_eval(self, assignment):
        """Returns the number of clauses satisfied by assignment"""
        return sum(c.satisfied(assignment) for c in self.rules)

    def satisfied(self, assignment):
        """Returns true iff all clauses are satisfied by assignment"""
        return all(c.satisfied(assignment) for c in self.rules)

    def __str__(self):
        return "\n".join(str(cl) for cl in self.rules)

    def is_stable_model(self, model):
        if self.is_horn:
            # TODO
            return model in self.minimal_models
        else:
            reduct = self._reduce_to(model)
            return model == reduct.minimal_models

    @property
    def stable_models(self):
        for model in self.models:
            if self.is_stable_model(model):
                yield model

    def _reduce_to(self, model):
        ll = []
        for cl in self.rules:
            cop = cl.copy()
            to_remove = False
            for v in cl.body_neg:
                if v not in model:
                    to_remove = True
                    break
                else:
                    cop.body_neg.remove(v)
            if not to_remove:
                ll.append(cop)
        return ImplyFormula(self.variables, ll)

    @property
    def disj_form(self):
        return " ^ ".join("(" + cl.disj_form + ")" for cl in self.rules)

    @property
    def is_horn(self):
        return all(c.is_horn for c in self.rules)

    @property
    def minimal_models(self):
        if self.is_horn:
            yield self._hornsat()
        else:
            ll = []
            for s in _generate_subsets(self.variables):
                s = set(s)
                if self.satisfied(s) and not any(map(lambda x: x <= s, ll)):
                    yield s
                    ll.append(s)

    def _hornsat(self):
        var_link = defaultdict(list)
        for i, cl in enumerate(self.rules):
            for var in cl.var_list:
                var_link[var].append(i)
        model = set()
        f = self.copy()
        cl_list = f.rules
        while True:
            unit_clause = self._find_unit_clause(cl_list)
            if not unit_clause:
                break
            l, *_ = unit_clause.head
            model.add(l)
            for i in var_link[l][:]:
                cl = cl_list[i]
                if cl and l in cl.head:
                    var_link[l].remove(i)
                    cl_list[i] = None
                elif cl and l in cl.body_pos:
                    if not cl.head and len(cl.body_pos) == 1:
                        return None
                    cl.body_pos.remove(l)
        return model

    def _find_unit_clause(self, clauses):
        for cl in clauses:
            if cl and len(cl.head) == 1 and not cl.body_pos:
                return cl
        return None

    @property
    def models(self):
        """Generator yielding all models for the formula"""
        for assignment in _generate_subsets(self.variables):
            if self.satisfied(assignment):
                yield set(assignment)

    def copy(self):
        rules = [r.copy() for r in self.rules]
        return ImplyFormula(variables=self.variables, rules=rules)


class Rule:
    """Class representing a disjunction of literals, a.k.a. clause"""
    # separator of clauses
    or_sep = "; "
    and_sep = ", "
    imply = " <- "

    def __init__(self, head=None, body_pos=None, body_neg=None):
        """Creates a disjunction with every variable in pos occurring
        positively and every variable in neg occurring negatively"""
        if not head and not body_pos and not body_neg:
            raise ValueError("Empty head and body")
        self.head = set(head or [])
        self.body_pos = set(body_pos or [])
        self.body_neg = set(body_neg or [])

    @property
    def var_list(self):
        """Returns the set of the variables occurring in the clause"""
        return self.head | self.body_pos | self.body_neg

    @property
    def is_horn(self):
        return len(self.head) <= 1 and not self.body_neg

    def satisfied(self, assignment):
        """Receives an interpretation in the form of a sequence of true
        variables and returns true iff it is a model for the clause"""
        for v in self.head:
            if v in assignment:
                return True
        for v in self.body_pos:
            if v not in assignment:
                return True
        for v in self.body_neg:
            if v in assignment:
                return True
        return False

    @property
    def disj_form(self):
        return str(self.as_clause)

    @property
    def as_clause(self):
        return Clause(pos=self.head | self.body_neg, neg=self.body_pos)

    def __str__(self):
        h = Rule.or_sep.join(var.name for var in self.head)
        bp = Rule.and_sep.join(var.name for var in self.body_pos)
        bn = Rule.or_sep.join("\xac" + var.name for var in self.body_neg)
        if bp or bn:
            if bp and bn:
                b = Rule.and_sep.join((bp, bn))
            else:
                b = bp or bn
            return Rule.imply.join((h, b))
        else:
            return h

    def __eq__(self, cl):
        return (self.head == cl.head and
                self.body_neg == cl.body_neg and
                self.body_pos == cl.body_pos)

    def __hash__(self):
        return (13 * hash(self.head) +
                17 * hash(self.body_pos) +
                23 * hash(self.body_neg))

    def copy(self):
        return Rule(head=set(self.head),
                    body_pos=set(self.body_pos),
                    body_neg=set(self.body_neg))


def _generate_subsets(variables):
    variables = sorted(variables)
    yield []
    for r in range(1, len(variables)):
        for subset in it.combinations(variables, r):
            yield list(subset)
    yield variables


def random_formula(nvar=3, nclauses=5):
    """Creates a random formula with nvar variables and
    a number of clauses uniformally distributed in [1, nclauses]"""
    nclauses = randint(1, nclauses + 1)
    variables = [Variable(ll) for ll, _ in zip(string.ascii_letters, range(nvar))]
    cl = []
    while len(cl) != nclauses:
        try:
            variables_c = sample(variables, randint(0, nvar))
            neg = sample(variables, randint(0, nvar))
            cl.append(Clause(variables_c, neg))
        except ValueError:
            pass
    return Formula(variables=variables, clauses=cl)


def random_horn_formula(nvar=3, nclauses=5):
    """Creates a random Horn formula with nvar variables and
    a number of clauses uniformly distributed in [1, nclauses]"""
    nclauses = randint(1, nclauses + 1)
    variables = [Variable(ll) for ll, _ in zip(string.ascii_letters, range(nvar))]
    cl = []
    while len(cl) != nclauses:
        try:
            variables_c = [choice(variables)] if random() < 0.5 else []
            neg = sample(variables, randint(0, nvar))
            cl.append(Clause(variables_c, neg))
        except ValueError:
            pass
    return Formula(variables=variables, clauses=cl)


class Parser:
    OR, AND, NOT, VAR, EOF = "OR", "AND", "NOT", "VAR", "EOF"

    def __init__(self, or_expr=r"[ V|]",
                 and_expr=r"[\^&]", not_expr="[!\xac]"):
        self.or_expr = re.compile(or_expr)
        self.and_expr = re.compile(and_expr)
        self.not_expr = re.compile(not_expr)
        self.tokens = None
        self.curVar = None
        self.curToken = None

    def _next(self, init=False, expression=""):
        if init:
            self.tokens = deque(expression.split())
        if not self.tokens:
            self.curToken = Parser.EOF
            return
        token = self.tokens.popleft()
        if self.not_expr.match(token[0]):
            v = token[1:]
            self.tokens.appendleft(v)
            self.curToken = Parser.NOT
        elif self.or_expr.match(token):
            self.curToken = Parser.OR
        elif self.and_expr.match(token):
            self.curToken = Parser.AND
        else:
            self.curVar = token
            self.curToken = Parser.VAR

    def _ensure(self, token):
        if self.curToken != token:
            s = "Expected {}, got {}"
            raise ValueError(s.format(token, self.curToken))

    def build_formula(self, expression):
        # removes parenthesis
        expression = re.sub(r"[)(]", "", expression)
        # variables encountered during processing (String -> Variable)
        variables = {}
        self._next(True, expression)
        f = self._formula(variables)
        self._ensure(Parser.EOF)
        return f

    def _varn(self, variables):
        if self.curToken == Parser.NOT:
            self._next()
            self._ensure(Parser.VAR)
            vn = self.curVar
            if vn in variables:
                var = variables[vn]
            else:
                var = Variable(vn)
                variables[vn] = var
            self._next()
            return var, False
        elif self.curToken == Parser.VAR:
            vn = self.curVar
            if vn in variables:
                var = variables[vn]
            else:
                var = Variable(vn)
                variables[vn] = var
            self._next()
            return var, True
        else:
            raise ValueError("Error in VARN")

    def _clause(self, variables):
        pos = []
        negs = []
        var, positive = self._varn(variables)
        (pos if positive else negs).append(var)
        while self.curToken == Parser.OR:
            self._next()
            var, positive = self._varn(variables)
            (pos if positive else negs).append(var)
        return Clause(pos, negs)

    def _formula(self, variables):
        clauses = [self._clause(variables)]
        while self.curToken == Parser.AND:
            self._next()
            clauses.append(self._clause(variables))
        return Formula(variables=list(variables.values()), clauses=clauses)


class DimacsParser:
    OR, AND, NOT, VAR, EOF = "OR", "AND", "NOT", "VAR", "EOF"

    def __init__(self):
        pass

    def _next(self, init=False, expression=""):
        # TODO
        pass

    def _ensure(self, token):
        # TODO
        pass

    def build_formula(self, expression):
        # removes parenthesis
        expression = re.sub("[)(]", "", expression)
        # variables encountered during processing (String -> Variable)
        # variables = {}
        self._next(True, expression)
        self._ensure(Parser.EOF)
        pass


def main():
    f = Parser().build_formula("b V !a V c ^ c V !a V b ^ c V !b ^ b V !c ^ b V !a ^ d V !b V !c")
    hs = [{"b"}, {"c"}, {"c"}, {"b"}, {"b"}, {"d"}]
    f.set_heads(hs)
    print(f)
    print(f.imply_form)
    print("Is horn?", f.is_horn, sep="\t\t")
    if f.is_horn:
        min_model = f.minimal_model
        print("Minimum model:", min_model, sep="\t\t")
        if min_model:
            print("Is it a model?", f.satisfied(min_model), sep="\t\t")
    print("List of Models:", list(f.models), sep="\t\t")
    print("List of Stable Models:", list(f.stable_models), sep="\t\t")


if __name__ == '__main__':
    main()
