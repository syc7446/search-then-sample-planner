"""File for implementing various heuristics for use by the planner.
"""

import abc
import copy
import os
import functools
import tempfile
from pyperplan.pddl.parser import Parser as PyperplanParser
from pyperplan.grounding import ground as pyperplan_ground
from pyperplan.planner import HEURISTICS as PYPERPLAN_HEURISTICS
from pyperplan.search import searchspace as pyperplan_searchspace
from pddlgym.parser import PDDLDomain, PDDLProblemParser
from planner.utils.structs import LiteralConjunction


class Heuristic:
    """Base heuristic.
    """
    def __init__(self, env, operators, objects):
        self._env = env
        self._operators = copy.deepcopy(operators)
        self._objects = objects
        self._goal = LiteralConjunction(list(env.literal_goal))

    @staticmethod
    def make_domain(operators, goal, objects, operators_as_actions,
                    ignore_action_lits=True):
        """Create a PDDLDomain object from the given operators and goal.
        """
        # Extract the predicates and types from the operators and goal.
        preds = {}
        types = {str(o.var_type): o.var_type for o in objects}
        if hasattr(goal, "literals"):
            goal_lits = goal.literals
        else:
            goal_lits = goal.body.literals
        for lit in goal_lits:
            preds[lit.predicate.name] = lit.predicate
            for var_type in lit.predicate.var_types:
                types[str(var_type)] = var_type
        for op in operators:
            to_delete = set()
            for pre in op.preconds.literals:
                assert not pre.is_negative
                if ignore_action_lits and pre == op.action:
                    # Ignore action literal.
                    to_delete.add(pre)
                    continue
                if any(var_type.is_continuous
                       for var_type in pre.predicate.var_types):
                    # Ignore continuous preconditions.
                    to_delete.add(pre)
                    continue
                for var_type in pre.predicate.var_types:
                    types[str(var_type)] = var_type
                preds[pre.predicate.name] = pre.predicate
            for pre in to_delete:
                # Remove continuous preconditions and action literal.
                op.preconds.literals.remove(pre)
            for eff in op.effects.literals:
                if eff.is_anti:
                    eff = eff.inverted_anti
                assert not any(var_type.is_continuous
                               for var_type in eff.predicate.var_types)
                for var_type in eff.predicate.var_types:
                    types[str(var_type)] = var_type
                preds[eff.predicate.name] = eff.predicate
            # Remove operator continuous parameters.
            op.params = [p for p in op.params if not p.is_continuous]
        # Make domain object.
        operators_dict = {o.name: o for o in operators}
        actions = {o.action.predicate for o in operators}
        domain = PDDLDomain(domain_name="dummydomain", types=types,
                            predicates=preds, operators=operators_dict,
                            type_hierarchy={}, actions=actions,
                            operators_as_actions=operators_as_actions)
        return domain

    @abc.abstractmethod
    def __call__(self, node):
        """See Planner.Node in planner.py for definition of a node.
        """
        raise NotImplementedError("Override me!")


class BFSHeuristic(Heuristic):
    """Heuristic that implements breadth-first search.
    """
    def __call__(self, node):
        return len(node.skeleton)  # expand shorter plans first


class PyperplanBaseHeuristic(Heuristic):
    """Base class for Pyperplan heuristics.
    """
    def __init__(self, env, operators, objects):
        super().__init__(env, operators, objects)
        self._heuristic_name = self._get_heuristic_name()
        self._heuristic = None
        domain = self.make_domain(self._operators, self._goal, self._objects,
                                  operators_as_actions=False)
        self._domain_fname = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(self._domain_fname)

    @abc.abstractmethod
    def _get_heuristic_name(self):
        raise NotImplementedError("Override me!")

    def __call__(self, node):
        if self._heuristic is None:
            self._heuristic = self._get_heuristic_fn(node.lits)
        return self._heuristic(frozenset(node.lits))

    def _get_heuristic_fn(self, lits, cache_maxsize=10000):
        # Make problem file and set up Pyperplan objects.
        lits = {lit for lit in lits
                if not any(obj.is_continuous for obj in lit.variables)}
        try:
            problem_fname = tempfile.NamedTemporaryFile(delete=False).name
            PDDLProblemParser.create_pddl_file(
                problem_fname, self._objects, lits, "dummyproblem",
                "dummydomain", self._goal, fast_downward_order=True)
            parser = PyperplanParser(self._domain_fname, problem_fname)
            pyperplan_domain = parser.parse_domain()
            pyperplan_problem = parser.parse_problem(pyperplan_domain)
        finally:
            try:
                os.remove(self._domain_fname)
                os.remove(problem_fname)
            except FileNotFoundError:
                pass
        task = pyperplan_ground(pyperplan_problem)
        heuristic = PYPERPLAN_HEURISTICS[self._heuristic_name](task)

        @functools.lru_cache(cache_maxsize)
        def _call_heuristic(cur_lits):
            cur_lits = {lit for lit in cur_lits
                        if not any(obj.is_continuous for obj in lit.variables)}
            cur_objects = {obj for lit in cur_lits for obj in lit.variables}
            assert cur_objects.issubset(self._objects), \
                "If your object set changes, make a new heuristic object"
            state = task.facts & {lit.pddl_str().lower() for lit in cur_lits}
            node = pyperplan_searchspace.make_root_node(state)
            h = heuristic(node)
            return h
        return _call_heuristic


class PyperplanHAddHeuristic(PyperplanBaseHeuristic):
    """Pyperplan's hadd heuristic.
    """
    def _get_heuristic_name(self):
        return "hadd"


class PyperplanHFFHeuristic(PyperplanBaseHeuristic):
    """Pyperplan's hff heuristic.
    """
    def _get_heuristic_name(self):
        return "hff"
