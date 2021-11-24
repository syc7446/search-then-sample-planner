"""General utilities.
"""

import os
from structs import Type

# Global world object for environments
WORLD = Type("world")("world")


def compute_static_preds(operators, predicates):
    """Compute the static predicates under the given operators.
    """
    static_preds = set()
    for pred in predicates:
        if any(_op_changes_predicate(op, pred) for op in operators):
            continue
        static_preds.add(pred)
    return static_preds


def _op_changes_predicate(op, pred):
    """Helper method for compute_static_preds.
    """
    for lit in op.effects.literals:
        assert not lit.is_negative
        if lit.is_anti:
            eff_pred = lit.inverted_anti.predicate
        else:
            eff_pred = lit.predicate
        if eff_pred == pred:
            return True
    return False


def compute_delete_relax_reachable_lits(state, operators):
    """Compute the literals reachable from the state under the operators
    """
    reachable_lits = set(state)
    while True:
        fixed_point_reached = True
        for op in operators:
            positive_preconditions = {lit for lit in op.preconds.literals
                                      if not lit.is_negative
                                      and lit != op.action
                                      and not any(v.is_continuous
                                                  for v in lit.variables)}
            if positive_preconditions.issubset(reachable_lits):
                positive_effects = {lit for lit in op.effects.literals
                                    if not lit.is_anti}
                for new_reachable_lit in positive_effects - reachable_lits:
                    fixed_point_reached = False
                    reachable_lits.add(new_reachable_lit)
        if fixed_point_reached:
            break
    return reachable_lits



def get_asset_path(asset_name):
    """Helper for locating files in the assets directory
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, "assets")
    return os.path.join(asset_dir_path, asset_name)
