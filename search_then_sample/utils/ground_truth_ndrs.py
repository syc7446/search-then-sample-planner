"""Definitions of ground-truth NDRs for environments.
"""

from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME
from pddlgym.structs import Anti


def get_groundtruth_ndrs(env_name, env):
    """Get ground-truth NDRs for the given env.
    """
    if env_name == "pickplace":
        return _gtndrs_pickplace(env)
    raise Exception(f"Unrecognized env: {env_name}")


def _gtndrs_pickplace(env):
    OnTable = env.OnTable
    OnStove = env.OnStove
    Holding = env.Holding
    HoldingSide = env.HoldingSide
    HandEmpty = env.HandEmpty
    HandFull = env.HandFull
    IsValidPick = env.IsValidPick
    IsValidPlace = env.IsValidPlace
    # Action predicates
    Pick = env.Pick
    Place = env.Place

    all_ndrs = {}

    action = Pick("?obj", "?basex", "?basey", "?basez", "?gripx", "?gripy", "?gripz")
    preconditions = [OnTable("?obj"), HandEmpty(),
                     IsValidPick("?basex", "?basey", "?basez",
                                 "?gripx", "?gripy", "?gripz", "?obj")]
    effects = [{Anti(OnTable("?obj")), Anti(HandEmpty()),
                HandFull(), HoldingSide("?obj"), Holding("?obj")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Place("?obj", "?basex", "?basey", "?basez", "?gripx", "?gripy", "?gripz")
    preconditions = [HandFull(), HoldingSide("?obj"), Holding("?obj"),
                     IsValidPlace("?basex", "?basey", "?basez",
                                 "?gripx", "?gripy", "?gripz", "?obj")]
    effects = [{OnStove("?obj"), HandEmpty(),
                Anti(HandFull()), Anti(HoldingSide("?obj")), Anti(Holding("?obj"))}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    return all_ndrs