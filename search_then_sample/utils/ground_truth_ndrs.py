"""Definitions of ground-truth NDRs for environments.
"""

from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME
from pddlgym.structs import Anti


def get_groundtruth_ndrs(env_name, env):
    """Get ground-truth NDRs for the given env.
    """
    if env_name == "pickplace":
        return _gtndrs_pickplace(env)
    elif env_name == "packinshelf":
        return _gtndrs_packinshelf(env)
    elif env_name == "namo":
        return _gtndrs_namo(env)
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


def _gtndrs_packinshelf(env):
    OnTable = env.OnTable
    InShelf = env.InShelf
    IsValidPickPlace = env.IsValidPickPlace
    # Action predicates
    Pack = env.Pack

    all_ndrs = {}

    action = Pack("?obj", "?basex", "?basey", "?basez", "?gripx", "?gripy", "?gripz")
    preconditions = [OnTable("?obj"),
                     IsValidPickPlace("?basex", "?basey", "?basez",
                                      "?gripx", "?gripy", "?gripz", "?obj")]
    effects = [{Anti(OnTable("?obj")), InShelf("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    return all_ndrs


def _gtndrs_namo(env):
    Blocked0 = env.Blocked0
    Cleared0 = env.Cleared0
    Blocked1 = env.Blocked1
    Cleared1 = env.Cleared1
    Blocked2 = env.Blocked2
    Cleared2 = env.Cleared2
    Blocked3 = env.Blocked3
    Cleared3 = env.Cleared3
    Blocked4 = env.Blocked4
    Cleared4 = env.Cleared4
    Blocked5 = env.Blocked5
    Cleared5 = env.Cleared5
    Blocked6 = env.Blocked6
    Cleared6 = env.Cleared6
    Blocked7 = env.Blocked7
    Cleared7 = env.Cleared7
    Blocked8 = env.Blocked8
    Cleared8 = env.Cleared8
    Blocked9 = env.Blocked9
    Cleared9 = env.Cleared9
    GoalReach = env.GoalReach
    IsValidClear = env.IsValidClear
    IsValidPickPlace = env.IsValidPickPlace
    # Action predicates
    Clear0 = env.Clear0
    Clear1 = env.Clear1
    Clear2 = env.Clear2
    Clear3 = env.Clear3
    Clear4 = env.Clear4
    Clear5 = env.Clear5
    Clear6 = env.Clear6
    Clear7 = env.Clear7
    Clear8 = env.Clear8
    Clear9 = env.Clear9
    PickPlace = env.PickPlace

    all_ndrs = {}

    action = Clear0("?obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked0("?obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked0("?obj")), Cleared0("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear1("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked1("?obj"), Cleared0("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked1("?obj")), Cleared1("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear2("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked2("?obj"), Cleared1("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked2("?obj")), Cleared2("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear3("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked3("?obj"), Cleared2("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked3("?obj")), Cleared3("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear4("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked4("?obj"), Cleared3("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked4("?obj")), Cleared4("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear5("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked5("?obj"), Cleared4("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked5("?obj")), Cleared5("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear6("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked6("?obj"), Cleared5("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked6("?obj")), Cleared6("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear7("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked7("?obj"), Cleared6("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked7("?obj")), Cleared7("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear8("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked8("?obj"), Cleared7("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked8("?obj")), Cleared8("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = Clear9("?obj", "?pre_obj", "?basex", "?basey", "?baset")
    preconditions = [Blocked9("?obj"), Cleared8("?pre_obj"),
                     IsValidClear("?basex", "?basey", "?baset", "?obj")]
    effects = [{Anti(Blocked9("?obj")), Cleared9("?obj")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    action = PickPlace("?target", "?obj", "?basex", "?basey", "?baset")
    preconditions = [Cleared9("?obj"),
                     IsValidPickPlace("?basex", "?basey", "?baset", "?target")]
    effects = [{GoalReach()}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    return all_ndrs