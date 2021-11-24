"""Definitions of ground-truth NDRs for environments.
"""

from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME
from pddlgym.structs import Anti


def get_groundtruth_ndrs(env_name, env):
    """Get ground-truth NDRs for the given env.
    """
    if env_name == "blocks":
        return _gtndrs_blocks(env)
    if env_name == "bins":
        return _gtndrs_bins(env)
    raise Exception(f"Unrecognized env: {env_name}")


def _gtndrs_blocks(env):
    On = env.On
    OnTable = env.OnTable
    Holding = env.Holding
    Clear = env.Clear
    HandEmpty = env.HandEmpty
    HandFull = env.HandFull
    IsValidGrasp = env.IsValidGrasp
    IsValidPlaceOnBlock = env.IsValidPlaceOnBlock
    IsValidPlaceOnTable = env.IsValidPlaceOnTable
    # Action predicates
    PickupFromBlock = env.PickupFromBlock
    PickupFromTable = env.PickupFromTable
    PutOnBlock = env.PutOnBlock
    PutOnTable = env.PutOnTable

    all_ndrs = {}

    # PickupFromTable
    action = PickupFromTable("?block", "?posex", "?posey", "?posez")
    preconditions = [Clear("?block"), OnTable("?block"), HandEmpty(),
                     IsValidGrasp("?posex", "?posey", "?posez", "?block")]
    effects = [{Anti(Clear("?block")), Anti(OnTable("?block")),
                Anti(HandEmpty()), HandFull(), Holding("?block")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    # PutOnTable
    action = PutOnTable("?block", "?posex", "?posey", "?posez")
    preconditions = [HandFull(), Holding("?block"),
                     IsValidPlaceOnTable("?posex", "?posey", "?posez")]
    effects = [{Anti(HandFull()), Anti(Holding("?block")),
                Clear("?block"), OnTable("?block"), HandEmpty()},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    # PutOnBlock
    action = PutOnBlock("?heldblock", "?otherblock",
                        "?posex", "?posey", "?posez")
    preconditions = [Holding("?heldblock"), Clear("?otherblock"),
                     HandFull(), IsValidPlaceOnBlock(
                         "?posex", "?posey", "?posez", "?otherblock")]
    effects = [{Anti(Holding("?heldblock")), Anti(Clear("?otherblock")),
                Anti(HandFull()), HandEmpty(), Clear("?heldblock"),
                On("?heldblock", "?otherblock")}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    # PickupFromBlock
    action = PickupFromBlock("?targetblock", "?otherblock",
                             "?posex", "?posey", "?posez")
    preconditions = [HandEmpty(), On("?targetblock", "?otherblock"),
                     Clear("?targetblock"), IsValidGrasp(
                         "?posex", "?posey", "?posez", "?targetblock")]
    effects = [{Anti(HandEmpty()), Anti(On("?targetblock", "?otherblock")),
                Anti(Clear("?targetblock")), HandFull(),
                Holding("?targetblock"), Clear("?otherblock")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    all_ndrs[action] = NDRSet(action, [NDR(action, preconditions,
                                           effect_probs, effects)])

    return all_ndrs


def _gtndrs_bins(env):
    OnTable = env.OnTable
    Holding = env.Holding
    HoldingSide = env.HoldingSide
    HoldingTop = env.HoldingTop
    InShelf = env.InShelf
    InBox = env.InBox
    HandEmpty = env.HandEmpty
    HandFull = env.HandFull
    IsTargetForObj = env.IsTargetForObj
    TargetInShelf = env.TargetInShelf
    TargetInBox = env.TargetInBox
    IsValidGrasp = env.IsValidGrasp
    IsValidPlace = env.IsValidPlace
    # Action predicates
    Pick = env.Pick
    Place = env.Place

    all_ndrs = {}

    # Pick
    pick_ndrs = []

    ## Side grasp
    action = Pick("?obj", "?basex", "?basey", "?basez",
                  "?gripx", "?gripy", "?gripz")
    preconditions = [OnTable("?obj"), HandEmpty(),
                     IsValidGrasp("?basex", "?basey", "?basez",
                                  "?gripx", "?gripy", "?gripz", "?obj")]
    effects = [{Anti(OnTable("?obj")), Anti(HandEmpty()),
                HandFull(), HoldingSide("?obj"), Holding("?obj")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))

    ## Top grasp
    action = Pick("?obj", "?basex", "?basey", "?basez",
                  "?gripx", "?gripy", "?gripz")
    preconditions = [OnTable("?obj"), HandEmpty(),
                     IsValidGrasp("?basex", "?basey", "?basez",
                                  "?gripx", "?gripy", "?gripz", "?obj")]
    effects = [{Anti(OnTable("?obj")), Anti(HandEmpty()),
                HandFull(), HoldingTop("?obj"), Holding("?obj")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    pick_ndrs.append(NDR(action, preconditions, effect_probs, effects))

    all_ndrs[action] = NDRSet(action, pick_ndrs)

    # Place
    place_ndrs = []

    ## Place into shelf while HoldingSide -- succeeds
    action = Place("?target", "?obj", "?basex", "?basey", "?basez",
                   "?gripx", "?gripy", "?gripz")
    preconditions = [HoldingSide("?obj"), Holding("?obj"), HandFull(),
                     IsTargetForObj("?target", "?obj"),
                     TargetInShelf("?target"),
                     IsValidPlace("?basex", "?basey", "?basez",
                                  "?gripx", "?gripy", "?gripz", "?target")]
    effects = [{Anti(HoldingSide("?obj")), Anti(Holding("?obj")),
                Anti(HandFull()), HandEmpty(), InShelf("?obj")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    place_ndrs.append(NDR(action, preconditions, effect_probs, effects))

    ## Place into shelf while HoldingTop -- fails
    action = Place("?target", "?obj", "?basex", "?basey", "?basez",
                   "?gripx", "?gripy", "?gripz")
    preconditions = [HoldingTop("?obj"), Holding("?obj"), HandFull(),
                     IsTargetForObj("?target", "?obj"),
                     TargetInShelf("?target"),
                     IsValidPlace("?basex", "?basey", "?basez",
                                  "?gripx", "?gripy", "?gripz", "?target")]
    effects = [{Anti(HoldingTop("?obj")), Anti(Holding("?obj")),
                Anti(HandFull()), HandEmpty()}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    place_ndrs.append(NDR(action, preconditions, effect_probs, effects))

    ## Place into box while HoldingTop -- succeeds
    action = Place("?target", "?obj", "?basex", "?basey", "?basez",
                   "?gripx", "?gripy", "?gripz")
    preconditions = [HoldingTop("?obj"), Holding("?obj"), HandFull(),
                     IsTargetForObj("?target", "?obj"),
                     TargetInBox("?target"),
                     IsValidPlace("?basex", "?basey", "?basez",
                                  "?gripx", "?gripy", "?gripz", "?target")]
    effects = [{Anti(HoldingTop("?obj")), Anti(Holding("?obj")),
                Anti(HandFull()), HandEmpty(), InBox("?obj")},
               {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    place_ndrs.append(NDR(action, preconditions, effect_probs, effects))

    ## Place into box while HoldingSide -- fails
    action = Place("?target", "?obj", "?basex", "?basey", "?basez",
                   "?gripx", "?gripy", "?gripz")
    preconditions = [HoldingSide("?obj"), Holding("?obj"), HandFull(),
                     IsTargetForObj("?target", "?obj"),
                     TargetInBox("?target"),
                     IsValidPlace("?basex", "?basey", "?basez",
                                  "?gripx", "?gripy", "?gripz", "?target")]
    effects = [{Anti(HoldingSide("?obj")), Anti(Holding("?obj")),
                Anti(HandFull()), HandEmpty()}, {NOISE_OUTCOME}]
    effect_probs = [1.0, 0.0]
    place_ndrs.append(NDR(action, preconditions, effect_probs, effects))

    all_ndrs[action] = NDRSet(action, place_ndrs)

    return all_ndrs
