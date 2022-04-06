import numpy as np

import search_then_sample.utils.structs as structs
from search_then_sample.utils.env_base import Environment, EnvironmentFailure
from search_then_sample.utils.utils import WORLD
from pybullet_planning.pybullet_tools.pr2_problems import holding_problem


class PickPlaceEnvironment(Environment):
    # Object types
    _obj_type = structs.Type("obj")
    _target_type = structs.Type("target")
    _xbase_type = structs.ContinuousType("xbase")
    _ybase_type = structs.ContinuousType("ybase")
    _zbase_type = structs.ContinuousType("zbase")
    _xgrip_type = structs.ContinuousType("xgrip")
    _ygrip_type = structs.ContinuousType("ygrip")
    _zgrip_type = structs.ContinuousType("zgrip")
    _xbase_type.set_domain([-10, 10])
    _ybase_type.set_domain([-10, 10])
    _zbase_type.set_domain([-10, 10])
    _xgrip_type.set_domain([-10, 10])
    _ygrip_type.set_domain([-10, 10])
    _zgrip_type.set_domain([-10, 10])

    # Constant world object
    _world = WORLD

    # Predicates
    OnTable = structs.Predicate("OnTable", 1, [_obj_type])
    Holding = structs.Predicate("Holding", 1, [_obj_type])
    HoldingSide = structs.Predicate("HoldingSide", 1, [_obj_type])
    HoldingTop = structs.Predicate("HoldingTop", 1, [_obj_type])
    InShelf = structs.Predicate("InShelf", 1, [_obj_type])
    InBox = structs.Predicate("InBox", 1, [_obj_type])
    HandEmpty = structs.Predicate("HandEmpty", 0, [])
    HandFull = structs.Predicate("HandFull", 0, [])
    IsTargetForObj = structs.Predicate("IsTargetForObj", 2,
                                       [_target_type, _obj_type])
    TargetInShelf = structs.Predicate("TargetInShelf", 1, [_target_type])
    TargetInBox = structs.Predicate("TargetInBox", 1, [_target_type])
    IsValidGrasp = structs.Predicate("IsValidGrasp", 7,
                                     [_xbase_type, _ybase_type, _zbase_type,
                                      _xgrip_type, _ygrip_type, _zgrip_type,
                                      _obj_type])
    IsValidPlace = structs.Predicate("IsValidPlace", 7,
                                     [_xbase_type, _ybase_type, _zbase_type,
                                      _xgrip_type, _ygrip_type, _zgrip_type,
                                      _target_type])
    _all_predicates = {OnTable, Holding, HoldingSide, HoldingTop,
                       InShelf, InBox, HandEmpty, HandFull,
                       IsTargetForObj, TargetInShelf, TargetInBox,
                       IsValidGrasp, IsValidPlace}
    _all_predicate_names_to_preds = {p.name: p for p in _all_predicates}
    _continuous_predicates = {pred for pred in _all_predicates
                              if any(t.is_continuous for t in pred.var_types)}

    # Actions
    Pick = structs.Predicate("Pick", 7,
                             [_obj_type,
                              _xbase_type, _ybase_type, _zbase_type,
                              _xgrip_type, _ygrip_type, _zgrip_type])
    Place = structs.Predicate("Place", 8,
                              [_target_type, _obj_type,
                               _xbase_type, _ybase_type, _zbase_type,
                               _xgrip_type, _ygrip_type, _zgrip_type])
    action_predicates = {Pick, Place}

    def __init__(self, num_objs, seed):
        super().__init__(num_objs, seed)
        self._num_targets = num_objs  # each object has a target
        self._objs = []
        self._targets = []
        for i in range(self._num_objs):
            self._objs.append(self._obj_type("obj{}".format(i)))
            self._targets.append(self._target_type("target{}".format(i)))
        self._initial_pybullet_setup()
        self._objs_to_obj_ids = {}
        self._targets_to_target_ids = {}
        self._transmodel_cache = {}

    def _initial_pybullet_setup(self):
        problem = holding_problem()