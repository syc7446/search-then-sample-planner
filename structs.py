"""Define structs that are useful throughout the code.
"""

from pddlgym import structs as pddlgym_structs
from pddlgym.parser import Operator as PDDLGymOperator


class Type(pddlgym_structs.Type):
    """Like a PDDLGym type, but entities contain a value in addition to a name.
    """
    is_continuous = False

    def __call__(self, entity_name, entity_value=None):
        assert entity_value is None, "Discrete entities can't have a value"
        return TypedEntity.__new__(TypedEntity, entity_name, self, entity_value)  # pylint:disable=too-many-function-args


class ContinuousType(Type):
    """A continuous type
    """
    is_continuous = True

    def set_domain(self, domain):
        """Required to call this method after initializing a ContinuousType.
        Sets the domain of this type.
        """
        assert isinstance(domain, list) and len(domain) == 2
        self.domain = domain  # pylint:disable=attribute-defined-outside-init

    def __call__(self, entity_name, entity_value=None):
        if not entity_name.startswith("?") and entity_value is None:
            raise Exception("Continuous entities must have a value, unless "
                            "they're lifted expressions (name starts with a ?)")
        return TypedEntity.__new__(TypedEntity, entity_name, self, entity_value,  # pylint:disable=too-many-function-args
                                   True)


class TypedEntity(pddlgym_structs.TypedEntity):
    """Like a PDDLGym entity, but contains a value in addition to a name.
    """
    def __new__(cls, name, var_type, value=None, is_continuous=False):
        obj = pddlgym_structs.TypedEntity.__new__(cls, name, var_type)  # pylint:disable=too-many-function-args
        if value is None:  # use name as value
            obj.value = obj.name
        else:
            obj.value = value
        obj.is_continuous = is_continuous
        return obj


class Predicate(pddlgym_structs.Predicate):
    """Unchanged; this class is just here so that this file
    can be a drop-in replacement for PDDLGym's structs.py.
    """
    pass


class LiteralConjunction(pddlgym_structs.LiteralConjunction):
    """Unchanged; this class is just here so that this file
    can be a drop-in replacement for PDDLGym's structs.py.
    """
    pass


class Operator(PDDLGymOperator):
    """Include in the operator a reference to the associated action.
    """
    def __init__(self, action, name, params, preconds, effects):
        self.action = action
        super().__init__(name, params, preconds, effects)
