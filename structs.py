"""Define structs that are useful throughout the code.
"""

from pddlgym import structs as pddlgym_structs
from pddlgym.parser import Operator as PDDLGymOperator
import abc


### Variables
class Variable:
    """Generic variable.
    """
    def __init__(self, name):
        self.name = name
        self._hash = hash(self.name)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return repr(self.name)

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    @abc.abstractmethod
    def sample(self):
        """Sample a random value.
        """
        raise NotImplementedError("Override me!")


class DiscreteVariable(Variable):
    """Represents a discrete variable. Each variable's name should be unique.
    """
    def __init__(self, name, size):
        self.size = size
        self.domain = list(range(size))
        self.arbitrary_value = self.domain[0]
        super().__init__(name)

    def sample(self):
        """Sample a random value.
        """
        raise NotImplementedError


class StateVariableBase(Variable):
    """Represents a state variable, which is a variable that has
    predecessors and successors (e.g., on-road and next-on-road).
    """
    def __init__(self, name, size, _prev_ptr=None):
        # Don't pass in a prev_ptr externally; it's only for internal use.
        super().__init__(name, size)
        if _prev_ptr is None:
            self.next = self.__class__("next-"+name, size, _prev_ptr=self)
            self.prev = None
            self.is_next = False
        else:
            self.prev = _prev_ptr
            self.next = None
            self.is_next = True


class StateVariable(StateVariableBase, DiscreteVariable):
    """Representation of a discrete state variable.
    """
    pass


LIMBO = StateVariable("limbo", 2)


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
