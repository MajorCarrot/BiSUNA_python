import sys
from enum import auto, Enum
from dataclasses import dataclass

from BiSUNA.parameters import *

STRUCTURE_DNA = True

# all neurons not activated by a control neuron are inactive
# by default
DEFAULT_INACTIVE = True

# structure and weight cycles are inexistent, all the evolution
# happens on only a structure cycle and imprecise weights
NO_WEIGHT_STRUCTURE_CYCLE = True

class NeuronTypes(Enum):
    IDENTITY = 0
    ACTIVATION = 1
    THRESHOLD = 2
    RANDOM = 3
    CONTROL = 4
    NUMBER_OF_NEURON_TYPES = 5
    INPUT_IDENTITY = 6
    OUTPUT_IDENTITY = 7


class ConnectionTypes(Enum):
    Recurrent = 0
    FeedForward = 1
    NumberConnTypes = 2


@dataclass
class connection_type:
    index: int
    cType: ConnectionTypes


# Starting levels of firing rate
class FiringRate(Enum):
    LEVEL1 = 1
    LEVEL7 = 7
    LEVEL49 = 49
    NUMER_OF_FIRING_RATE_LEVELS = 3


@dataclass
class neuron:
    neuron_id: int = -1
    firing_rate: FiringRate = None
    neuron_type: NeuronTypes = None
    interface_index: int = None


@dataclass
class connection:
    from_neuron_id: int = -1
    to_neuron_id: int = -1
    weight: ParameterType = None
    neuro_modulation: int = None


def randomFiringRateLevel(random):
    level = random.randint(0, FiringRate.NUMER_OF_FIRING_RATE_LEVELS - 1)
    if level == 0:
        return FiringRate.LEVEL1
    elif level == 1:
        return FiringRate.LEVEL7
    elif level == 2:
        return FiringRate.LEVEL49
    else:
        print(f"Unknown level {level} returned by random")
        sys.exit(1)

def strNeuronType(neuron_type) -> None:
    if neuron_type == NeuronTypes.IDENTITY:
        return "Identity"
    elif neuron_type == NeuronTypes.ACTIVATION:
        return "Activation"
    elif neuron_type == NeuronTypes.RANDOM:
        return "Random"
    elif neuron_type == NeuronTypes.THRESHOLD:
        return "Threshold"
    elif neuron_type == NeuronTypes.CONTROL:
        return "Control"
    elif neuron_type == NeuronTypes.INPUT_IDENTITY:
        return "Input Identity"
    elif neuron_type == NeuronTypes.OUTPUT_IDENTITY:
        return "Output Identity"
    else:
        print(f"ERROR: Incorrect value of neuron type {neuron_type}")
        raise ValueError


def betaFromFR(fr) -> ParameterType:
    if CONTINUOUS_PARAM:
        frD = float(fr)
        partial = 1 / frD;
        return partial
    else:
        if fr == FiringRate.LEVEL1:
            return MAXIMUM_WEIGHT
        elif fr == FiringRate.LEVEL7:
            return 2047
        elif fr == FiringRate.LEVEL49:
            return 63
        else:
            return 0


def xnor(x, y) -> ParameterType:
    if CONTINUOUS_PARAM:
        return 0
    else:
        return ~(x ^ y);
