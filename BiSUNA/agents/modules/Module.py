import sys
from copy import deepcopy
from math import isnan, isinf
from random import Random

from BiSUNA.agents.modules.structural_dna import *


class Module:
    def __init__(
        self,
        number_of_inputs,
        number_of_outputs,
        suggested_allocation_length=None,
        n=None,
        c=None,
        dna_allocated_length=None,
    ):
        self.random = Random
        if suggested_allocation_length is not None:
            self.allocated_space = suggested_allocation_length
            self.n = [neuron()] * suggested_allocation_length
            self.c = [connection()] * suggested_allocation_length
            self.max_neuron_id = -1
            self.number_of_neurons = 0
            self.number_of_connections = 0

            self.internal_neuron_state = [0] * self.allocated_space
            self.neuron_state = [0] * self.allocated_space
            self.previous_neuron_state = [0] * self.allocated_space

            self.neuron_excitation = [0] * self.allocated_space
            self.is_fired = [False] * self.allocated_space
            self.connection_state = [
                ConnectionTypes.Recurrent
            ] * self.allocated_space
            self.previous_connection_state = [
                ConnectionTypes.Recurrent
            ] * self.allocated_space
            self.primer_list = [0] * self.allocated_space

            self.number_of_primers = 0

            self.primer_list[self.number_of_primers] = -1

            self.createFixedInterfaceNeurons(
                number_of_inputs, number_of_outputs
            )
        else:
            self.n = n
            self.c = c
            self.allocated_space = dna_allocated_length

            if self.random is None:
                print(
                    "Static Random variable not set!!!!"
                    "\nPlayer take 5 damage =P!"
                )
                sys.exit(1)

            self.max_neuron_id = 0
            self.number_of_neurons = 0
            self.number_of_connections = 0

            i = 0
            for ni in self.n:
                if ni.neuron_id < 0:
                    break

                if ni.neuron_id > self.max_neuron_id:
                    self.max_neuron_id = ni.neuron_id

                i += 1
                self.number_of_neurons += 1

            for ci in self.c:
                if ci.from_neuron_id < 0:
                    break
                self.number_of_connections += 1

            self.internal_neuron_state = [0] * self.allocated_space
            self.neuron_state = [0] * self.allocated_space
            self.previous_neuron_state = [0] * self.allocated_space

            self.neuron_excitation = [0] * self.allocated_space
            self.is_fired = [False] * self.allocated_space
            self.connection_state = [
                ConnectionTypes.Recurrent
            ] * self.allocated_space
            self.previous_connection_state = [
                ConnectionTypes.Recurrent
            ] * self.allocated_space
            self.primer_list = [0] * self.allocated_space

            self.updatePrimerList()
            self.createFixedInterfaceNeurons(
                number_of_inputs, number_of_outputs
            )

    def setRandom(self, random):
        self.random = random

    def clone(self, brother):
        assert isinstance(brother, Module)
        self.allocated_space = brother.allocated_space
        self.max_neuron_id = brother.max_neuron_id
        self.number_of_neurons = brother.number_of_neurons
        self.number_of_connections = brother.number_of_connections
        self.number_of_primers = brother.number_of_primers
        self.n = deepcopy(brother.n)
        self.c = deepcopy(brother.c)
        self.primer_list = deepcopy(brother.primer_list)

        self.clearMemory()

    def process(self, inputs, output):
        self.input = inputs
        self.output = output

        self.processInputNeurons()
        self.processPrimers()
        self.processControlNeurons()
        self.processRemainingNeurons()

        for i in range(self.number_of_neurons):
            self.is_fired[i] = False

            if isnan(self.previous_neuron_state[i]) or isinf(
                self.previous_neuron_state[i]
            ):
                print(f"Previous neuron state is nan or inf at {i}")

            if isnan(self.neuron_state[i]) or isinf(self.neuron_state[i]):
                print(f"Neuron state is nan or inf at {i}")

            ns = self.neuron_state[i]
            self.previous_neuron_state[i] = ns
            self.neuron_excitation[i] = 0.0
            self.neuron_state[i] = 0.0

        for i in range(self.number_of_connections):
            self.previous_connection_state[i] = self.connection_state[i]
            self.connection_state[i] = ConnectionTypes.Recurrent

    def weightMutation(self):
        roulette = self.random.uniform(0.0, 1.0)
        if roulette < CHANCE_OF_WEIGHT_MUTATION:
            for ci in self.c:
                if ci.from_neuron_id < 0:
                    break

                if ci.neuro_modulation < 0:
                    cw = ci.weight
                    if CONTINUOUS_PARAM:
                        variance = WEIGHT_MUTATION_CHANGE_PERCENTAGE
                        perturbation = self.random.uniform(-variance, variance)
                        ci.weight += perturbation
                    else:
                        varianceP = cw << WEIGHT_MUTATION_CHANGE_PERCENTAGE
                        varianceN = cw >> WEIGHT_MUTATION_CHANGE_PERCENTAGE
                        perturbation = self.random.uniform(
                            varianceN, varianceP
                        )
                        ci.weight = perturbation | cw

                    if ci.weight > MAXIMUM_WEIGHT or isinf(ci.weight):
                        ci.weight = MAXIMUM_WEIGHT

                    if CONTINUOUS_PARAM:
                        if ci.weight < -MAXIMUM_WEIGHT:
                            ci.weight = -MAXIMUM_WEIGHT

    def structuralMutation(self):
        mutation_chance = MUTATION_PROBABILITIES
        roulette = self.random.uniform(0, 1)
        total = 0
        for i in range(len(mutation_chance)):
            total += mutation_chance[i]
            if roulette < total:
                mutation_type = i
                break
        else:
            mutation_type = len(mutation_chance)

        if mutation_type == 1:
            new_index = self.number_of_neurons
            self.number_of_neurons += 1
            new_id = self.smallestFreeId()

            if self.max_neuron_id < new_id:
                self.max_neuron_id = new_id

            if self.allocated_space <= self.number_of_neurons + 1:
                self.reallocEverything()

            if self.random.uniform(0, 1) < CHANCE_OF_CONTROL_NEURON:
                self.n[new_index].type = NeuronTypes.CONTROL
            else:
                nont = int(NeuronTypes.NUMBER_OF_NEURON_TYPES)
                val = self.random.randint(0, nont - 1)
                self.n[new_index].type = NeuronTypes(val)

            self.n[new_index].neuron_id = new_id
            self.n[new_index].firing_rate = randomFiringRateLevel(self.random)
            self.n[new_index].neuron_id = -1

            if self.n[new_index].type == NeuronTypes.CONTROL:
                self.primer_list[self.number_of_primers] = new_id
                self.number_of_primers += 1
                self.primer_list[self.number_of_primers] = -1

            self.connectNewNeuronToNetwork(new_id)
        elif mutation_type == 2:
            delete_index = self.random.randint(0, self.number_of_neurons - 1)
            delete_id = self.n[delete_index].neuron_id

            if delete_id == -1:
                return

            if self.n[delete_index].type in (
                NeuronTypes.INPUT_IDENTITY,
                NeuronTypes.OUTPUT_IDENTITY,
            ):
                return

            if self.n[delete_index].type == NeuronTypes.CONTROL:
                for i in range(self.number_of_primers):
                    if i >= self.number_of_primers:
                        break

                    if self.primer_list[i] < 0:
                        break

                    if self.primer_list[i] == delete_id:
                        self.primer_list[i] = self.primer_list[
                            self.number_of_primers - 1
                        ]
                        self.number_of_primers -= 1
                        self.primer_list[self.number_of_primers] = -1

            # Deleting
            self.n[delete_index].neuron_id = self.n[
                self.number_of_neurons - 1
            ].neuron_id
            self.n[delete_index].type = self.n[self.number_of_neurons - 1].type
            self.n[delete_index].firing_rate = self.n[
                self.number_of_neurons - 1
            ].firing_rate
            self.previous_neuron_state[
                delete_index
            ] = self.previous_neuron_state[self.number_of_neurons - 1]
            self.internal_neuron_state[
                delete_index
            ] = self.internal_neuron_state[self.number_of_neurons - 1]

            self.number_of_neurons -= 1
            self.n[self.number_of_neurons].neuron_id = -1

            found = True
            while found:
                found = False
                i = 0
                while True:
                    if self.c[i].from_neuron_id < 0 or found:
                        break

                    if delete_id in (
                        self.c[i].from_neuron_id,
                        self.c[i].to_neuron_id,
                    ):
                        self.removeConnection(i)
                        found = True

                    i += 1
        elif mutation_type == 3:
            if self.allocated_space <= self.number_of_connections + 1:
                self.reallocEverything()

            if self.n[0].id == -1:
                return

            random_index = self.random.randint(0, self.number_of_neurons - 1)
            self.c[self.number_of_connections].from_neuron_id = self.n[
                random_index
            ].neuron_id

            random_index = self.random.randint(0, self.number_of_neurons - 1)
            self.c[self.number_of_connections].to_neuron_id = self.n[
                random_index
            ].neuron_id

            if self.random.uniform(0, 1) < CHANCE_OF_NEUROMODULATION:
                random_index = self.random.randint(
                    0, self.number_of_neurons - 1
                )
                self.c[self.number_of_connections].neuro_modulation = self.n[
                    random_index
                ].neuron_id
                result = 1
            else:
                random_index = self.random.randint(0, 1)
                self.c[self.number_of_connections].neuro_modulation = -1
                if random_index == 0:
                    result = 1
                else:
                    result = -1

            self.c[self.number_of_connections].weight = result
            self.number_of_connections += 1
            self.c[self.number_of_connections].from_neuron_id = -1
        elif mutation_type == 4:
            if self.number_of_connections == 0:
                return

            delete_index = self.random.randint(
                0, self.number_of_connections - 1
            )
            self.removeConnection(delete_index)
        else:
            print(f"Unknown Mutation Type {mutation_type}")
            raise ValueError

    def addConnection(
        self, from_neuron_id, to_neuron_id, neuro_modulation, weight
    ):
        if self.allocated_space <= self.number_of_connections + 1:
            self.reallocEverything()

        if self.n[0].neuron_id == -1:
            return

        self.c[self.number_of_connections].from_neuron_id = from_neuron_id
        self.c[self.number_of_connections].to_neuron_id = to_neuron_id
        self.c[self.number_of_connections].neuro_modulation = neuro_modulation
        self.c[self.number_of_connections].weight = weight

        self.number_of_connections += 1
        self.c[self.number_of_connections].from_neuron_id = -1

    def connectNewNeuronToNetwork(self, new_neuron_id):
        if self.allocated_space <= self.number_of_connections + 2:
            self.reallocEverything()

        # Create a random connection from the new neuron
        self.c[self.number_of_connections].from_neuron_id = new_neuron_id
        random_index = self.random.randint(0, self.number_of_neurons - 1)
        self.c[self.number_of_connections].to_neuron_id = self.n[
            random_index
        ].neuron_id

        weight = ParameterType(1)

        if self.random.uniform(0, 1) < CHANCE_OF_NEUROMODULATION:
            random_index2 = self.random.randint(0, self.number_of_neurons - 1)
            self.c[self.number_of_connections].neuro_modulation = self.n[
                random_index2
            ].neuron_id
        else:
            self.c[self.number_of_connections].neuro_modulation = -1
            random_index2 = self.random.randint(0, 1)
            if random_index2:
                weight = -1

        self.c[self.number_of_connections].weight = weight

        # Second connection --------------------------------
        self.number_of_connections += 1
        self.c[self.number_of_connections].from_neuron_id = -1

        # Create a new connection to the new neuron
        random_index = self.random.randint(0, self.number_of_neurons - 1)
        self.c[self.number_of_connections].from_neuron_id = self.n[
            random_index
        ].neuron_id
        self.c[self.number_of_connections].to_neuron_id = new_neuron_id

        weight = 1

        if self.random.uniform(0, 1) < CHANCE_OF_NEUROMODULATION:
            random_index2 = self.random.randint(0, self.number_of_neurons - 1)
            self.c[self.number_of_connections].neuro_modulation = self.n[
                random_index2
            ].neuron_id
        else:
            self.c[self.number_of_connections].neuro_modulation = -1
            random_index2 = self.random.randint(0, 1)
            if random_index2:
                weight = -1

        self.c[self.number_of_connections].weight = weight
        self.number_of_connections += 1
        self.c[self.number_of_connections].from_neuron_id = -1

    def loadDNA(self, filename):
        pass

    def saveDNA(self, filename):
        pass

    def neuronIdToDNAIndex(self, neuron_id) -> int:
        i = 0
        while True:
            if self.n[i].neuron_id < 0:
                break

            if self.n[i].neuron_id == neuron_id:
                return i

            i += 1

        print(f"ERROR: Neuron with id {neuron_id} not found")
        sys.exit(1)
        return -1

    def smallestFreeId(self) -> int:
        id_found = False
        i = 0
        while True:
            if self.n[i].neuron_id < 0:
                break

            id_found = False

            j = 0
            while True:
                if self.n[j].neuron_id < 0 or id_found:
                    break

                if self.n[j].neuron_id == i:
                    id_found = True

            if not id_found:
                return i

            i += 1

        return i

    def updatePrimerList(self):
        self.number_of_primers = 0
        i = 0
        while True:
            if self.n[i].neuron_id < 0:
                break

            cn_id = self.n[i].neuron_id
            is_primer = True

            j = 0
            while True:
                if self.c[j].from_neuron_id < 0 or not is_primer:
                    break

                source = self.c[j].from_neuron_id

                if self.c[j].to_neuron_id != cn_id or source == cn_id:
                    continue

                dna_index = self.neuronIdToDNAIndex(source)
                if self.n[dna_index].type == NeuronTypes.CONTROL:
                    is_primer = False

                j += 1

            if is_primer:
                self.primer_list[self.number_of_primers] = cn_id
                self.number_of_primers += 1

            i += 1

        self.primer_list[self.number_of_primers] = -1

    def clearMemory(self):
        # for i in range(self.number_of_neurons):
        self.is_fired = [False] * self.number_of_neurons
        self.previous_neuron_state = [0.0] * self.number_of_neurons
        self.neuron_state = [0.0] * self.number_of_neurons
        self.internal_neuron_state = [0.0] * self.number_of_neurons
        self.neuron_excitation = [0.0] * self.number_of_neurons

        self.previous_connection_state = [
            ConnectionTypes.Recurrent
        ] * self.number_of_connections
        self.connection_state = [
            ConnectionTypes.Recurrent
        ] * self.number_of_connections

    def execute(
        self, neuron_index, ignore_if_all_recurrent=False
    ) -> ParameterType:
        pass

    def processPrimers(self):
        i = 0
        while True:
            if self.primer_list[i] == -1:
                break
            neuron_id = self.primer_list[i]
            index = self.neuronIdToDNAIndex(neuron_id)
            self.execute(index)
            i += 1

        i = 0
        while True:
            if self.primer_list[i] == -1:
                break
            neuron_id = self.primer_list[i]
            index = self.neuronIdToDNAIndex(neuron_id)
            self.is_fired[index] = True

            i += 1

        i = 0
        while True:
            if self.primer_list[i] == -1:
                break

            control_id = self.primer_list[i]
            control_index = self.neuronIdToDNAIndex(control_id)

            j = 0
            while True:
                if self.c[j].from_neuron_id < 0:
                    break

                if self.c[j].from_neuron_id != control_id:
                    continue

                destination = self.c[j].to_neuron_id
                destination_index = self.neuronIdToDNAIndex(destination)
                modulator = self.c[j].neuro_modulation
                ns = ParameterType(self.neuron_state[control_index])
                cw = ParameterType(self.c[j].weight)

                if modulator < 0:
                    if CONTINUOUS_PARAM:
                        self.neuron_excitation[destination_index] += cw * ns
                    else:
                        self.neuron_excitation[destination_index] |= cw & ns
                else:
                    modulator_index = self.neuronIdToDNAIndex(modulator)
                    modulator_input = ParameterType(0)
                    neM = self.neuron_excitation[modulator_index]
                    if (CONTINUOUS_PARAM and neM >= EXCITATION_THRESHOLD) or (
                        not CONTINUOUS_PARAM and neM < EXCITATION_THRESHOLD
                    ):
                        if self.is_fired[modulator_index]:
                            modulator_input = self.neuron_state[
                                modulator_index
                            ]
                        else:
                            modulator_input = self.previous_neuron_state[
                                modulator_index
                            ]

                    if CONTINUOUS_PARAM:
                        self.neuron_excitation[destination_index] += (
                            modulator_input * ns
                        )
                    else:
                        neD = ParameterType(
                            self.neuron_excitation[destination_index]
                        )
                        self.neuron_excitation[destination_index] = (
                            modulator_input & ns
                        ) | neD

                j += 1

            i += 1

    def processControlNeurons(self):
        pass

    def processInputNeurons(self):
        pass

    def processRemainingNeurons(self):
        pass

    def removeConnection(self, index):
        if self.c[index].from_neuron_id == -1:
            print("No connections to delete")
            return

        # move from last to this position
        c_last = self.c[self.number_of_connections - 1]
        self.c[index].from_neuron_id = c_last.from_neuron_id
        self.c[index].to_neuron_id = c_last.to_neuron_id
        self.c[index].neuro_modulation = c_last.neuro_modulation
        self.c[index].weight = c_last.weight

        self.number_of_connections -= 1
        self.c[self.number_of_connections].from_neuron_id = -1

        print(
            "Just making sure at index, from_neuron_id:"
            f" {self.c[index].from_neuron_id} should not be -1"
        )

    def createFixedInterfaceNeurons(self, number_of_inputs, number_of_outputs):
        for i in range(number_of_inputs):
            new_index = self.number_of_neurons
            self.number_of_neurons += 1

            new_id = self.smallestFreeId()

            if self.max_neuron_id < new_id:
                max_neuron_id = new_id

            if self.allocated_space <= self.number_of_neurons + 1:
                self.reallocEverything()

            self.n[new_index] = neuron(
                new_id, FiringRate.LEVEL1, NeuronTypes.INPUT_IDENTITY, i
            )
            self.n[self.number_of_neurons].id = -1


        for i in range(number_of_outputs):
            new_index = self.number_of_neurons
            self.number_of_neurons += 1

            new_id = self.smallestFreeId()

            if self.max_neuron_id < new_id:
                self.max_neuron_id = new_id

            if self.allocated_space <= self.number_of_neurons + 1:
                self.reallocEverything()
            
            self.n[new_index] = neuron(
                new_id, FiringRate.LEVEL1, NeuronTypes.OUTPUT_IDENTITY, i
            )
            self.n[self.number_of_neurons].id = -1

    def reallocEverything(self, given_allocated_space=None):
        if given_allocated_space is None:
            given_allocated_space = self.allocated_space * 2

        previous_space = self.allocated_space
        self.allocated_space = given_allocated_space
        space_diff = given_allocated_space - previous_space
        assert space_diff > 0

        self.n += [neuron()] * space_diff
        self.c += [connection()] * space_diff

        self.internal_neuron_state += [0] * space_diff
        self.neuron_state += [0] * space_diff
        self.previous_neuron_state += [0] * space_diff

        self.neuron_excitation += [0] * space_diff
        self.is_fired += [False] * space_diff
        self.connection_state += [ConnectionTypes.Recurrent] * space_diff
        self.previous_connection_state += [
            ConnectionTypes.Recurrent
        ] * space_diff
        self.primer_list += [0] * space_diff
