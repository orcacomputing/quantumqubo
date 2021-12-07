import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from quantumqubo.models import ExperimentModel
from quantumqubo.qubo.utils_qubo import parity_mapping


class QuboOneConfiguration():
    """A class used to run a QUBO optimisation loop in one given configuration of photon numbers and modes

    To achieve roughly equal probabilities of generating arbitrary bit strings, a full QUBO optimisation requires 4 configurations.

    Methods:
        readout(measurements): function used to convert the raw measurement results to a QUBO energy
        train(args): runs the training loop
        get_final_states(n_samples): samples from the output state distribution using current beam splitter settings
        get_final_bit_strings(): samples bit strings produced by the PT-Series using current beam splitter settings
    """

    def __init__(
            self,
            input_state,
            optimization_function,
            n_fixed_beam_splitter,
            parity,
    ):
        """
        Args:
            input_state (tuple): input state to be used, for example (1,1,0) for input state |110>
            optimization_function (function): function to be optimised
            n_fixed_beam_splitter (0 or 1): number of beam splitters at the start of the simulation that are fixed and fully reflective
            parity (0 or 1): in the parity mapping used to map measurement results to bit strings, selects whether to use even or odd
        """

        self.input_state = input_state
        self.optimization_function = optimization_function
        self.n_fixed_beam_splitter = n_fixed_beam_splitter
        self.parity = parity

        self.metrics = []
        self.E_min_encountered = 1000
        self.config_min_encountered = 1000

        # Define encoder
        n_variable_beam_splitter = len(input_state) - n_fixed_beam_splitter

        self.theta_angles = torch.nn.Parameter(torch.randn(1, n_variable_beam_splitter, requires_grad=True))

        # The experiment model defines the way this algorithm interacts with the TBI
        self.experiment_model = ExperimentModel(self.readout)

        self.encoder_optimizer = torch.optim.Adam([self.theta_angles])


    def readout(self, measurements):
        """
        Args:
            measurements (dict): dict of state:counts

        Returns:
            tensor: 1d tensor with values of the correlators
        """

        total_counts = sum(counts for counts in measurements.values())

        E = 0
        for state, counts in measurements.items():
            bit_string = parity_mapping(state[self.n_fixed_beam_splitter:], self.parity)
            E_state = self.optimization_function(bit_string)
            E += E_state*counts/total_counts

            if E_state < self.E_min_encountered:
                self.E_min_encountered = E_state
                self.config_min_encountered = tuple(bit_string)
        return torch.FloatTensor([E])


    def train(
            self,
            learning_rate=5e-2,
            updates=100,
            samples_per_point=500,
            parameter_shift=np.pi/6,
            print_frequency=10,
            verbose=True
        ):
        """Runs the training loop

        Args:
            learning_rate (float, optional): learning rate for the pytorch optimiser in the training loop.
            updates (int, optional): Number of parameter updates to perform.
            samples_per_point (int, optional): Number of samples used to determine expectation values.
            parameter_shift (float, optional): Parameter shift to use with the parameter shift rule
            print_frequency (int, optional): How often training information is printed
            verbose (bool, optional): whether to print information about epochs
        """

        for param in self.encoder_optimizer.param_groups:
            param['lr'] = learning_rate

        X = torch.ones((1, 1))

        loss = []

        for i in range(updates):

            if self.n_fixed_beam_splitter == 1:
                theta_angles = torch.cat((torch.FloatTensor([0]).unsqueeze(1), self.theta_angles), dim=1)
            else:
                theta_angles = self.theta_angles

            # Calculate current fitness
            with torch.no_grad():
                readout_tensor = self.experiment_model.calculate_readout(
                    self.input_state,
                    theta_angles,
                    samples_per_point=samples_per_point
                    )
                current_fitness = readout_tensor.sum()

            # Calculate derivatives of theta angles wrt to the loss
            theta_derivatives = self.experiment_model.estimate_theta_derivatives(
                self.input_state,
                theta_angles,
                samples_per_point=samples_per_point,
                parameter_shift=parameter_shift
                )

            # Training the encoder using the previously calculated derivatives of theta angles wrt the loss
            encoder_loss = torch.sum(torch.FloatTensor(theta_derivatives)*theta_angles)
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()

            # Print results once in a while
            if i%print_frequency == print_frequency-1 and verbose:
                loss.append(current_fitness.numpy())
                print("Training loop {}: loss is {:.2f}".format(i+1, current_fitness.detach().numpy()))

            self.metrics.append(current_fitness.detach().numpy())


class QUBO():
    """
    A class that is used to run a QUBO problem on a PT-Series.

    To ensure that all bit strings have a roughly equal probability of being generated, we run the QUBO optimisation loop in 4 different configurations

    Important attributes:
        energies: dictionary with keys 'config1', ..., 'config4' that contains the energy of each configuration for each step of the gradient descent.
        res: dictionary with state:value, where each entry is the minimum found by at least one of the 4 configurations

    Methods:
        train(M, optimization_function): implement the optimization routine for 4 different configurations.
     """

    def __init__(
            self,
            M,
            optimization_function,
    ):
        """
        Args:
            M (int): Dimension of the QUBO problem
            optimization_function (function): function to be optimised
        """
        self.M = M
        self.optimization_function = optimization_function

        self.energies = {}
        self.res = {}

    def train(self, 
            learning_rate=5e-2,
            updates=100,
            samples_per_point=500,
            parameter_shift=np.pi/6,
            print_frequency=10,
            verbose=True
        ):
        """
        Args:
            learning_rate (float, optional): learning rate to be used in the optimisation loop
            updates (int, optional): number of parameter updates
            samples_per_point (int, optional): number of samples used to estimate expectation values
            parameter_shift (float, optional): Parameter shift to use with the parameter shift rule
            print_frequency (int, optional): how often to print training information
            verbose (bool, optional): whether to print information
        """

        #M modes, M photons
        #We send M photons with the first beam splitter completely reflective.
        #The first mode is thus always 0. By removing it, we get M output modes with M photons.
        print('Configuration 1 - M modes, M photons, parity 0')
        input_state = (1,)*self.M
        qubo1 = QuboOneConfiguration(
            input_state,
            self.optimization_function,
            n_fixed_beam_splitter=1,
            parity=0
            )
        qubo1.train(
            learning_rate=learning_rate,
            updates=updates,
            samples_per_point=samples_per_point,
            parameter_shift=parameter_shift,
            print_frequency=print_frequency,
            verbose=verbose,
            )
        self.energies['config1'] = qubo1.metrics
        self.res[qubo1.config_min_encountered] = qubo1.E_min_encountered

        print('\nConfiguration 2 - M modes, M photons, parity 1')
        qubo2 = QuboOneConfiguration(
            input_state,
            self.optimization_function,
            n_fixed_beam_splitter=1,
            parity=1
            )
        qubo2.train(learning_rate=learning_rate,
            updates=updates,
            samples_per_point=samples_per_point,
            parameter_shift=parameter_shift,
            print_frequency=print_frequency,
            verbose=verbose
            )
        self.energies['config2'] = qubo2.metrics
        self.res[qubo2.config_min_encountered] = qubo2.E_min_encountered

        # M modes, M-1 photons
        # We send M-1 photons without any fixed beam splitter.
        # We thus get M output modes with M-1 photons
        print('\nConfiguration 3 - M modes, M-1 photons, parity 0')
        input_state = (1,) * (self.M-1)
        qubo3 = QuboOneConfiguration(
            input_state,
            self.optimization_function,
            n_fixed_beam_splitter=0,
            parity=0
            )
        qubo3.train(
            learning_rate=learning_rate,
            updates=updates,
            samples_per_point=samples_per_point,
            parameter_shift=parameter_shift,
            print_frequency=print_frequency,
            verbose=verbose
            )
        self.energies['config3'] = qubo3.metrics
        self.res[qubo3.config_min_encountered] = qubo3.E_min_encountered

        print('\nConfiguration 4 - M modes, M-1 photons, parity 1')
        qubo4 = QuboOneConfiguration(
            input_state,
            self.optimization_function,
            n_fixed_beam_splitter=0,
            parity=1
            )
        qubo4.train(
            learning_rate=learning_rate, 
            updates=updates, 
            samples_per_point=samples_per_point, 
            parameter_shift=parameter_shift,
            print_frequency=print_frequency, 
            verbose=verbose
            )
        self.energies['config4'] = qubo4.metrics
        self.res[qubo4.config_min_encountered] = qubo4.E_min_encountered


if __name__ == '__main__':


    M = 4
    Q = (np.random.rand(M, M) - 0.5) * 2

    def qubo_function(vect):
        return np.dot(vect, np.dot(Q, vect))

    def mobeus_function(vect):
        return np.dot(vect, np.dot(Q, vect))

    qubo = QUBO(
        M,
        qubo_function
    )
    qubo.train(
        learning_rate=1e-1,
        updates=20,
        samples_per_point=100,
        print_frequency=5
        )

    print('res = ', qubo.res)

    plt.figure(figsize=(6, 4))
    plt.plot(qubo.energies['config1'], label='config1')
    plt.plot(qubo.energies['config2'], label='config2')
    plt.plot(qubo.energies['config3'], label='config3')
    plt.plot(qubo.energies['config4'], label='config4')
    plt.ylabel('energy')
    plt.xlabel('updates')
    plt.legend()
    plt.show()
