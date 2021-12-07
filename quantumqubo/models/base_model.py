from functools import lru_cache
import numpy as np
import torch

from quantumqubo.tbi import TBISampler

import time


class ExperimentModel:
    """A simple model for an experiment with variational parameters to be learned to directly optimise some expectation value

    This class calculates the results, as well as derivatives wrt to the BS angles to be used by the learning algorithm.

    Methods:
        forward(x, samples_per_point=1): for input tensor x, returns tensor of readout function
        calculate_readout(theta_angles_tensor, samples_per_point=1): returns expectation value defined by the readout function
        estimate_theta_derivatives(theta_angles, samples_per_point=1): returns tensor of derivatives of theta_angles wrt expectation of decoder
    """


    def __init__(self,
        readout_function
    ):
        """
        Args:
            readout_function (function): function that calculates some expectation value of the measurement results
        """

        self.tbi = TBISampler()
        self.readout_function = readout_function


    def forward(self, input_state, theta_angles, samples_per_point=1):
        """Returns the values of the readout function for the specified input state and beam splitter angles

        Args:
            input_state (tuple): input state to be used for the TBI simulation, for example (1,1,0) for |1,1,0>
            samples_per_point (int, optional): Number of samples used to estimate expectation value. Defaults to 1.

        Returns:
            tensor: torch tensor of expectation values, of shape (batch_size, 1)
        """

        readout_tensor = self.calculate_readout(input_state, theta_angles, samples_per_point)
        return readout_tensor


    def calculate_readout(self, input_state, theta_angles_tensor, samples_per_point=1):
        """Returns an expectation value determined by readout function for the input angles

        Args:
            input_state (tuple): input state to be used for the TBI simulation, for example (1,1,0) for |1,1,0>
            theta_angles_tensor (tensor): pytorch tensor of shape (batch_size, n_angles)
            samples_per_point (int, optional): Number of samples used to estimate expectation value. Defaults to 1.

        Returns:
            tensor: tensor of expectation values, of size batch_size
        """

        readouts = []
        for theta_angles in theta_angles_tensor:
            state_counts = self.tbi.sample(
                input_state,
                theta_angles.detach().numpy(),
                n_samples=samples_per_point
                )
            readout = self.readout_function(state_counts)
            readouts.append(readout)
            
        return torch.stack(readouts)


    @torch.no_grad()
    def estimate_theta_derivatives(
        self,
        input_state,
        theta_angles_tensor,
        samples_per_point=1,
        parameter_shift=np.pi/6
        ):
        """Returns derivatives of beam splitter angles wrt the objective function, using the parameter shift rule.

        Note: the use of the parameter shift rule implies estimating 2N parameters, where N is the number of
        theta angles. Other schemes exist to approximate these derivatives using many fewer estimations.

        Args:
            input_state (tuple): input state to be used for the TBI simulation, for example (1,1,0) for |1,1,0>
            theta_angles_tensor (tensor): pytorch tensor of shape (batch_size, n_angles)
            samples_per_point (int, optional): Number of samples used to estimate expectation value. Defaults to 1.
            parameter_shift (float, optional): Parameter shift to use with the parameter shift rule

        Returns:
            tensor: tensor of derivatives, of shape (batch_size, n_angles)
        """

        # Loop through each array of beam splitter angles in the batch
        derivatives_tensor = torch.zeros((theta_angles_tensor.shape[0], theta_angles_tensor.shape[1]))
        for i, theta_angles in enumerate(theta_angles_tensor):

            # For each beam splitter angle, we require 2 measurements
            with torch.no_grad():
                for j in range(len(theta_angles)):
                    upshifted_theta = theta_angles.clone()
                    upshifted_theta[j] = upshifted_theta[j] + parameter_shift

                    downshifted_theta = theta_angles.clone()
                    downshifted_theta[j] = downshifted_theta[j] - parameter_shift

                    upshifted_expectation = self.calculate_readout(
                        input_state,
                        upshifted_theta.unsqueeze(0),
                        samples_per_point
                    )
                    downshifted_expectation = self.calculate_readout(
                        input_state,
                        downshifted_theta.unsqueeze(0),
                        samples_per_point
                    )

                    # Parameter shift rule for the gradient of the readout wrt to the theta values
                    readout_wrt_theta_gradient = (upshifted_expectation[0]-downshifted_expectation[0])/np.sin(2*parameter_shift)

                    derivatives_tensor[i,j] = readout_wrt_theta_gradient

        return derivatives_tensor
