"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network.

        :param observation: observation(s) to query the policy
        :return: a distribution from which actions can be sampled
        """
        mean = self.mean_net(observation)
        std = torch.exp(self.logstd)  # Convert log standard deviation to standard deviation
        return distributions.Normal(mean, std)

    def update(self, observations, actions):
        """
        Updates/trains the policy.

        :param observations: observation(s) to query the policy, as a NumPy array
        :param actions: actions we want the policy to imitate, as a NumPy array
        :return: a dictionary containing the training loss
        """
        # Convert observations and actions from NumPy arrays to PyTorch tensors
        observations = torch.tensor(observations, dtype=torch.float32, device=ptu.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=ptu.device)

        # Forward pass to get the distribution of actions
        action_distributions = self.forward(observations)
        
        # Calculate the log probability of the actual actions under the distribution
        log_probs = action_distributions.log_prob(actions)
        
        # Negative log likelihood loss
        loss = -log_probs.mean()
        
        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss.detach()),
        }
        
    def act(self, observation):
        """
        Decide an action based on the policy for a given observation by using the forward method.

        :param observation: A single observation from the environment, should be a NumPy array
        :return: A sampled action as a NumPy array
        """
        # Convert observation to a PyTorch tensor and add a batch dimension
        observation = torch.tensor(observation[None, :], dtype=torch.float32, device=ptu.device)

        # Use the forward method to get the action distribution
        action_distribution = self.forward(observation)

        # Sample an action from the distribution
        action = action_distribution.sample()

        # Remove the batch dimension and convert the action back to a NumPy array
        return action.cpu().numpy().flatten()

