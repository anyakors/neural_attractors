"""
Feedforward neural network with feedback that produces attractors.
"""

import numpy as np
from typing import Tuple, List, Optional


class NeuralAttractor:
    """
    A feedforward neural network with feedback that produces chaotic attractors.
    
    The network has:
    - D inputs (dimension of the input vector)
    - N neurons in the hidden layer
    - 1 output, fed back to the input
    
    Parameters:
    -----------
    N : int
        Number of neurons in the hidden layer
    D : int
        Dimension of the input vector
    s : float
        Scaling factor for the output
    seed : Optional[int]
        Random seed for reproducibility
    """
    
    def __init__(self, N: int = 4, D: int = 16, s: float = 0.75, seed: Optional[int] = None):
        self.N = N
        self.D = D
        self.s = s
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and biases
        self.w = 2.0 * np.random.random((N, D)) - 1.0  # Uniform in [-1, 1]
        self.b = s * np.random.random(N)  # Uniform in [0, s]
        
        # Initialize state vectors
        self.x = np.zeros(N)  # Neuron outputs
        self.y = np.zeros(D)  # Input vector
        
    def reset(self, init_value: float = 0.001):
        """Reset the network state to initial conditions."""
        self.x = np.ones(self.N) * init_value
        self.y = np.zeros(self.D)
        
    def iterate(self) -> np.ndarray:
        """
        Perform one iteration of the network and return the neuron outputs.
        
        Returns:
        --------
        np.ndarray
            The state of the neurons after the iteration
        """
        # Calculate the output y0
        y0 = np.sum(self.b * self.x)
        
        # Shift the input vector
        self.y[1:] = self.y[:-1]
        self.y[0] = y0
        
        # Calculate the neuron inputs and apply activation function
        for i in range(self.N):
            u = np.sum(self.w[i] * self.y)
            self.x[i] = np.tanh(u)
            
        return self.x.copy()
    
    def generate_trajectory(self, tmax: int, discard: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a trajectory of the network for tmax iterations.
        
        Parameters:
        -----------
        tmax : int
            Number of iterations
        discard : int
            Number of initial iterations to discard (transient)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Two arrays containing the first two neuron outputs
            (or any two selected neurons)
        """
        self.reset()
        
        # Discard initial transient
        for _ in range(discard):
            self.iterate()
        
        # Collect trajectory
        x1_traj = np.zeros(tmax)
        x2_traj = np.zeros(tmax)
        
        for t in range(tmax):
            x = self.iterate()
            x1_traj[t] = x[0]
            x2_traj[t] = x[1]
            
        return x1_traj, x2_traj
