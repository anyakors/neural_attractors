"""
Utility functions for the neural attractor project.
"""

import os
import numpy as np
from typing import Tuple, Optional, List, Callable
import matplotlib.pyplot as plt


def ensure_directory_exists(path: str):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Parameters:
    -----------
    path : str
        Path to the directory
    """
    os.makedirs(path, exist_ok=True)


def generate_filename(base_dir: str, prefix: str, N: int, D: int, s: float, 
                     ext: str = 'png') -> str:
    """
    Generate a filename based on parameters.
    
    Parameters:
    -----------
    base_dir : str
        Base directory
    prefix : str
        Prefix for the filename
    N : int
        Number of neurons
    D : int
        Dimension of input vector
    s : float
        Scaling factor
    ext : str
        File extension
        
    Returns:
    --------
    str
        Generated filename
    """
    # Ensure directory exists
    ensure_directory_exists(base_dir)
    
    # Generate filename
    filename = f"{prefix}_N{N}_D{D}_s{s:.2f}.{ext}"
    
    return os.path.join(base_dir, filename)


def save_trajectory_data(x1: np.ndarray, x2: np.ndarray, 
                        output_dir: str, prefix: str,
                        N: int, D: int, s: float):
    """
    Save trajectory data to files.
    
    Parameters:
    -----------
    x1 : np.ndarray
        First trajectory component
    x2 : np.ndarray
        Second trajectory component
    output_dir : str
        Output directory
    prefix : str
        Prefix for filenames
    N : int
        Number of neurons
    D : int
        Dimension of input vector
    s : float
        Scaling factor
    """
    # Ensure directory exists
    ensure_directory_exists(output_dir)
    
    # Generate filenames
    x1_file = generate_filename(output_dir, f"{prefix}_x1", N, D, s, "npy")
    x2_file = generate_filename(output_dir, f"{prefix}_x2", N, D, s, "npy")
    
    # Save data
    np.save(x1_file, x1)
    np.save(x2_file, x2)
    
    return x1_file, x2_file


def load_trajectory_data(x1_file: str, x2_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectory data from files.
    
    Parameters:
    -----------
    x1_file : str
        Path to first trajectory component
    x2_file : str
        Path to second trajectory component
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Loaded trajectory data
    """
    x1 = np.load(x1_file)
    x2 = np.load(x2_file)
    
    return x1, x2


def check_if_interesting(x1: np.ndarray, x2: np.ndarray, 
                        grid_size: int = 100, threshold: float = 0.1) -> bool:
    """
    Check if the attractor is interesting by examining the fraction of phase space covered.
    
    Parameters:
    -----------
    x1 : np.ndarray
        First trajectory component
    x2 : np.ndarray
        Second trajectory component
    grid_size : int
        Grid size for binning
    threshold : float
        Threshold for the fraction of covered cells
        
    Returns:
    --------
    bool
        True if the attractor is interesting, False otherwise
    """
    # Create a grid and count occupied cells
    hist, _, _ = np.histogram2d(x1, x2, bins=grid_size, range=[[-1, 1], [-1, 1]])
    
    # Count the number of occupied cells
    occupied_cells = np.sum(hist > 0)
    
    # Calculate the fraction of occupied cells
    fraction = occupied_cells / (grid_size * grid_size)
    
    # Check if the fraction is above the threshold
    return fraction > threshold and fraction < 0.9


def compute_lyapunov_exponent(network_factory: Callable, 
                             num_iterations: int = 10000, 
                             num_discard: int = 1000,
                             epsilon: float = 1e-8,
                             num_exponents: int = 1) -> Tuple[List[float], np.ndarray]:
    """
    Compute the Lyapunov exponent(s) for a neural network attractor.
    
    Parameters:
    -----------
    network_factory : Callable
        A function that returns a new instance of the neural network
    num_iterations : int
        Number of iterations for Lyapunov exponent calculation
    num_discard : int
        Number of initial iterations to discard (transient)
    epsilon : float
        Initial perturbation size for nearby trajectories
    num_exponents : int
        Number of Lyapunov exponents to compute (defaults to 1, the largest)
        
    Returns:
    --------
    Tuple[List[float], np.ndarray]
        The final Lyapunov exponent(s) and the history of Lyapunov exponent(s) over iterations
    """
    # Initialize the reference network
    network = network_factory()
    
    # Initialize arrays for Lyapunov exponents
    lyapunov_history = np.zeros((num_iterations, num_exponents))
    lyapunov_sums = np.zeros(num_exponents)
    
    # Discard transient behavior
    for _ in range(num_discard):
        network.iterate()
    
    # For each iteration
    for i in range(num_iterations):
        # Save the current state
        reference_state_x = network.x.copy()
        reference_state_y = network.y.copy()
        
        # Perform one iteration of the reference trajectory
        reference_next_state = network.iterate()
        
        # For each Lyapunov exponent
        for j in range(num_exponents):
            # Create a perturbed network
            perturbed_network = network_factory()
            
            # Initialize with the same state as the reference network
            perturbed_network.x = reference_state_x.copy()
            perturbed_network.y = reference_state_y.copy()
            
            # Apply a small perturbation in the j-th dimension
            if j < len(perturbed_network.x):
                perturbed_network.x[j] += epsilon
            else:
                idx = j - len(perturbed_network.x)
                if idx < len(perturbed_network.y):
                    perturbed_network.y[idx] += epsilon
            
            # Evolve the perturbed state one step
            perturbed_next_state = perturbed_network.iterate()
            
            # Calculate the distance between the reference and perturbed trajectories
            distance = np.linalg.norm(reference_next_state - perturbed_next_state)
            
            # Calculate the local Lyapunov exponent
            if distance > 0:
                local_lyapunov = np.log(distance / epsilon)
            else:
                local_lyapunov = -np.inf  # Trajectories converged
            
            # Update the running sum
            lyapunov_sums[j] += local_lyapunov
            
            # Store the current estimate
            lyapunov_history[i, j] = lyapunov_sums[j] / (i + 1)
    
    # Calculate the final Lyapunov exponents
    lyapunov_exponents = lyapunov_sums / num_iterations
    
    return lyapunov_exponents.tolist(), lyapunov_history


def plot_lyapunov_exponent(lyapunov_history: np.ndarray,
                          figsize: Tuple[float, float] = (10, 6),
                          output_path: Optional[str] = None,
                          dpi: int = 300,
                          show: bool = True,
                          title: str = "Lyapunov Exponent Convergence",
                          parameters: Optional[dict] = None):
    """
    Plot the convergence of the Lyapunov exponent(s).
    
    Parameters:
    -----------
    lyapunov_history : np.ndarray
        History of Lyapunov exponent estimates
    figsize : Tuple[float, float]
        Size of the figure in inches
    output_path : Optional[str]
        Path to save the figure
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the figure
    title : str
        Title for the plot
    parameters : Optional[dict]
        Dictionary of parameters to display in the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot Lyapunov exponent history
    num_exponents = lyapunov_history.shape[1]
    x = np.arange(1, len(lyapunov_history) + 1)
    
    for i in range(num_exponents):
        label = f"λ{i+1}" if num_exponents > 1 else "Largest Lyapunov Exponent"
        ax.plot(x, lyapunov_history[:, i], label=label)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Lyapunov Exponent')
    
    # Create title with parameters if provided
    if parameters:
        param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        full_title = f"{title}\n({param_str})"
    else:
        full_title = title
    
    ax.set_title(full_title)
    
    # Add legend
    if num_exponents > 1:
        ax.legend()
    
    # Add text with final Lyapunov exponent values
    final_values = lyapunov_history[-1, :]
    text_str = "\n".join([f"λ{i+1} = {val:.6f}" for i, val in enumerate(final_values)])
    
    # Determine if the system is chaotic
    is_chaotic = any(val > 0 for val in final_values)
    behavior_str = "Chaotic" if is_chaotic else "Regular"
    text_str += f"\nBehavior: {behavior_str}"
    
    # Position the text
    ax.text(0.02, 0.02, text_str, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
