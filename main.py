#!/usr/bin/env python

"""
Neural Network Attractors Generator

This script generates and visualizes attractors created by simple
feedforward neural networks with feedback, based on the paper
"Artificial Neural Net Attractors" by J.C. Sprott.

Usage:
    python main.py [options]
    
Examples:
    # Generate and visualize an attractor with default parameters
    python main.py
    
    # Generate an attractor with specific parameters
    python main.py --N 4 --D 32 --s 0.5 --tmax 100000
    
    # Generate multiple attractors and save only interesting ones
    python main.py --count 10 --auto-filter --save-data
    
    # Generate an attractor with a specific random seed
    python main.py --seed 42
    
    # Visualize a previously generated attractor
    python main.py --load-data path/to/x1_file.npy path/to/x2_file.npy
    
    # Generate an attractor and show time series and scatter plots
    python main.py --plot-time-series --plot-scatter --show
    
    # Calculate and plot the Lyapunov exponent for the attractor
    python main.py --plot-lyapunov
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# Add parent directory to path if running as a script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    __package__ = "neural_attractors"
    
    # Try to import the package, if it fails, we're running as a script
    try:
        from neural_attractors.network import NeuralAttractor
    except ImportError:
        # We may be running the script directly from the project root
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        __package__ = None

# Import the package modules
try:
    # Try to import as a package if installed
    from neural_attractors.network import NeuralAttractor
    from neural_attractors.visualization import (
        plot_attractor_trajectory, 
        plot_attractor_time_series, 
        plot_attractor_phase_scatter
    )
    from neural_attractors.utils import (
        ensure_directory_exists,
        save_trajectory_data,
        load_trajectory_data,
        check_if_interesting,
        compute_lyapunov_exponent,
        plot_lyapunov_exponent
    )
except ImportError:
    # If that fails, import directly
    from network import NeuralAttractor
    from visualization import (
        plot_attractor_trajectory, 
        plot_attractor_time_series, 
        plot_attractor_phase_scatter
    )
    from utils import (
        ensure_directory_exists,
        save_trajectory_data,
        load_trajectory_data,
        check_if_interesting,
        compute_lyapunov_exponent,
        plot_lyapunov_exponent
    )


def calculate_and_plot_lyapunov(args, seed, N, D, s, prefix):
    """
    Calculate and plot the Lyapunov exponent for a neural network attractor.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    seed : int
        Random seed for the neural network
    N : int
        Number of neurons
    D : int
        Dimension of input vector
    s : float
        Scaling factor
    prefix : str
        Prefix for output files
    """
    print(f"Calculating Lyapunov exponent with {args.lyapunov_iterations} iterations...")
    
    # Define a factory function to create identical networks
    def network_factory():
        return NeuralAttractor(N, D, s, seed=seed)
    
    # Calculate the Lyapunov exponent
    start_time = time.time()
    lyapunov_exponents, lyapunov_history = compute_lyapunov_exponent(
        network_factory,
        num_iterations=args.lyapunov_iterations,
        num_discard=args.discard,
        num_exponents=1
    )
    elapsed_time = time.time() - start_time
    print(f"Lyapunov calculation completed in {elapsed_time:.2f} seconds")
    
    # Print the result
    is_chaotic = lyapunov_exponents[0] > 0
    behavior = "chaotic" if is_chaotic else "regular"
    print(f"Largest Lyapunov exponent: {lyapunov_exponents[0]:.6f} ({behavior} behavior)")
    
    # Plot the Lyapunov exponent
    output_path = os.path.join(args.output_dir, f"{prefix}_lyapunov.png") if not args.show else None
    parameters = {
        'N': N,
        'D': D,
        's': s,
        'seed': seed
    }
    
    plot_lyapunov_exponent(
        lyapunov_history,
        figsize=(10, 6),
        output_path=output_path,
        dpi=args.dpi,
        show=args.show,
        parameters=parameters
    )
    
    if output_path:
        print(f"Lyapunov exponent plot saved to {output_path}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Neural Network Attractors Generator")
    
    # Neural network parameters
    nn_group = parser.add_argument_group('Neural Network Parameters')
    nn_group.add_argument("--N", type=int, default=4,
                        help="Number of neurons in the hidden layer (default: 4)")
    nn_group.add_argument("--D", type=int, default=16,
                        help="Dimension of the input vector (default: 16)")
    nn_group.add_argument("--s", type=float, default=0.75,
                        help="Scaling factor for the output (default: 0.75)")
    nn_group.add_argument("--tmax", type=int, default=100000,
                        help="Number of iterations (default: 100000)")
    nn_group.add_argument("--discard", type=int, default=1000,
                        help="Number of initial iterations to discard (default: 1000)")
    nn_group.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    
    # Visualization parameters
    vis_group = parser.add_argument_group('Visualization Parameters')
    vis_group.add_argument("--skip-value", type=int, default=16,
                        help="Number of points to skip for visualization (default: 16)")
    vis_group.add_argument("--figsize", type=float, nargs=2, default=(12, 12),
                        help="Figure size in inches (default: 12 12)")
    vis_group.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures (default: 300)")
    vis_group.add_argument("--cmap", type=str, default="Spectral",
                        help="Colormap for visualization (default: 'Spectral')")
    vis_group.add_argument("--linewidth", type=float, default=0.1,
                        help="Line width for visualization (default: 0.1)")
    vis_group.add_argument("--alpha", type=float, default=0.1,
                        help="Alpha (transparency) for visualization (default: 0.1)")
    vis_group.add_argument("--interpolate-steps", type=int, default=3,
                        help="Interpolation steps for visualization (default: 3)")
    
    # Output parameters
    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument("--output-dir", type=str, default="output",
                        help="Output directory (default: 'output')")
    output_group.add_argument("--prefix", type=str, default="attractor",
                        help="Prefix for output files (default: 'attractor')")
    output_group.add_argument("--save-data", action="store_true",
                        help="Save trajectory data to files")
    output_group.add_argument("--load-data", nargs=2, metavar=('X1_FILE', 'X2_FILE'),
                        help="Load trajectory data from files")
    
    # Batch generation parameters
    batch_group = parser.add_argument_group('Batch Generation Parameters')
    batch_group.add_argument("--count", type=int, default=1,
                        help="Number of attractors to generate (default: 1)")
    batch_group.add_argument("--auto-filter", action="store_true",
                        help="Filter out uninteresting attractors")
    
    # Plotting options
    plot_group = parser.add_argument_group('Plotting Options')
    plot_group.add_argument("--plot-trajectory", action="store_true", default=True,
                        help="Plot the attractor trajectory (default: True)")
    plot_group.add_argument("--plot-time-series", action="store_true",
                        help="Plot the time series (default: False)")
    plot_group.add_argument("--plot-scatter", action="store_true",
                        help="Plot the scatter plot (default: False)")
    plot_group.add_argument("--plot-lyapunov", action="store_true",
                        help="Calculate and plot the Lyapunov exponent (default: False)")
    plot_group.add_argument("--lyapunov-iterations", type=int, default=5000,
                        help="Number of iterations for Lyapunov exponent calculation (default: 5000)")
    plot_group.add_argument("--no-plot", action="store_true",
                        help="Do not plot anything, just generate data")
    plot_group.add_argument("--show", action="store_true",
                        help="Show plots (default: False)")
    
    return parser.parse_args()


def generate_and_plot_attractor(args):
    """
    Generate and plot a neural network attractor.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create output directory if it doesn't exist
    ensure_directory_exists(args.output_dir)
    
    # If loading data from files
    if args.load_data:
        print(f"Loading data from {args.load_data[0]} and {args.load_data[1]}...")
        x1, x2 = load_trajectory_data(args.load_data[0], args.load_data[1])
        N, D, s = 0, 0, 0.0  # Unknown parameters
        
        # Plot the attractor
        if not args.no_plot:
            plot_attractor(x1, x2, args, N, D, s, args.prefix)
            
            # Cannot calculate Lyapunov exponent for loaded data without network parameters
            if args.plot_lyapunov:
                print("Warning: Cannot calculate Lyapunov exponent for loaded data without network parameters.")
        return
    
    # Count how many attractors to generate
    count = max(1, args.count)
    
    # Loop until we generate the requested number of attractors
    successful_count = 0
    attempt_count = 0
    max_attempts = count * 10  # Limit the number of attempts
    
    while successful_count < count and attempt_count < max_attempts:
        attempt_count += 1
        
        # Create a new seed for each attempt if not specified
        current_seed = args.seed if args.seed is not None else np.random.randint(0, 2**32 - 1)
        
        # Create neural network
        nn = NeuralAttractor(args.N, args.D, args.s, seed=current_seed)
        
        # Generate trajectory
        print(f"Generating trajectory with N={args.N}, D={args.D}, s={args.s}, seed={current_seed}...")
        start_time = time.time()
        x1, x2 = nn.generate_trajectory(args.tmax, args.discard)
        elapsed_time = time.time() - start_time
        print(f"Generation completed in {elapsed_time:.2f} seconds")
        
        # Check if the attractor is interesting
        if args.auto_filter:
            is_interesting = check_if_interesting(x1, x2)
            print(f"Attractor {'is' if is_interesting else 'is not'} interesting")
            if not is_interesting:
                print(f"Attempt {attempt_count}, trying again...")
                continue
        
        # Increment successful count
        successful_count += 1
        
        # Set parameters for filenames
        N, D, s = args.N, args.D, args.s
        
        # Generate a unique prefix for this attractor
        current_prefix = f"{args.prefix}_{successful_count}" if count > 1 else args.prefix
        
        # Save trajectory data if requested
        if args.save_data:
            x1_file, x2_file = save_trajectory_data(
                x1, x2, args.output_dir, current_prefix, N, D, s
            )
            print(f"Saved trajectory data to {x1_file} and {x2_file}")
        
        # Calculate Lyapunov exponent if requested
        if args.plot_lyapunov and not args.no_plot:
            calculate_and_plot_lyapunov(args, current_seed, N, D, s, current_prefix)
        
        # Skip plotting if we're generating multiple attractors and not at the last one
        if successful_count < count and not args.show and not args.save_data:
            continue
        
        # Plot the attractor
        if not args.no_plot:
            plot_attractor(x1, x2, args, N, D, s, current_prefix)
    
    if successful_count == 0:
        print(f"Failed to generate any interesting attractors after {attempt_count} attempts.")
    else:
        print(f"Successfully generated {successful_count} attractors.")


def plot_attractor(x1, x2, args, N, D, s, prefix):
    """
    Plot an attractor.
    
    Parameters:
    -----------
    x1 : np.ndarray
        First trajectory component
    x2 : np.ndarray
        Second trajectory component
    args : argparse.Namespace
        Command-line arguments
    N : int
        Number of neurons
    D : int
        Dimension of input vector
    s : float
        Scaling factor
    prefix : str
        Prefix for output files
    """
    # Set up plotting parameters
    import matplotlib as mpl
    cmap = mpl.colormaps.get_cmap(args.cmap)
    
    # Plot trajectory if requested
    if args.plot_trajectory:
        output_path = os.path.join(args.output_dir, f"{prefix}_trajectory.png") if not args.show else None
        plot_attractor_trajectory(
            x1, x2,
            skip_value=args.skip_value,
            cmap=cmap,
            linewidth=args.linewidth,
            alpha=args.alpha,
            figsize=args.figsize,
            interpolate_steps=args.interpolate_steps,
            output_path=output_path,
            dpi=args.dpi,
            show=args.show
        )
        print(f"Trajectory plot saved to {output_path}" if output_path else "Trajectory plot displayed")
    
    # Plot time series if requested
    if args.plot_time_series:
        output_path = os.path.join(args.output_dir, f"{prefix}_time_series.png") if not args.show else None
        plot_attractor_time_series(
            x1,
            n_points=1000,
            figsize=(10, 4),
            output_path=output_path,
            dpi=args.dpi,
            show=args.show
        )
        print(f"Time series plot saved to {output_path}" if output_path else "Time series plot displayed")
    
    # Plot scatter plot if requested
    if args.plot_scatter:
        output_path = os.path.join(args.output_dir, f"{prefix}_scatter.png") if not args.show else None
        plot_attractor_phase_scatter(
            x1, x2,
            n_points=1000,
            cmap=cmap,
            figsize=(8, 8),
            output_path=output_path,
            dpi=args.dpi,
            show=args.show
        )
        print(f"Scatter plot saved to {output_path}" if output_path else "Scatter plot displayed")


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Check for invalid combinations
    if args.no_plot and args.show:
        print("Warning: --no-plot and --show are mutually exclusive. Ignoring --show.")
        args.show = False
        
    # Generate and plot attractor
    generate_and_plot_attractor(args)