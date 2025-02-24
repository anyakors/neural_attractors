<p align="center"><img width=50% src="https://github.com/anyakors/neural_attractors/blob/main/misc/attractor_candyshape.png"></p>

# Neural Attractors

This project generates and visualizes beautiful chaotic attractors using simple feedforward neural networks with feedback, based on the paper "Artificial Neural Net Attractors" by J.C. Sprott.

## Overview

The neural network architecture consists of:
- An input layer with D elements (embedding dimension)
- A hidden layer with N neurons
- A single output, which is fed back to the input

When the network is iterated, it produces fascinating patterns that can be visualized in various ways.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-attractors.git
   cd neural-attractors
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Generate and visualize an attractor with default parameters:

```bash
python main.py
```

### Customizing Neural Network Parameters

```bash
python main.py --N 4 --D 16 --s 0.75 --tmax 100000
```

- `--N`: Number of neurons in the hidden layer (default: 4)
- `--D`: Dimension of the input vector (default: 16)
- `--s`: Scaling factor for the output (default: 0.75)
- `--tmax`: Number of iterations (default: 100000)
- `--discard`: Number of initial iterations to discard (default: 1000)
- `--seed`: Random seed for reproducibility

### Visualization Options

```bash
python main.py --skip-value 8 --figsize 16 16 --cmap viridis --linewidth 0.2 --alpha 0.15
```

- `--skip-value`: Number of points to skip for visualization (default: 16)
- `--figsize`: Figure size in inches (default: 12 12)
- `--dpi`: DPI for saved figures (default: 300)
- `--cmap`: Colormap for visualization (default: 'Spectral')
- `--linewidth`: Line width for visualization (default: 0.1)
- `--alpha`: Alpha (transparency) for visualization (default: 0.1)
- `--interpolate-steps`: Interpolation steps for visualization (default: 3)

### Plotting Options

```bash
python main.py --plot-time-series --plot-scatter --plot-lyapunov --show
```

- `--plot-trajectory`: Plot the attractor trajectory (default: True)
- `--plot-time-series`: Plot the time series
- `--plot-scatter`: Plot a scatter plot of the attractor
- `--plot-lyapunov`: Calculate and plot the Lyapunov exponent
- `--lyapunov-iterations`: Number of iterations for Lyapunov exponent calculation (default: 5000)
- `--no-plot`: Don't create any plots
- `--show`: Display plots instead of saving them

### Output Options

```bash
python main.py --output-dir images --prefix my_attractor --save-data
```

- `--output-dir`: Output directory (default: 'output')
- `--prefix`: Prefix for output files (default: 'attractor')
- `--save-data`: Save trajectory data to files

### Batch Generation

Generate multiple attractors:

```bash
python main.py --count 10 --auto-filter --save-data
```

- `--count`: Number of attractors to generate (default: 1)
- `--auto-filter`: Filter out uninteresting attractors

### Loading Existing Data

```bash
python main.py --load-data output/attractor_x1.npy output/attractor_x2.npy
```

## Examples

### Generate a high-quality attractor

```bash
python main.py --N 4 --D 32 --s 0.5 --tmax 1000000 --discard 10000 --skip-value 8 --figsize 16 16 --dpi 600
```

### Calculate and plot the Lyapunov exponent

```bash
python main.py --plot-lyapunov --lyapunov-iterations 10000
```

### Generate multiple attractors and save interesting ones

```bash
python main.py --count 20 --auto-filter --save-data
```

### Create a series of attractors with varying parameters

```bash
for s in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    python main.py --s $s --prefix "attractor_s${s}" --save-data
done
```

### Analyze how Lyapunov exponents change with parameter s

```bash
for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4; do
    python main.py --s $s --prefix "lyapunov_s${s}" --plot-lyapunov --save-data
done
```

## Lyapunov Exponent

The Lyapunov exponent is a measure of the rate of separation of infinitesimally close trajectories in a dynamical system. It's a key indicator of chaos in a system:

- **Positive Lyapunov exponent**: Indicates chaos. Nearby trajectories diverge exponentially over time.
- **Zero Lyapunov exponent**: Indicates stability or a bifurcation point.
- **Negative Lyapunov exponent**: Indicates stability. Nearby trajectories converge over time.

For neural network attractors, the Lyapunov exponent calculation:

1. Takes an initial state and runs the system forward
2. Takes a slightly perturbed initial state and runs that forward
3. Measures the divergence between the trajectories
4. Computes the logarithm of this divergence rate

The implementation in this project:
- Uses the direct method for calculating Lyapunov exponents
- Provides visualization of how the exponent converges
- Labels the system as "chaotic" or "regular" based on the sign of the exponent
- Shows parameter values in the plot title for reference

Example usage:
```bash
python main.py --N 4 --D 32 --s 0.8 --plot-lyapunov --lyapunov-iterations 8000
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Inspired by the original work by J.C. Sprott "Artificial Neural Net Attractors" published in Computers & Graphics, Vol. 22, No. 1, pp. 143-149, 1998.
