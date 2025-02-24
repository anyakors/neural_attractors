"""
Visualization tools for neural network attractors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from typing import Tuple, Optional, Callable


def make_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Create list of line segments from x and y coordinates.
    
    Parameters:
    -----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
        
    Returns:
    --------
    np.ndarray
        Array of line segments in the format required by LineCollection
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    cmap = plt.get_cmap("jet"),
    norm = plt.Normalize(0.0, 1.0),
    linewidth: float = 1.0,
    alpha: float = 0.05,
    ax = None
):
    """
    Plot a colored line with coordinates x and y.
    
    Parameters:
    -----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : Optional[np.ndarray]
        Array of values to color the line
    cmap : matplotlib.colors.Colormap
        Colormap to use
    norm : matplotlib.colors.Normalize
        Normalize instance for scaling data values
    linewidth : float
        Width of the line
    alpha : float
        Transparency of the line
    ax : matplotlib.axes.Axes
        Axes to plot on
        
    Returns:
    --------
    matplotlib.collections.LineCollection
        The line collection that was added to the plot
    """
    if ax is None:
        ax = plt.gca()
        
    # Default colors equally spaced on [0, 1]
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    
    # Create line segments
    segments = make_segments(x, y)
    
    # Create LineCollection
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )
    
    # Add to plot
    ax.add_collection(lc)
    
    return lc


def plot_attractor_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    skip_value: int = 1,
    color_function: Optional[Callable] = None,
    cmap = plt.get_cmap("Spectral"),
    linewidth: float = 0.1,
    alpha: float = 0.1,
    figsize: Tuple[float, float] = (10, 10),
    interpolate_steps: int = 3,
    output_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
):
    """
    Plot an attractor trajectory.
    
    Parameters:
    -----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    skip_value : int
        Number of points to skip for sparser plotting
    color_function : Optional[Callable]
        Function to generate colors based on x and y
    cmap : matplotlib.colors.Colormap
        Colormap to use
    linewidth : float
        Width of the line
    alpha : float
        Transparency of the line
    figsize : Tuple[float, float]
        Size of the figure in inches
    interpolate_steps : int
        Number of interpolation steps for smoother trajectory
    output_path : Optional[str]
        Path to save the figure
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Interpolate path for smoother curves
    if interpolate_steps > 1:
        path = mpath.Path(np.column_stack([x, y]))
        verts = path.interpolated(steps=interpolate_steps).vertices
        x, y = verts[:, 0], verts[:, 1]
    
    # Apply skip value
    x_plot = x[::skip_value]
    y_plot = y[::skip_value]
    
    # Default color function
    if color_function is None:
        z = abs(np.sin(1.6 * y_plot + 0.4 * x_plot))
    else:
        z = color_function(x_plot, y_plot)
    
    # Plot colored line
    colorline(x_plot, y_plot, z, cmap=cmap, linewidth=linewidth, alpha=alpha, ax=ax)
    
    # Set limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    
    # Remove axes for cleaner look
    ax.set_axis_off()
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
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


def plot_attractor_time_series(
    x: np.ndarray,
    n_points: int = 1000,
    figsize: Tuple[float, float] = (10, 4),
    output_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
):
    """
    Plot the time series of an attractor.
    
    Parameters:
    -----------
    x : np.ndarray
        Time series data
    n_points : int
        Number of points to display
    figsize : Tuple[float, float]
        Size of the figure in inches
    output_path : Optional[str]
        Path to save the figure
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Show the last n_points
    x_plot = x[-n_points:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot time series
    ax.plot(np.arange(len(x_plot)), x_plot, linewidth=1)
    
    # Add labels
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Neural Network Attractor Time Series')
    
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


def plot_attractor_phase_scatter(
    x: np.ndarray,
    y: np.ndarray,
    n_points: int = 1000,
    color_data: Optional[np.ndarray] = None,
    cmap = plt.get_cmap('rainbow'),
    figsize: Tuple[float, float] = (8, 8),
    output_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
):
    """
    Create a scatter plot of the attractor in phase space.
    
    Parameters:
    -----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    n_points : int
        Number of points to display
    color_data : Optional[np.ndarray]
        Data to use for coloring the points
    cmap : matplotlib.colors.Colormap
        Colormap to use
    figsize : Tuple[float, float]
        Size of the figure in inches
    output_path : Optional[str]
        Path to save the figure
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Show the last n_points
    x_plot = x[-n_points:]
    y_plot = y[-n_points:]
    
    # Default color data
    if color_data is None:
        color_data = x_plot
    else:
        color_data = color_data[-n_points:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(x_plot, y_plot, c=color_data, cmap=cmap, s=1, alpha=0.5)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax)
    
    # Add labels
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Neural Network Attractor Phase Space')
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
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
