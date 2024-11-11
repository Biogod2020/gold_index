import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.patches as patches


def add_scalebar(ax, length_μm, units_per_μm, label='1000 μm', **kwargs):
    """
    Add a scale bar to an axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the scale bar to.
        length_μm (float): The length of the scale bar in micrometers.
        units_per_μm (float): The number of units per micrometer.
        label (str): The label for the scale bar.
        kwargs: Additional arguments for the Rectangle patch.

    Returns:
        None
    """
    length_units = length_μm * units_per_μm
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xpos = xlim[0] + (xlim[1] - xlim[0]) * 0.8
    ypos = ylim[0] + (ylim[1] - ylim[0]) * 0.9
    
    scalebar = patches.Rectangle((xpos, ypos), length_units, (ylim[1] - ylim[0]) * 0.01,
                                 linewidth=1, edgecolor='white', facecolor='white', **kwargs)
    ax.add_patch(scalebar)
    
    ax.text(xpos + length_units / 2, ypos + (ylim[1] - ylim[0]) * 0.02, label,
            color='white', ha='center', va='bottom', fontsize=10)


def filter_grid(grid_z, obs, grid_points_coords):
    """
    Filter grid to remove invalid values.

    Args:
        grid_z (numpy.ndarray): Grid values.
        obs (pandas.DataFrame): DataFrame containing cell observations.
        grid_points_coords (numpy.ndarray): Coordinates of grid points.

    Returns:
        numpy.ndarray: Filtered grid.
    """
    original_points = np.c_[obs['x'].values, obs['y'].values]
    tree = cKDTree(grid_points_coords)
    distances, indices = tree.query(original_points)
    inx_unique = np.array(list(set(indices)))
    grid_z_flat = grid_z.flatten()
    mask = np.zeros(shape=len(grid_z_flat), dtype=bool)
    mask[inx_unique] = True
    grid_z_flat[~mask] = 0
    return grid_z_flat.reshape(grid_z.shape)

def filter_and_smooth_grid(grid_z, obs, grid_points, sigma, filter_function):
    grid_z_fil = filter_function(grid_z, obs, grid_points)
    
    # Record positions of 0 values in grid_z_fil
    zero_positions = np.where(grid_z_fil == 0)
    
    # Apply Gaussian filter
    grid_z_smooth = gaussian_filter(grid_z_fil, sigma=sigma)
    
    # Reset the 0 positions back to NaN
    grid_z_smooth[zero_positions] = None
    
    return grid_z_smooth

def plot_observation_data(df: pd.DataFrame, 
                          value_column: str, 
                          category_column: str, 
                          padding: float = 0.05, 
                          sigma: float = 1.0, 
                          vmin: float = 0.1, 
                          vmax: float = 0.5, 
                          save_name: str = "contour", 
                          show: bool = True) -> None:
    """
    Plots observation data with contour interpolation by category.

    Parameters:
    - df (pd.DataFrame): Input data containing 'x', 'y' coordinates and observation values.
    - value_column (str): Name of the column with the values to interpolate and plot.
    - category_column (str): Name of the column representing categories.
    - padding (float): Padding percentage around the plot boundary.
    - sigma (float): Smoothing factor for grid data.
    - vmin (float): Minimum value for color scale.
    - vmax (float): Maximum value for color scale.
    - save_name (str): File name for saving the plot.
    - show (bool): Whether to display the plot.

    Returns:
    - None
    """

    # Adjust color scale limits for specific value columns
    vmax = 0.25 if value_column == "pime_gp2" else vmax
    vmin = 0.20 if value_column == "pime_gp1" else vmin
    
    # Normalize values in value_column between 0 and 1
    obs = df.copy()
    obs[value_column] = (obs[value_column] - obs[value_column].min()) / (obs[value_column].max() - obs[value_column].min())
    obs[category_column] = obs[category_column].astype("category")
    
    # Calculate padding around the x and y axes for interpolation grid
    x_pad = padding * (obs['x'].max() - obs['x'].min())
    y_pad = padding * (obs['y'].max() - obs['y'].min())

    # Create a grid of x and y coordinates for interpolation
    grid_x, grid_y = np.mgrid[
        (obs['x'].min() - x_pad):(obs['x'].max() + x_pad):150j,
        (obs['y'].min() - y_pad):(obs['y'].max() + y_pad):150j
    ]
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Dictionary to store NaN locations for each category's grid data
    cat_na_loc = {}
    for categ in obs[category_column].unique():
        # Mask rows belonging to the current category
        mask = obs[category_column] == categ
        obs[f"{value_column}_{categ}"] = obs[value_column]
        obs.loc[~mask, f"{value_column}_{categ}"] = None

        # Interpolate category-specific values onto the grid
        scaled_values = obs[f"{value_column}_{categ}"]
        grid_z = griddata((obs['x'], obs['y']), scaled_values, (grid_x, grid_y), method='nearest')
        cat_na_loc[categ] = np.isnan(grid_z)

    # Interpolate and smooth overall grid data for plotting
    scaled_values = obs[value_column]
    grid_z = griddata((obs['x'], obs['y']), scaled_values, (grid_x, grid_y), method='nearest')
    grid_z_smooth = filter_and_smooth_grid(grid_z, obs, grid_points, sigma=sigma, filter_function=filter_grid)
    
    grid_z_final = grid_z_smooth
    
    # Dictionary to store category-specific smoothed grid values
    cat_z_value = {}
    for categ in obs[category_column].unique():
        cat_z_value[categ] = grid_z_final.copy()
        cat_z_value[categ][cat_na_loc[categ]] = None

    # Calculate mean values for each category to adjust colormap scales
    category_mean_value = np.mean(obs.groupby(category_column)[value_column].mean().values)
    
    # Split colormap 'Reds' into lower and upper halves
    cmap = plt.get_cmap("Reds")
    cmap_lower = ListedColormap(cmap(np.linspace(0, 0.5, 256)))
    cmap_upper = ListedColormap(cmap(np.linspace(0.5, 1, 256)))

    # Calculate mean grid values for each category and determine lower and upper mean categories
    means = {categ: np.nanmean(grid_z_final) for categ, grid_z_final in cat_z_value.items()}
    lower_mean_category = min(means, key=means.get)
    upper_mean_category = max(means, key=means.get)

    # Plot contour and scatter data
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(data=obs, x="x", y="y", hue="plaque", linewidth=0,
                    palette={True: "red", False: "lightgrey"}, s=20, ax=ax)
    
    # Plot filled contours for each category using appropriate colormap
    for categ, grid_z_final in cat_z_value.items():
        if categ == lower_mean_category:
            cmap_to_use = cmap_lower
            contour_filled_lower = ax.contourf(grid_x, grid_y, grid_z_final, levels=15, cmap=cmap_to_use, alpha=0.65, vmin=vmin, vmax=vmax)
        else:
            cmap_to_use = cmap_upper
            contour_filled_upper = ax.contourf(grid_x, grid_y, grid_z_final, levels=15, cmap=cmap_to_use, alpha=0.65, vmin=vmin, vmax=vmax)
        
        # Add contour lines to highlight data regions
        ax.contour(grid_x, grid_y, grid_z_final, levels=15, linewidths=1, colors="white")

    # Add scalebar and plot formatting
    add_scalebar(ax, length_μm=1000, label="1mm", units_per_μm=2)
    ax.set_facecolor("black")
    ax.set_aspect("equal", "box")
    ax.grid(False)
    plt.title(f"{save_name}_{value_column}")

    # Save or display the plot based on user preference
    if save_name:
        plt.savefig(f"./contour_fig_Aug_16_{value_column}{save_name}.pdf", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

# Example usage
# plot_observation_data(obs, 'pime_gp1', 'pal_region')

