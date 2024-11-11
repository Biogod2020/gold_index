import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

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

def plot_observation_data(df, value_column, category_column, padding=0.05, sigma=1, vmin=0.1, vmax=0.5,
                          save_name="contour", show=True):
    vmax = 0.25 if value_column == "pime_gp2" else vmax
    vmin = 0.20 if value_column == "pime_gp1" else vmin
    
    # Calculate padding for the grid
    obs = df.copy()

    obs[value_column] = (obs[value_column] - obs[value_column].min()) / (obs[value_column].max() - obs[value_column].min())

    obs[category_column] = obs[category_column].astype("category")
    
    x_pad = padding * (obs['x'].max() - obs['x'].min())
    y_pad = padding * (obs['y'].max() - obs['y'].min())

    # Create grid for interpolation
    grid_x, grid_y = np.mgrid[
        (obs['x'].min() - x_pad):(obs['x'].max() + x_pad):150j,
        (obs['y'].min() - y_pad):(obs['y'].max() + y_pad):150j
    ]
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]


    cat_na_loc = {}
    for categ in obs[category_column].unique():
        mask = obs[category_column] == categ
        obs[f"{value_column}_{categ}"] = obs[value_column]
        obs.loc[~mask, f"{value_column}_{categ}"] = None

        # Interpolate values on the grid
        scaled_values = obs[f"{value_column}_{categ}"]
        grid_z = griddata((obs['x'], obs['y']), scaled_values, (grid_x, grid_y), method='nearest')
        cat_na_loc[categ] = np.isnan(grid_z)

    # Interpolate values on the grid
    scaled_values = obs[f"{value_column}"]
    grid_z = griddata((obs['x'], obs['y']), scaled_values, (grid_x, grid_y), method='nearest')
    # Filter and smooth grid
    grid_z_smooth = filter_and_smooth_grid(grid_z, obs, grid_points, sigma=sigma, filter_function=filter_grid)
    
    grid_z_final = grid_z_smooth
    
    cat_z_value = {}
    for categ in obs[category_column].unique():
        cat_z_value[categ] = grid_z_final.copy()
        cat_z_value[categ][cat_na_loc[categ]] = None


    # Calculate mean of the category for color bar separation
    category_mean_value = np.mean(obs.groupby(category_column)[value_column].mean().values)


    
    # Get the original 'Reds' colormap
    cmap = plt.get_cmap("Reds")
    # Split the colormap into two halves
    cmap_lower = ListedColormap(cmap(np.linspace(0, 0.5, 256)))
    cmap_upper = ListedColormap(cmap(np.linspace(0.5, 1, 256)))

    # Calculate the mean of grid_z_final for each category
    means = {categ: np.nanmean(grid_z_final) for categ, grid_z_final in cat_z_value.items()}


    # Determine which category has the lower mean
    lower_mean_category = min(means, key=means.get)
    upper_mean_category = max(means, key=means.get)

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(data=obs, x="x", y="y", hue="plaque", linewidth=0,
                    palette={True: "red", False: "lightgrey"}, s=20, ax=ax)
    
    # Loop through the categories and apply the appropriate colormap
    for categ, grid_z_final in cat_z_value.items():
        if categ == lower_mean_category:
            cmap_to_use = cmap_lower
            contour_filled_lower = ax.contourf(grid_x, grid_y, grid_z_final, levels=15, cmap=cmap_to_use, alpha=0.65, vmin=vmin, vmax=vmax)
        else:
            cmap_to_use = cmap_upper
            contour_filled_upper = ax.contourf(grid_x, grid_y, grid_z_final, levels=15, cmap=cmap_to_use, alpha=0.65, vmin=vmin, vmax=vmax)
        
        # Draw contour lines
        ax.contour(grid_x, grid_y, grid_z_final, levels=15, linewidths=1, colors="white")



    add_scalebar(ax, length_μm=1000, label="1mm", units_per_μm=2)

    ax.set_facecolor("black")
    ax.set_aspect("equal", "box")
    ax.grid(False)
    plt.title(f"{save_name}_{value_column}")

    if save_name:
        plt.savefig(f"./contour_fig_Aug_16_{value_column}{save_name}.pdf", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

# Example usage
# plot_observation_data(obs, 'pime_gp1', 'pal_region')
