import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon, Patch
import matplotlib.cm as cmp
from matplotlib.colors import Normalize, ListedColormap

# Plot individual cells with continuous or discrete colors
def plot_cells(ax, obs, cell_data_dict, color_palette, plaque_set, value_column, is_continuous):
    legend_patches = {}  # Store unique patches for the legend

    if is_continuous:
        # Normalize the continuous data for color mapping
        norm = Normalize(vmin=obs[value_column].min(), vmax=obs[value_column].max())
        cmap = cmp.PuRd
        color_mapper = lambda value: cmap(norm(value))
    else:
        # Use discrete colors
        unique_values = obs[value_column].unique()
        discrete_cmap = ListedColormap([color_palette[val] for val in unique_values if val in color_palette])
        color_mapper = lambda value: color_palette.get(value, "white")

    # Precompute colors for all rows
    obs['color'] = obs[value_column].map(color_mapper)

    for index, row in obs.iterrows():
        x, y, cluster = row['x'], row['y'], row['fine']
        value = row[value_column]

        if (x, y) in plaque_set:
            cluster = "Plaque"

        slice_id = list(cell_data_dict.keys())[0].split(sep=":")[0]
        sample_id = f"{slice_id}:{x}_{y}"

        if sample_id in cell_data_dict:
            ndarray = process_ndarray(cell_data_dict[sample_id])
            if not ndarray.size:
                continue

            mass_center = np.array([x, y])
            vertices = ndarray + mass_center

            polygon_color = "red" if cluster == "Plaque" else row['color']

            polygon = patches.Polygon(vertices, closed=True, color=polygon_color)
            ax.add_patch(polygon)

            cluster_label = cluster if is_continuous else value

            if not is_continuous and cluster_label not in legend_patches:
                legend_patches[cluster_label] = Patch(color=polygon_color, label=cluster_label)

    return legend_patches if not is_continuous else None

# Process ndarray to filter out invalid values
def process_ndarray(ndarray):
    mask = ~np.any(ndarray == 32767, axis=1)
    return ndarray[mask]

# Highlight selected plaque
def highlight_selected_plaque(ax, plaque_coord, tissue_id):
    selected_plaque = plaque_coord[plaque_coord['tissue'] == tissue_id]
    centroid = selected_plaque[['row', 'col']].mean()

    x_lim = (centroid['col'] - 500, centroid['col'] + 500)
    y_lim = (centroid['row'] - 500, centroid['row'] + 500)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    return ax

# Draw the outline box representing the viewing region
def draw_viewing_region(ax, color='black', linewidth=1):
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    width = x_limits[1] - x_limits[0]
    height = y_limits[1] - y_limits[0]
    outline_box = Rectangle((x_limits[0], y_limits[0]), width, height,
                            edgecolor=color, facecolor='none', lw=linewidth)

    ax.add_patch(outline_box)

    return ax

# Add a legend to the axis for discrete values
def plot_legend(ax, legend_patches):
    if legend_patches:
        ax.legend(handles=legend_patches.values(), loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)

# Add colorbar for continuous values
def plot_colorbar(ax, cmap, norm):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

# Set the view window based on the win_size
def set_axes_view(ax, x_start, y_start, win_size):
    if win_size:
        ax.set_xlim([x_start, x_start + win_size])
        ax.set_ylim([y_start, y_start + win_size])

# Set axes styles
def set_axes_style(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.subplots_adjust(right=0.75)

# Main function for plotting with continuous or discrete coloring
def plot_combined_v5(ax, cell_data_dict, obs, color_palette, plaque_coord, value_column, is_continuous):
    plaque_set = set(zip(plaque_coord['row'], plaque_coord['col']))
    legend_patches = plot_cells(ax, obs, cell_data_dict, color_palette, plaque_set, value_column, is_continuous)
    if not is_continuous:
        plot_legend(ax, legend_patches)
    else:
        norm = Normalize(vmin=obs[value_column].min(), vmax=obs[value_column].max())
        plot_colorbar(ax, cmp.PuRd, norm)

    return ax

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cmp
