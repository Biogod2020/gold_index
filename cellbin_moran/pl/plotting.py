import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from .palettes import cell_type_colors, general_type_colors, brain_region_colors, age_group_colors

def plot_normalized_umap(
    slide: dict,
    num_rows: int = 1,
    num_cols: int = 7,
    cell_type: str = "Micro",
    color: str = "min_center_dist",
    palette: dict = cell_type_colors,
    **umap_kwargs
) -> plt.Figure:
    """
    Plots a grid of UMAP projections with normalized values for each AnnData object in 'slide'.

    Args:
        slide: Dictionary where keys are identifiers and values are AnnData objects.
        num_rows: Number of rows in the subplot grid.
        num_cols: Number of columns in the subplot grid.
        cell_type: The cell type to filter on for plotting. If None, plot all cells.
        color: The column name in `adata.obs` to normalize and plot.
        palette: Optional dictionary specifying the color palette to use if 'color' is categorical.
        **umap_kwargs: Arbitrary keyword arguments to pass to the `sc.pl.umap` function.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    def normalize(values):
        min_val = values.min()
        max_val = values.max()
        return (values - min_val) / (max_val - min_val)

    # Determine the number of subplots needed
    # num_plots = len(slide)
    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), gridspec_kw={'width_ratios': [1] * num_cols})
    axes = axes.flatten()  # Flatten the array to easily iterate through it

    # Rearrange the slide items to start with the second key and put the first key last
    slide_items = list(slide.items())
    slide_items = slide_items[1:] + slide_items[:1]

    # Plot each adata in the appropriate subplot
    for i, (key, adata) in enumerate(slide_items):
        tmp = adata.copy()
        if i >= len(axes):  # Prevent indexing errors if there are more slides than subplots
            break
        if (i < len(axes) - 1) or isinstance(tmp.obs[color][0], (int, float, complex)):
            leg_stat = None
        else:
            leg_stat = "right margin"

        if cell_type:
            mask = tmp.obs["celltype"] == cell_type
        else:
            mask = slice(None)  # Select all cells if cell_type is None

        if color == "min_center_dist":
            tmp.obs.loc[mask, f"{color}_normalized"] = normalize(tmp.obs.loc[mask, color])
            plot_color = f"{color}_normalized"
        else:
            plot_color = color

        sc.pl.umap(tmp[mask], color=plot_color, vmin=0, vmax=1 if color == "min_center_dist" else None, 
                   ax=axes[i], show=False, colorbar_loc=None, legend_loc=leg_stat, palette=palette, **umap_kwargs)
        axes[i].set_title(key.split(sep="_")[0])
        for pos in ['top', 'bottom', 'left']:
            axes[i].spines[pos].set_visible(False)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Add an axis for the color bar on the right border if color is numerical and default
    if isinstance(tmp.obs[color][0], (int, float, complex)):
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f'{color}_normalized')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for the color bar
    return fig


def save_figure_to_pdf(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Saves the given figure object to a PDF file.

    Args:
        fig: The matplotlib figure object to save.
        filename: The name of the output PDF file.
        dpi: The resolution of the output PDF file.
    """
    with PdfPages(filename) as pdf:
        fig.savefig(pdf, format='pdf', dpi=dpi)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np

def plot_kde_normalized_distance(
    pfcdf: pd.DataFrame,
    cell_type: str = "Micro",
    dist_col: str = "min_center_dist",
    fine_col: str = "fine",
    palette: dict = cell_type_colors,
    save_path: str = None,
    fig_size: tuple = (8, 4),
    dpi: int = 350,
    scaling_method: str = "standard",  # Parameter to choose the scaling method
    ax: plt.Axes = None,
    **kde_kwargs
) -> plt.Axes:
    """
    Plots a KDE of normalized and log-transformed distances for a specific cell type in the DataFrame.

    Args:
        pfcdf: DataFrame containing the data.
        cell_type: The cell type to filter on for plotting.
        dist_col: The column name in `pfcdf` to normalize and plot.
        fine_col: The column name representing finer categorization to plot separately.
        palette: Optional dictionary specifying the color palette to use for each `fine` value.
        save_path: Optional path where the plot should be saved. If None, the plot will not be saved.
        fig_size: Size of the figure.
        dpi: Dots per inch for saving the figure.
        scaling_method: Method to use for scaling ('standard', 'minmax', 'maxabs').
        ax: Optional matplotlib axis object to draw on. If None, a new figure and axis will be created.
        **kde_kwargs: Arbitrary keyword arguments to pass to the `sns.kdeplot` function.

    Returns:
        plt.Axes: The matplotlib axis object used for plotting.
    """
    def scale_values(values, method):
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "maxabs":
            scaler = MaxAbsScaler()
        else:
            raise ValueError("Unsupported scaling method. Choose from 'standard', 'minmax', or 'maxabs'.")
        return scaler.fit_transform(values.reshape(-1, 1)).flatten()
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.figure
        
    mask = (pfcdf["celltype"] == cell_type) & (pfcdf[dist_col] > 0)
    filtered_data = pfcdf[mask]
    filtered_data = filtered_data.copy()
    filtered_data["fine"] = filtered_data[fine_col].astype(str)

    # Scale the distance values
    filtered_data["dist_scaled"] = scale_values(filtered_data[dist_col].values, scaling_method)
    filtered_data["Normalized and -log10 transformed distance"] = -np.log10(filtered_data["dist_scaled"] + 1e-3)
    plot_color = "Normalized and -log10 transformed distance"

    # Iterate over each unique 'fine' value to plot the KDEs separately
    for fine_value in filtered_data["fine"].unique():
        sns.kdeplot(
            data=filtered_data[filtered_data["fine"] == fine_value],
            x=plot_color,
            label=fine_value,
            ax=ax,
            color=palette[fine_value] if palette else None,
            fill=True,
            alpha=0.3,
            **kde_kwargs
        )

    ax.grid(visible=False)
    ax.legend(title="Cell's Subtypes")  # Add a legend to differentiate KDEs

    # Save the figure only if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
    
    return ax

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc

def calculate_pca(
    adata,
    genes: list,
    n_components: int = 2,
    scale: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculates PCA for the specified genes in the AnnData object.

    Args:
        adata: AnnData object containing the data.
        genes: List of genes to use for PCA calculation.
        n_components: Number of principal components to compute. Default is 2.
        scale: Whether to scale the data before PCA. Default is True.
        verbose: Whether to print additional information. Default is False.

    Returns:
        np.ndarray: PCA result for the specified genes.
    """

    if scale:
        sc.pp.scale(adata, zero_center=True)

    available_genes = [gene for gene in genes if gene in adata.var_names]
    missing_genes = set(genes) - set(available_genes)
    if missing_genes:
        print(f"The following genes are missing in the data and will be ignored: {missing_genes}")

    gene_data = adata[:, available_genes].X
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(gene_data)

    if verbose:
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return pca_result

def plot_genes_in_spatial(
    adata_input,
    sample_ids: list = None,
    genes: list = [
        "Rbfox3", "Map2", "Gap43", "Syn1", "Nefl",
        "Vim", "Gfap", "Eno2", "Tubb3", "Camk2a"
    ],
    cut_off: float = 99.5,
    fig_size_per_gene: tuple = (4, 4),
    gene_col: str = "gene",
    sample_col: str = "sample",
    embedding_basis: str = "spatial",
    size: int = 6,
    save_path: str = None,
    dpi: int = 350,
    calc_pca: bool = False,
    pca_params: dict = None,
    n_pcs_to_plot: int = 2,
    **kwargs
) -> plt.Figure:
    """
    Plots spatial embeddings for a list of neuronal soma genes across multiple samples.
    Optionally includes PCA calculation for the provided genes using sklearn to avoid modifying anndata.

    Args:
        adata_input: Either an AnnData object or a dictionary of AnnData objects.
        sample_ids: List of sample identifiers to plot. If None, will be derived from adata_input.
        genes: List of genes to plot. Defaults to a predefined list of neuronal soma genes.
        fig_size_per_gene: Size of the figure for each gene subplot.
        gene_col: Column name for genes in the AnnData object.
        sample_col: Column name for samples in the AnnData object.
        embedding_basis: Basis for spatial embedding in the AnnData object.
        size: Size of the points in the plot.
        save_path: Optional path where the plot should be saved. If None, the plot will not be saved.
        dpi: Dots per inch for saving the figure.
        calc_pca: Boolean flag to calculate PCA.
        pca_params: Dictionary of parameters for the calculate_pca function.
        n_pcs_to_plot: Number of principal components to plot. Default is 2.
        **kwargs: Additional keyword arguments to pass to `sc.pl.embedding`.

    Returns:
        plt.Figure: The matplotlib figure object used for plotting.
    """

    # Determine sample IDs if not provided
    if sample_ids is None:
        if isinstance(adata_input, dict):
            sample_ids = list(adata_input.keys())
        else:
            sample_ids = adata_input.obs[sample_col].unique().tolist()

    # Calculate top 0.5% value for each gene
    if cut_off:
        top_5_percent_values = {}
        for gene in genes:
            all_values = []
            if isinstance(adata_input, dict):
                for adata in adata_input.values():
                    if gene in adata.var_names:
                        all_values.extend(adata[:, gene].X.flatten())
            else:
                if gene in adata_input.var_names:
                    all_values.extend(adata_input[:, gene].X.flatten())
    
            if all_values:
                top_5_percent_values[gene] = np.percentile(all_values, cut_off)

    # Ensure PCA is computed for the genes if requested
    if calc_pca:
        if pca_params is None:
            pca_params = {}
        pca_params.update({'n_components': n_pcs_to_plot})
        if isinstance(adata_input, dict):
            for adata in adata_input.values():
                adata.obsm['X_pca'] = calculate_pca(adata, genes, **pca_params)
                for i in range(n_pcs_to_plot):
                    adata.obs[f"pc_{i}"] = adata.obsm['X_pca'][:, i] 
        else:
            adata_input.obsm['X_pca'] = calculate_pca(adata_input, genes, **pca_params)
            adata_input.obs[f"pc_{i}"] = adata_input.obsm['X_pca'][:, i]

    n_genes = len(genes)
    n_samples = len(sample_ids)

    fig_size = (fig_size_per_gene[0] * n_samples, fig_size_per_gene[1] * n_genes)
    fig, axs = plt.subplots(n_genes, n_samples, figsize=fig_size)

    if calc_pca:
        genes = [f"pc_{i}" for i in range(n_pcs_to_plot)]

    for i, gene in enumerate(genes):
        if cut_off:
            vmax = top_5_percent_values.get(gene, None)
        else:
            vmax = 5
        for j, sample_id in enumerate(sample_ids):
            ax = axs[i, j] if n_genes > 1 and n_samples > 1 else (axs[j] if n_genes == 1 else axs[i])
            try:
                if isinstance(adata_input, dict):
                    adata_sample = adata_input[sample_id]
                else:
                    adata_sample = adata_input[adata_input.obs[sample_col] == sample_id]


                sc.pl.embedding(
                    adata_sample,
                    basis=embedding_basis,
                    ax=ax,
                    color=gene,
                    size=size,
                    show=False,
                    vmin=0,
                    vmax=vmax,
                    **kwargs
                )
                ax.set_title(f"{sample_id}_{gene}")
                ax.set_aspect('equal', adjustable='box')
            except Exception as e:
                print(f"Error plotting {gene} for sample {sample_id}: {str(e)}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi)

    return fig

def plot_genes_in_spatial_by_genes(
    
    adata_input,
    sample_ids: list = None,
    genes: list = [
        "Rbfox3", "Map2", "Gap43", "Syn1", "Nefl",
        "Vim", "Gfap", "Eno2", "Tubb3", "Camk2a"
    ],
    cut_off: float = 99.5,
    fig_size_per_gene: tuple = (4, 4),
    gene_col: str = "gene",
    sample_col: str = "sample",
    embedding_basis: str = "spatial",
    size: int = 6,
    save_path: str = None,
    dpi: int = 100,
    calc_pca: bool = False,
    pca_params: dict = None,
    n_pcs_to_plot: int = 2,
    pattern: str = None,
    **kwargs
) -> plt.Figure:
    """
    Plots spatial embeddings for a list of neuronal soma genes across multiple samples.
    Optionally includes PCA calculation for the provided genes using sklearn to avoid modifying anndata.
    Args:
        adata_input: Either an AnnData object or a dictionary of AnnData objects.
        sample_ids: List of sample identifiers to plot. If None, will be derived from adata_input.
        genes: List of genes to plot. Defaults to a predefined list of neuronal soma genes.
        fig_size_per_gene: Size of the figure for each gene subplot.
        gene_col: Column name for genes in the AnnData object.
        sample_col: Column name for samples in the AnnData object.
        embedding_basis: Basis for spatial embedding in the AnnData object.
        size: Size of the points in the plot.
        save_path: Optional path where the plot should be saved. If None, the plot will not be saved.
        dpi: Dots per inch for saving the figure.
        calc_pca: Boolean flag to calculate PCA.
        pca_params: Dictionary of parameters for the calculate_pca function.
        n_pcs_to_plot: Number of principal components to plot. Default is 2.
        pattern: Pattern to extract from the sample ID to group samples in rows.
        **kwargs: Additional keyword arguments to pass to `sc.pl.embedding`.
    Returns:
        plt.Figure: The matplotlib figure object used for plotting.
    """

    import re
    import re
    from collections import defaultdict
    
    # Define the function to group strings
    def group_strings_by_substring(strings, pattern = pattern):
        # Regular expression pattern to extract substring between HZ and M
        pattern = re.compile(pattern)
        
        # Dictionary to store grouped strings
        grouped_strings = defaultdict(list)
        
        # Iterate through each string in the input list
        for string in strings:
            # Find the substring between HZ and M
            match = pattern.search(string)
            if match:
                # Extracted substring as the key
                substring = match.group()
                # Append the original string to the corresponding list in the dictionary
                grouped_strings[substring].append(string)
        
        return dict(grouped_strings)

    # Determine sample IDs if not provided
    if sample_ids is None:
        if isinstance(adata_input, dict):
            sample_ids = list(adata_input.keys())
        else:
            sample_ids = adata_input.obs[sample_col].unique().tolist()
    
    # Group sample IDs by pattern
    grouped_samples = {}
    grouped_samples = group_strings_by_substring(sample_ids)
    # Custom sort key function
    def custom_sort_key(key):
        # Treat "12" specially to ensure it's last
        if key == '12':
            return float('inf')
        return int(key)
    
    # Sort the dictionary keys using the custom sort key
    sorted_keys = sorted(grouped_samples, key=custom_sort_key)
    
    # Create a new dictionary with keys sorted
    sorted_dict = {key: grouped_samples[key] for key in sorted_keys}
    
    # Output the sorted dictionary
    sorted_dict

    # Calculate top 0.5% value for each gene
    if cut_off:
        top_5_percent_values = {}
        for gene in genes:
            all_values = []
            if isinstance(adata_input, dict):
                for adata in adata_input.values():
                    if gene in adata.var_names:
                        all_values.extend(adata[:, gene].X.flatten())
            else:
                if gene in adata_input.var_names:
                    all_values.extend(adata_input[:, gene].X.flatten())
    
            if all_values:
                top_5_percent_values[gene] = np.percentile(all_values, cut_off)

    # Ensure PCA is computed for the genes if requested
    if calc_pca:
        if pca_params is None:
            pca_params = {}
        pca_params.update({'n_components': n_pcs_to_plot})
        if isinstance(adata_input, dict):
            for adata in adata_input.values():
                adata.obsm['X_pca'] = calculate_pca(adata, genes, **pca_params)
                for i in range(n_pcs_to_plot):
                    adata.obs[f"pc_{i}"] = adata.obsm['X_pca'][:, i] 
        else:
            adata_input.obsm['X_pca'] = calculate_pca(adata_input, genes, **pca_params)
            adata_input.obs[f"pc_{i}"] = adata_input.obsm['X_pca'][:, i]
            
    if calc_pca:
        genes = [f"pc_{i}" for i in range(n_pcs_to_plot)]
    # Loop through each gene for plotting
    for gene in genes:
        # Create subplots based on the number of groups and the maximum number of samples in a group
        n_groups = len(sorted_dict)
        max_samples_in_group = max(len(samples) for samples in sorted_dict.values())
        
        fig, axs = plt.subplots(n_groups, max_samples_in_group, figsize=(fig_size_per_gene[0] * max_samples_in_group, fig_size_per_gene[1] * n_groups))

        for i, (group_id, sample_id_list) in enumerate(sorted_dict.items()):
            for j, sample_id in enumerate(sample_id_list):
                ax = axs[i, j] if n_groups > 1 and max_samples_in_group > 1 else (axs[j] if n_groups == 1 else axs[i])
                if cut_off:
                    vmax = top_5_percent_values.get(gene, None)
                else:
                    vmax = 5
                try:
                    if isinstance(adata_input, dict):
                        adata_sample = adata_input[sample_id]
                    else:
                        adata_sample = adata_input[adata_input.obs[sample_col] == sample_id]

                    sc.pl.embedding(
                        adata_sample,
                        basis=embedding_basis,
                        ax=ax,
                        color=gene,
                        size=size,
                        show=False,
                        vmin=0,
                        vmax=vmax,
                        **kwargs
                    )
                    ax.set_title(f"{sample_id}_{gene}")
                    ax.set_aspect('equal', adjustable='box')
                except Exception as e:
                    print(f"Error plotting {gene} for sample {sample_id}: {str(e)}")

        plt.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}_{gene}.png", dpi=dpi)

        plt.close()

    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import matplotlib.cm as cmp
from matplotlib.colors import Normalize, ListedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

def plot_cells(ax, obs, cell_data_dict, color_palette, plaque_set, value_column, is_continuous):
    """
    Plot individual cells with continuous or discrete colors.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot the cells on.
        obs (pandas.DataFrame): DataFrame containing cell observations with 'x', 'y', 'fine', and value_column.
        cell_data_dict (dict): Dictionary with cell data arrays.
        color_palette (dict): Dictionary mapping discrete values to colors.
        plaque_set (set): Set of coordinates representing plaques.
        value_column (str): Column name in `obs` containing the values for coloring.
        is_continuous (bool): Whether the values are continuous.

    Returns:
        dict or None: Dictionary of legend patches if discrete values, otherwise None.
    """
    legend_patches = {}  # Store unique patches for the legend

    if is_continuous:
        norm = Normalize(vmin=obs[value_column].min(), vmax=obs[value_column].max())
        cmap = cmp.PuRd
    else:
        unique_values = obs[value_column].unique()
        discrete_cmap = ListedColormap([color_palette[val] for val in unique_values if val in color_palette])

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

            if cluster == "Plaque":
                polygon_color = "red"
            elif is_continuous:
                polygon_color = cmap(norm(value))
            else:
                polygon_color = color_palette.get(value, "white")

            polygon = patches.Polygon(vertices, closed=True, color=polygon_color)
            ax.add_patch(polygon)

            cluster_label = cluster if is_continuous else value

            if not is_continuous and cluster_label not in legend_patches:
                legend_patches[cluster_label] = patches.Patch(color=polygon_color, label=cluster_label)

    return legend_patches if not is_continuous else None

def highlight_selected_plaque(ax, plaque_coord, tissue_id):
    """
    Highlight the selected plaque in the plot.

    Args:
        ax (matplotlib.axes.Axes): The axis to highlight the plaque on.
        plaque_coord (pandas.DataFrame): DataFrame containing plaque coordinates.
        tissue_id (str): The tissue identifier to highlight.

    Returns:
        matplotlib.axes.Axes: The axis with the highlighted plaque.
    """
    selected_plaque = plaque_coord[plaque_coord['tissue'] == tissue_id]
    centroid = selected_plaque[['row', 'col']].mean()

    x_lim = (centroid['col'] - 500, centroid['col'] + 500)
    y_lim = (centroid['row'] - 500, centroid['row'] + 500)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    return ax

def draw_viewing_region(ax, color='black', linewidth=1):
    """
    Draw the outline box representing the viewing region.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw the outline on.
        color (str): The color of the outline.
        linewidth (int): The width of the outline.

    Returns:
        matplotlib.axes.Axes: The axis with the viewing region outline.
    """
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    width = x_limits[1] - x_limits[0]
    height = y_limits[1] - y_limits[0]
    outline_box = Rectangle((x_limits[0], y_limits[0]), width, height,
                            edgecolor=color, facecolor='none', lw=linewidth)

    ax.add_patch(outline_box)

    return ax

def process_ndarray(ndarray):
    """
    Process ndarray to filter out invalid values.

    Args:
        ndarray (numpy.ndarray): The ndarray to process.

    Returns:
        numpy.ndarray: The filtered ndarray.
    """
    mask = ~np.any(ndarray == 32767, axis=1)
    return ndarray[mask]

def plot_legend(ax, legend_patches):
    """
    Add a legend to the axis for discrete values.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the legend to.
        legend_patches (dict): Dictionary of legend patches.

    Returns:
        None
    """
    if legend_patches:
        ax.legend(handles=legend_patches.values(), loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)

def plot_colorbar(ax, cmap, norm):
    """
    Add colorbar for continuous values.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the colorbar to.
        cmap (matplotlib.colors.Colormap): The colormap for the colorbar.
        norm (matplotlib.colors.Normalize): The normalization for the colorbar.

    Returns:
        None
    """
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

def set_axes_view(ax, x_start, y_start, win_size):
    """
    Set the view window based on the win_size.

    Args:
        ax (matplotlib.axes.Axes): The axis to set the view on.
        x_start (float): The starting x-coordinate.
        y_start (float): The starting y-coordinate.
        win_size (float): The size of the view window.

    Returns:
        None
    """
    if win_size:
        ax.set_xlim([x_start, x_start + win_size])
        ax.set_ylim([y_start, y_start + win_size])

def set_axes_style(ax):
    """
    Set axes styles.

    Args:
        ax (matplotlib.axes.Axes): The axis to set the styles on.

    Returns:
        None
    """
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.subplots_adjust(right=0.75)

def plot_combined_v5(ax, cell_data_dict, obs, color_palette, plaque_coord, value_column, is_continuous):
    """
    Main function for plotting with continuous or discrete coloring.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        cell_data_dict (dict): Dictionary with cell data arrays.
        obs (pandas.DataFrame): DataFrame containing cell observations.
        color_palette (dict): Dictionary mapping discrete values to colors.
        plaque_coord (pandas.DataFrame): DataFrame containing plaque coordinates.
        value_column (str): Column name in `obs` containing the values for coloring.
        is_continuous (bool): Whether the values are continuous.

    Returns:
        matplotlib.axes.Axes: The axis with the plotted data.
    """
    plaque_set = set(zip(plaque_coord['row'], plaque_coord['col']))
    legend_patches = plot_cells(ax, obs, cell_data_dict, color_palette, plaque_set, value_column, is_continuous)
    if not is_continuous:
        plot_legend(ax, legend_patches)
    else:
        norm = Normalize(vmin=obs[value_column].min(), vmax=obs[value_column].max())
        plot_colorbar(ax, cmp.PuRd, norm)
    set_axes_style(ax)
    return ax

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
                                 linewidth=1, edgecolor='black', facecolor='black', **kwargs)
    ax.add_patch(scalebar)
    
    ax.text(xpos + length_units / 2, ypos + (ylim[1] - ylim[0]) * 0.02, label,
            color='black', ha='center', va='bottom', fontsize=10)

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

def add_contours(ax, obs, value_column, contour_cmap, levels=10, vmin=None, 
                 vmax=None, sigma=1.0, padding=0.05, scatter=False, alpha = 0.5, 
                 line_color = "white", line_width = 1, value_lable = False):
    """
    Add contours to the plot.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the contours to.
        obs (pandas.DataFrame): DataFrame containing cell observations.
        value_column (str): Column name in `obs` containing the values for contouring.
        contour_cmap (str): Colormap for the contours.
        levels (int): Number of contour levels.
        vmin (float): Minimum value for the contours.
        vmax (float): Maximum value for the contours.
        sigma (float): Standard deviation for Gaussian filter.
        padding (float): Padding around the data for contouring.
        scatter (bool): Whether to scatter plot the points.

    Returns:
        None
    """
    x_pad = padding * (obs['x'].max() - obs['x'].min())
    y_pad = padding * (obs['y'].max() - obs['y'].min())

    grid_x, grid_y = np.mgrid[(obs['x'].min() - x_pad):(obs['x'].max() + x_pad):150j, 
                              (obs['y'].min() - y_pad):(obs['y'].max() + y_pad):150j]
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    scaled_values = obs[value_column]

    grid_z = griddata((obs['x'], obs['y']), scaled_values, (grid_x, grid_y), method='nearest')

    grid_z_fil = filter_grid(grid_z, obs, grid_points)

    # Record positions of 0 values in grid_z_fil
    zero_positions = np.where(grid_z_fil == 0)
    
    grid_z = gaussian_filter(grid_z_fil, sigma=sigma)
    
    # Reset the 0 positions back to 0
    grid_z[zero_positions] = None

    # Standardize grid_z using Z-score
    grid_z_mean = np.nanmean(grid_z)  # Calculate the mean ignoring NaNs
    grid_z_std = np.nanstd(grid_z)    # Calculate the standard deviation ignoring NaNs
    grid_z = (grid_z - grid_z_mean) / grid_z_std  # Z-score standardization

    if vmin is None and vmax is None:
        contours = ax.contour(grid_x, grid_y, grid_z, levels=levels, linewidths=line_width, colors=line_color) #, vmin=vmin, vmax=vmax)
        ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=contour_cmap, alpha=alpha) #, vmin=vmin, vmax=vmax)
    else:
        contours = ax.contour(grid_x, grid_y, grid_z, levels=levels, linewidths=line_width, colors=line_color, vmin = vmin, vmax = vmax) #, vmin=vmin, vmax=vmax)
        ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=contour_cmap, alpha=alpha, vmin = vmin, vmax = vmax) #, vmin=vmin, vmax=vmax)

    

    if value_lable == True:
        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')














