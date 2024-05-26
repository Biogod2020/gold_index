import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

def plot_normalized_umap(
    slide: dict,
    num_rows: int = 1,
    num_cols: int = 7,
    cell_type: str = "Micro",
    color: str = "min_center_dist",
    palette: dict = None,
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


def plot_kde_normalized_distance(
    pfcdf: pd.DataFrame,
    cell_type: str = "Micro",
    dist_col: str = "min_center_dist",
    fine_col: str = "fine",
    palette: dict = None,
    save_path: str = None,
    fig_size: tuple = (8, 4),
    dpi: int = 350,
    scaling_method: str = "standard",  # Parameter to choose the scaling method
    **kde_kwargs
) -> plt.Figure:
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
        **kde_kwargs: Arbitrary keyword arguments to pass to the `sns.kdeplot` function.

    Returns:
        plt.Figure: The matplotlib figure object.
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
    
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    mask = (pfcdf["celltype"] == cell_type) & (pfcdf[dist_col] > 0)
    filtered_data = pfcdf[mask]
    filtered_data = filtered_data.copy()
    filtered_data["fine"] = filtered_data[fine_col].astype(str)

    # Scale the distance values
    filtered_data["dist_scaled"] = scale_values(filtered_data[dist_col].values, scaling_method)
    filtered_data["Normalized_and_log_transformed_Distance"] = -np.log10(filtered_data["dist_scaled"] + 1e-3)
    plot_color = "Normalized_and_log_transformed_Distance"

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
    
    return fig
