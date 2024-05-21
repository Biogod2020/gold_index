import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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

        sc.pl.umap(tmp[mask], color=plot_color, vmin=0, vmax=1 if color == "min_center_dist" else None, ax=axes[i], show=False, colorbar_loc=None, legend_loc=leg_stat, palette=palette, **umap_kwargs)
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
