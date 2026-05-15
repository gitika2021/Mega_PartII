import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def plot_images_from_file(filename, savefig_as = None, n_cols = 10, suptitle=None):
    images = np.load(filename)
   
    n_total = images.shape[0]
    #n_cols = 10                                # number of images per row
    n_groups = int(np.ceil(n_total / n_cols))  # number of full groups
    
    for group in range(n_groups):
        start = group * n_cols
        end = min(start + n_cols, n_total)
        count = end - start
    
        fig, axes = plt.subplots(2, count, figsize=(count * 1.2, 2.5), constrained_layout=True)
    
        for i in range(count):
            # Plot image
            ax_img = axes[0, i] if count > 1 else axes[0]
            ax_img.imshow(images[start + i], cmap='viridis')
            #ax_img.set_title(f"{kepnames[start + i]}\n$R_p/R_s$: {rp_rs_ratio[start + i]:.2f}\n$snr$: {snr[start + i]:.2f}", fontsize=8)
            ax_img.axis('off')
    
            # Plot 1D profile
            ax_prof = axes[1, i] if count > 1 else axes[1]
            # ax_prof.imshow(predictions[start + i], cmap='viridis')
            # ax_prof.set_xticks([])
            # ax_prof.set_yticks([])
            ax_prof.set_visible(False)
        if suptitle is not None:    
            plt.suptitle(f"Images and 1D Profiles: Group {group+1}/{n_groups}", fontsize=12)
        plt.tight_layout()
        if savefig_as is not None:
            plt.savefig(f'{savefig_as}.png')
        plt.show()


def plot_curves_from_file(filename, savefig_as = None, n_cols = 10, suptitle=None):
    images = np.load(filename)
    phase = np.linspace(-0.5,0.5,images.shape[1])
    n_total = images.shape[0]
    #n_cols = 10                                # number of images per row
    n_groups = int(np.ceil(n_total / n_cols))  # number of full groups
    
    for group in range(n_groups):
        start = group * n_cols
        end = min(start + n_cols, n_total)
        count = end - start
    
        fig, axes = plt.subplots(2, count, figsize=(count * 1.2, 2.5), constrained_layout=True)
    
        for i in range(count):
            # Plot image
            ax_img = axes[0, i] if count > 1 else axes[0]
            # ax_img.imshow(images[start + i], cmap='viridis')
            #ax_img.set_title(f"{kepnames[start + i]}\n$R_p/R_s$: {rp_rs_ratio[start + i]:.2f}\n$snr$: {snr[start + i]:.2f}", fontsize=8)
            # ax_img.axis('off')
            ax_img.set_visible(False)
    
            # Plot 1D profile
            ax_prof = axes[1, i] if count > 1 else axes[1]
            #ax_prof.plot(images[start + i], '.-k')
            ax_prof.scatter(phase,images[start + i], color='blue',s=5)
            # ax_prof.set_xticks([])
            # ax_prof.set_yticks([])
            #ax_prof.set_visible(False)
            # ax_prof.set_xlabel('Phase')
            # ax_prof.set_ylabel('Normalized Flux')
            plt.suptitle(f"Images and 1D Profiles: Group {group+1}/{n_groups}", fontsize=12)
        # if suptitle is None:  
        #     ax_prof.set_xticks([])
        #     ax_prof.set_yticks([])
        #     plt.suptitle(f"Images and 1D Profiles: Group {group+1}/{n_groups}", fontsize=12)
        plt.tight_layout()
        if savefig_as is not None:
            plt.savefig(f'{savefig_as}.png')

        # fig.text(0.5, 0.02, "Phase-Folded Light Curves", ha='center')
        # fig.text(0.02, 0.5, "Normalized Fluxes", va='center', rotation='vertical')

        plt.show()


def scatter_with_half_slanted_bg(
    nrows=1,
    ncols=1,
    figsize=(6, 6),
    xlim=None,
    ylim=None,
    line_spacing=0.08,
    line_width=2,
    line_color="lightgray",
    split=0.5,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax in axes.flat:

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Blank half
        ax.add_patch(
            Rectangle(
                (0 if xlim is None else xlim[0], 
                 0 if ylim is None else ylim[0]),
                (1 if xlim is None else (xlim[1] - xlim[0]) * split),
                1 if ylim is None else (ylim[1] - ylim[0]),
                facecolor="white",
                edgecolor="none",
                zorder=0, alpha=0.5
            )
        )

        # Slanted lines (draw in axes coordinates)
        for i in np.arange(-1, 2, line_spacing):
            ax.add_line(
                Line2D(
                    [split, 1.5],
                    [i, i + 1],
                    transform=ax.transAxes,
                    linewidth=line_width,
                    color=line_color,
                    zorder=0, alpha=0.5
                )
            )

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_frame_on(False)

    return fig, axes
    
def add_scatter(ax, x, y, color="tab:blue", size=80, zorder=2,label='', **kwargs):
    ax.scatter(
        x,
        y,
        s=size,
        c=color,
        alpha=0.5,
        zorder=zorder,
        label = label,
        **kwargs,
    )

    # call like this
    # x = np.random.rand(30)
    # y = np.random.rand(30)
    # print(x.shape, y.shape, type(x), type(kepler_lcs_snr.to_numpy()),kepler_lcs_snr.to_numpy().shape)
    # fig, axes = scatter_with_half_slanted_bg(
    #     nrows=1,
    #     ncols=2,
    #     figsize=(10, 5),
    #     line_width=0.5,
    #     split=0,
    #     xlim=(median_error.min()-median_error.min()*0.25,median_error.max()+median_error.max()*0.25),
    #     ylim=(kepler_lcs_snr.to_numpy().min()-kepler_lcs_snr.to_numpy().min()*0.1,kepler_lcs_snr.to_numpy().max()+kepler_lcs_snr.to_numpy().max()*0.1),
    # )
    
    # add_scatter(axes[0, 0], median_error, kepler_lcs_snr, color="crimson", size=10)
    # add_scatter(axes[0, 1], median_error, kepler_lcs_depth, color="navy", size=20)
    
    # axes[0, 0].set_yscale('log')
    # axes[0, 1].set_xlim([-0.00001,0.001])
    # axes[0, 1].set_ylim([-0.001,0.04])
    # plt.show()


