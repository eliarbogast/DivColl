import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

fig, axs = plt.subplots(2, 3, figsize=(12.5, 10))
# Add invisible axes for row titles
for row, title in enumerate(
        ["Connectors trained and tested on diverse pairs", "Connectors trained and tested on uniform pairs"]):
    ax_title = fig.add_subplot(2, 3, row * 3 + 2, frame_on=False)
    ax_title.axis('off')
    ax_title.set_title(title, fontweight='bold', fontsize=16, pad=25)

counter = -1
row_titles = ["Connectors trained and tested on diverse pairs", "Connectors trained and tested on uniform pairs"]
for row in range(2):
    X_uni_sep_transformed = np.load(f"pacmap_data/X_uni_sep_transformed_{row}.npy")
    X_div_sep_transformed = np.load(f"pacmap_data/X_div_sep_transformed_{row}.npy")


    def find_overlap_2d_numpy(array1, array2):
        set1 = set(map(tuple, array1.round(0)))
        set2 = set(map(tuple, array2.round(0)))
        overlap = np.array(list(set1.intersection(set2)))
        return overlap


    def find_nonoverlap_2d_numpy(array1, array2):
        set1 = set(map(tuple, array1.round(0)))
        set2 = set(map(tuple, array2.round(0)))
        overlap = np.array(list(set1.difference(set2)))
        return overlap


    def normalize_opacity(counts, min_count, max_count, min_alpha=0.05, max_alpha=1.0):
        if len(counts) == 0:
            return counts
        if min_count == max_count:
            return np.full_like(counts, max_alpha)
        norm = (counts - min_count) / (max_count - min_count)
        return norm * (max_alpha - min_alpha) + min_alpha


    # Round both arrays to the same precision
    A = np.round(X_uni_sep_transformed)
    B = np.round(X_div_sep_transformed)

    # Convert to tuples for set operations
    A_tuples = list(map(tuple, A))
    B_tuples = list(map(tuple, B))
    # Count occurrences in each array
    count_uni = Counter(A_tuples)
    count_div = Counter(B_tuples)

    # Sets for overlap and non-overlap
    set_uni = set(count_uni.keys())
    set_div = set(count_div.keys())
    overlap_set = set_uni & set_div
    only_uni_set = set_uni - set_div
    only_div_set = set_div - set_uni

    # Unique points as arrays
    X_both = np.array(list(overlap_set))
    X_only_uni_conn = np.array(list(only_uni_set))
    X_only_div_conn = np.array(list(only_div_set))

    # Opacity arrays (sum counts for overlap)
    opacity_both = np.array([count_uni[pt] + count_div[pt] for pt in overlap_set])
    opacity_only_uni = np.array([count_uni[pt] for pt in only_uni_set])
    opacity_only_div = np.array([count_div[pt] for pt in only_div_set])

    smallest_c = min(min(min(opacity_both), min(opacity_only_uni)), min(opacity_only_div))
    highest_c = max(max(max(opacity_both), max(opacity_only_uni)), max(opacity_only_div))

    if row == 0:
        smallest_c_div = smallest_c
        highest_c_div = highest_c
    else:
        smallest_c_uni = smallest_c
        highest_c_uni = highest_c

    alpha_both = normalize_opacity(opacity_both, smallest_c, highest_c)
    alpha_only_uni = normalize_opacity(opacity_only_uni, smallest_c, highest_c)
    alpha_only_div = normalize_opacity(opacity_only_div, smallest_c, highest_c)

    if row == 0:
        edgecol = "#0072b2"
        ls = "solid"
    else:
        edgecol = "#e69f00"
        ls = "dashed"

    axs[row, 0].scatter(X_only_uni_conn[:, 0], X_only_uni_conn[:, 1], c=edgecol, alpha=alpha_only_uni, s=17, marker='s',
                        edgecolor=edgecol,
                        linestyle=ls, linewidths=0.0)
    axs[row, 0].set_title('Seen only in training environment', fontsize=12)
    axs[row, 0].set_xlabel('PacMAP dimension 1', fontsize=12);
    axs[row, 0].set_ylabel('PacMAP dimension 2', fontsize=12)
    axs[row, 0].set_xlim(-38, 38)

    # Middle: Both
    axs[row, 1].scatter(X_both[:, 0], X_both[:, 1], c=edgecol, alpha=alpha_both, s=17, marker='s', edgecolor=edgecol,
                        linestyle=ls, linewidths=0.0)
    axs[row, 1].set_title('Seen in both training and wind environments', fontsize=12)
    axs[row, 1].set_xlabel('PacMAP dimension 1', fontsize=12)
    axs[row, 1].set_xlim(-38, 38)

    # Right: Only in test environment
    axs[row, 2].scatter(X_only_div_conn[:, 0], X_only_div_conn[:, 1], c=edgecol, alpha=alpha_only_div, s=17, marker='s',
                        edgecolor=edgecol,
                        linestyle=ls, linewidths=0.0)
    axs[row, 2].set_title('Seen only in wind environment', fontsize=12)
    axs[row, 2].set_xlabel('PacMAP dimension 1', fontsize=12)
    axs[row, 2].set_xlim(-38, 38)

for row in range(2):
    for col in range(3):
        ax = axs[row, col]

        if row == 0:  # diverse collectives
            color = "#0072b2"
            linestyle = 'solid'
        else:  # uniform collectives
            color = "#e69f00"
            linestyle = (0, (5, 3))

        # Set spines color and linestyle
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linestyle(linestyle)
            spine.set_linewidth(2)

import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Colorbar properties
opacity_norm = mcolors.Normalize(vmin=0, vmax=1)

# Create custom colormaps that linearly interpolate from white to blue/orange
blue_cmap = mcolors.LinearSegmentedColormap.from_list("bice_blue_cmap",
                                                      [(1, 1, 1), (0.0, 114 / 255, 178 / 255)])

orange_cmap = mcolors.LinearSegmentedColormap.from_list("harvest_gold_cmap",
                                                        [(1, 1, 1), (230 / 255, 159 / 255, 0.0)])

# Define the bottom positions for colorbar - span across all three panels
left = axs[0, 0].get_position().x0
right = axs[0, 2].get_position().x1
width = right - left
height = 0.02  # Adjust height of colorbar

# Top row horizontal colorbar
cbar_ax_top = fig.add_axes([left, 0.09, width, height])
sm_top = cm.ScalarMappable(norm=opacity_norm, cmap=blue_cmap)
sm_top.set_array([])
cb_top = fig.colorbar(sm_top, cax=cbar_ax_top, orientation='horizontal')
cb_top.set_ticks([0, 0.25, 0.5, 0.75, 1])
cb_top.set_ticklabels([])

# Bottom row horizontal colorbar
left2 = axs[1, 0].get_position().x0
right2 = axs[1, 2].get_position().x1
width2 = right2 - left2
cbar_ax_bottom = fig.add_axes([left2, 0.05, width2, height])
sm_bottom = cm.ScalarMappable(norm=opacity_norm, cmap=orange_cmap)
sm_bottom.set_array([])
cb_bottom = fig.colorbar(sm_bottom, cax=cbar_ax_bottom, orientation='horizontal')
cb_bottom.set_ticks([0, 0.25, 0.5, 0.75, 1])
cb_bottom.set_ticklabels([str(int(x)) for x in np.linspace(smallest_c_uni, highest_c_uni, 5)])
cb_bottom.set_label('State Frequency', fontsize=12)

plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space at bottom for colorbar

plt.show()
