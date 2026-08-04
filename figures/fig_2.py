import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymer4.models import Lmer
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

def add_compact_significance(ax, x_indices, y_pos, p_values, bracket_height, bar_tops, fontsize=20):
    """Draws a compact significance bar with one horizontal bar and connector lines to each compared bar."""
    # Draw horizontal bar spanning all compared bars
    min_x, max_x = min(x_indices), max(x_indices)
    ax.plot([min_x, max_x], [y_pos, y_pos], color='black', linewidth=1.5)

    # Draw vertical connectors down to tops of each compared bar
    for xi, bar_top in zip(x_indices, bar_tops):
        ax.plot([xi, xi], [y_pos, bar_top + bracket_height], color='black', linewidth=1.5)

    # Place asterisks above each connector (or next to)
    for xi, p_value in zip(x_indices, p_values):
        if p_value is None:
            continue  # Don't annotate the reference bar
        if p_value < 0.001:
            text = '***'
        elif p_value < 0.01:
            text = '**'
        elif p_value < 0.05:
            text = '*'
        else:
            text = 'n.s.'
        # Place above horizontal bar or at the top right of connector
        ax.text(xi, y_pos + bracket_height * 0.3, text, ha='center', va='bottom', fontsize=fontsize)

def add_significance_bar(ax, x1, x2, y, p_value, bracket_height, fontsize=20):
    """Draws a significance bar with asterisks above two bars."""
    ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color='black', linewidth=1.5)
    offset = 0.15
    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'n.s.'
        offset = 0.95
        fontsize = 15
    ax.text((x1 + x2) / 2, y + bracket_height * offset, text, ha='center', va='bottom', fontsize=fontsize)

plt.style.use('tableau-colorblind10')

df = pd.read_csv("csvs/performance.csv")
# Load body length data
lengths = pd.read_csv('csvs/body_lengths.csv')

# Merge to get body length for id_1
df = df.merge(lengths.rename(columns={'id': 'id_1', 'length': 'length_1'}), on='id_1', how='left')
# Merge to get body length for id_2
df = df.merge(lengths.rename(columns={'id': 'id_2', 'length': 'length_2'}), on='id_2', how='left')

# Compute normalized performance
df['bl_perf_1'] = df['perf_1'] / df['length_1']
df['bl_perf_2'] = df['perf_2'] / df['length_2']

# Compute the average
df['bl_avg'] = df[['bl_perf_1', 'bl_perf_2']].mean(axis=1)

fig, axs = plt.subplots(7, 2, figsize=(30, 25), constrained_layout=True,
                        gridspec_kw={'width_ratios': [1.5, 1.5]})
fig.get_layout_engine().set(rect=(0.025, 0.025, 0.95, 0.95))
fig.set_constrained_layout_pads(w_pad=10. / 72., h_pad=10. / 72., wspace=0.02, hspace=0.02)

# Colorblind-friendly palette
color_div = '#0072b2'  # blue
color_uni = '#e69f00'  # orange
color_sep = 'white'

title_dict = {"in_dist": "Training", "obs": "Obstacle", "slope_down": "Decline", "slope_up": "Incline",
              "step_down": "Step down", "step_up": "Step up", "gap": "Gap", "ice": "Ice", "sticky": "Sticky ground",
              "wind": "Wind", "sand": "Sand", "treadmill": "Treadmill", "low_gravity": "Low gravity"}

letter_dict = {"in_dist": "B", "obs": "F", "slope_down": "E", "slope_up": "I",
              "step_down": "D", "step_up": "G", "gap": "H", "ice": "L", "sticky": "M",
              "wind": "K", "sand": "C", "treadmill": "J", "low_gravity": "N"}
counter = 0
for specific_env in ["in_dist", "sand", "step_down", "slope_down", "obs","step_up", "gap", "slope_up", "treadmill",
                     "wind", "ice","sticky", "low_gravity"]:
    counter += 1
    row_counter = counter // 2
    col_counter = counter % 2
    ax = axs[row_counter, col_counter]
    ax.set_facecolor("white")
    df_filtered = df[df['env'] == specific_env]

    # Grouped bar order: diverse1, diverse2, uniform1, uniform2
    # Clustered bar positions in the middle
    x = [2.58, 2.7, 2.82, 2.94]  # Bars are tightly packed in the center

    means = [
        df_filtered[(df_filtered['setting'] == '_conn_uni') & (df_filtered['trained_on'] == 'uni')]['bl_avg'].mean(),
        df_filtered[(df_filtered['setting'] == '_conn_div') & (df_filtered['trained_on'] == 'uni')]['bl_avg'].mean(),
        df_filtered[(df_filtered['setting'] == '_conn_uni') & (df_filtered['trained_on'] == 'div')]['bl_avg'].mean(),
        df_filtered[(df_filtered['setting'] == '_conn_div') & (df_filtered['trained_on'] == 'div')]['bl_avg'].mean(),
    ]
    errors = [
        df_filtered[(df_filtered['setting'] == '_conn_uni') & (df_filtered['trained_on'] == 'uni')]['bl_avg'].std() /
        np.sqrt(df_filtered[(df_filtered['setting'] == '_conn_uni') & (df_filtered['trained_on'] == 'uni')][
                    'bl_avg'].count()),
        df_filtered[(df_filtered['setting'] == '_conn_div') & (df_filtered['trained_on'] == 'uni')]['bl_avg'].std() /
        np.sqrt(df_filtered[(df_filtered['setting'] == '_conn_div') & (df_filtered['trained_on'] == 'uni')][
                    'bl_avg'].count()),
        df_filtered[(df_filtered['setting'] == '_conn_uni') & (df_filtered['trained_on'] == 'div')]['bl_avg'].std() /
        np.sqrt(df_filtered[(df_filtered['setting'] == '_conn_uni') & (df_filtered['trained_on'] == 'div')][
                    'bl_avg'].count()),
        df_filtered[(df_filtered['setting'] == '_conn_div') & (df_filtered['trained_on'] == 'div')]['bl_avg'].std() /
        np.sqrt(df_filtered[(df_filtered['setting'] == '_conn_div') & (df_filtered['trained_on'] == 'div')][
                    'bl_avg'].count()),
    ]
    bar_colors = [color_uni, color_uni, color_div, color_div]

    bars = ax.bar(
        x, means, yerr=errors,
        color=bar_colors, capsize=8, edgecolor='black', width=0.08
    )

    # Custom border line styles for each bar
    for p_idx, patch in enumerate(bars):
        patch.set_linewidth(4)
        if p_idx == 0:  # e.g. First bar: dotted
            patch.set_linestyle('--')
        elif p_idx == 2:  # Third bar: dashed
            patch.set_linestyle('--')

    # Reference line still in place
    unpaired_mean = df_filtered[df_filtered['setting'] == '_not_conn_uni']['bl_avg'].mean()
    ax.axhline(unpaired_mean, color='black', linestyle=':', linewidth=3, xmin=0.38, xmax=0.58)

    # Tick marks at bar centers, labels in your desired order
    ax.set_xticklabels([])
    ax.set_xlim(1.5, 4.1)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=20,length=0)

    # Group pair indices now point to the bars above
    bar_indices = {'_conn_uni_uni': 0, '_conn_div_uni': 1, '_conn_uni_div': 2, '_conn_div_div': 3}

    # Compare all other groups against reference group ('_conn_div', 'div')
    ref_setting = '_conn_div'
    ref_train = 'div'
    ref_idx = bar_indices[ref_setting + '_' + ref_train]
    ref_x = x[ref_idx]
    ref_top = means[ref_idx] + errors[ref_idx]

    group_pairs = [
        ((ref_setting, ref_train), ('_conn_uni', 'div')),
        ((ref_setting, ref_train), ('_conn_div', 'uni')),
        ((ref_setting, ref_train), ('_conn_uni', 'uni')),
    ]
    bracket_tops = []
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    bracket_offset = 0.1 * y_range
    bracket_spacing = 0.12 * y_range
    bracket_height = 0.04 * y_range

    x_indices = [ref_x]
    bar_tops = [ref_top]
    p_values = [None]

    for (ref, comp) in group_pairs:
        g1, t1 = ref
        g2, t2 = comp
        this_df = df_filtered[
            ((df_filtered['setting'] == g1) & (df_filtered['trained_on'] == t1)) |
            ((df_filtered['setting'] == g2) & (df_filtered['trained_on'] == t2))
            ].dropna(subset=['bl_avg', 'id_1', 'id_2'])
        if this_df.empty:
            continue
        this_df['robot1'] = this_df['id_1']
        this_df['robot2'] = this_df['id_2']
        this_df['group_label'] = this_df['trained_on'] + this_df['setting']

        formula = 'bl_avg ~ group_label + (1|robot1) + (1|robot2)'
        model = Lmer(formula, data=this_df)
        result = model.fit()
        p_value = result["P-val"][1]

        comp_idx = bar_indices[g2 + '_' + t2]
        x_indices.append(x[comp_idx])  # x-location of comparison bar
        bar_tops.append(means[comp_idx] + errors[comp_idx])  # top of comparison bar
        p_values.append(p_value)

        # y-position for compact bar: above all tops by some offset
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        y_pos = max(bar_tops) + 0.1 * y_range
        bracket_height = 0.04 * y_range

    if len(x_indices) > 1:
        add_compact_significance(ax, x_indices, y_pos, p_values, bracket_height, bar_tops)

    final_ymax = max(bracket_tops) + 0.06 * y_range if bracket_tops else y_max
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    leftmost_bar_x = 2.5
    ax.spines['left'].set_position(('data', leftmost_bar_x))

    # Optional: leave spine visible, or make sure it's not invisible
    ax.spines['left'].set_visible(True)

    # Optionally, move tick labels so they don't overlap with the bar
    ax.yaxis.set_label_coords(leftmost_bar_x - 0.05, 0.5)

    # Optionally, extend ticks a little away from the bar to avoid overlap
    ax.tick_params(axis='y', pad=10)  # 10 pts away from spine

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Coordinates for top right (small offset from boundaries)
    text_x = xlim[1] - 0.25 * (xlim[1] - xlim[0])
    text_y = ylim[1] - 0.1 * (ylim[1] - ylim[0])

    text_x_2 = xlim[0] + 0.25 * (xlim[1] - xlim[0])
    text_y_2 = ylim[1] - 0.1 * (ylim[1] - ylim[0])

    ax.text(
            text_x, text_y,
            title_dict[specific_env],
            ha='center', va='top',
            fontsize=28, fontweight='bold'
        )

    ax.text(
            text_x_2, text_y_2,
            letter_dict[specific_env],
            ha='right', va='top',
            fontsize=28, fontweight='bold'
        )

axs[0, 0].axis('off')
plt.show()
