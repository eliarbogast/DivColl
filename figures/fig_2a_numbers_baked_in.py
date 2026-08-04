import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymer4.models import Lmer
import seaborn as sns

def add_compact_significance(ax, x_indices, y_pos, p_values, bracket_height, bar_tops, fontsize=20):
    """Draws a compact significance bar with one horizontal bar and connector lines to each compared bar."""
    min_x, max_x = min(x_indices), max(x_indices)
    ax.plot([min_x, max_x], [y_pos, y_pos], color='black', linewidth=1.5)
    for xi, bar_top in zip(x_indices, bar_tops):
        ax.plot([xi, xi], [y_pos, bar_top + bracket_height], color='black', linewidth=1.5)
    for xi, p_value in zip(x_indices, p_values):
        if p_value is None:
            continue
        if p_value < 0.001:
            text = '***'
        elif p_value < 0.01:
            text = '**'
        elif p_value < 0.05:
            text = '*'
        else:
            text = 'n.s.'
        ax.text(xi, y_pos + bracket_height * 0.3, text, ha='center', va='bottom', fontsize=fontsize)

# ------------------- DATA SETUP -------------------
df_loose = pd.read_csv("csvs/performance.csv")
# Load body length data
lengths = pd.read_csv('csvs/body_lengths.csv')
# Merge to get body length for id_1
df_loose = df_loose.merge(lengths.rename(columns={'id': 'id_1', 'length': 'length_1'}), on='id_1', how='left')
# Merge to get body length for id_2
df_loose = df_loose.merge(lengths.rename(columns={'id': 'id_2', 'length': 'length_2'}), on='id_2', how='left')
# Compute normalized performance
df_loose['bl_perf_1'] = df_loose['perf_1'] / df_loose['length_1']
df_loose['bl_perf_2'] = df_loose['perf_2'] / df_loose['length_2']
# Compute the average
df_loose['bl_avg'] = df_loose[['bl_perf_1', 'bl_perf_2']].mean(axis=1)

df = pd.read_csv("csvs/performance_tight.csv", encoding="latin1")

# Merge body lengths for both id_1 and id_2
df = df.merge(lengths.rename(columns={'id': 'id_1', 'length': 'length_1'}), on='id_1', how='left')
df = df.merge(lengths.rename(columns={'id': 'id_2', 'length': 'length_2'}), on='id_2', how='left')
df['bl_perf_1'] = df['perf_1'] / df['length_1']
df['bl_perf_2'] = df['perf_2'] / df['length_2']
df['bl_avg'] = df[['bl_perf_1', 'bl_perf_2']].mean(axis=1)

bar_width = 0.7
narrow_width = 0.23
offset = 0.17
color_uni = '#e69f00'
color_div = '#0072b2'

# Main bars metadata
main_bar_info = [
    # label,           trained_on, coupling_strength, color
    ("Loose Uni",      "uni",      "loose",          color_uni),
    ("Loose Div",      "div",      "loose",          color_div),
    ("Tight Uni",      "uni",      "tight",          color_uni),
    ("Tight Div",      "div",      "tight",          color_div),
]
bar_positions = [0, 1, 3, 4]
labels = [info[0] for info in main_bar_info]

# Extract main in_dist bar heights
main_avgs = [
    df.query("env == 'in_dist' and trained_on == @trained_on and coupling_strength == @coupling_strength")["bl_avg"].mean()
    for (_, trained_on, coupling_strength, _) in main_bar_info
]

# Define narrow bar groupings and styles
narrow_bar_info = []
for i, (label, trained_on, coupling_strength, color) in enumerate(main_bar_info):
    for setting, linestyle, lw in [("_conn_uni", "--", 2 if coupling_strength=="loose" else 5),
                                   ("_conn_div", "-", 2 if coupling_strength=="loose" else 5)]:
        narrow_bar_info.append({
            "main_bar_idx": i,
            "setting": setting,
            "trained_on": trained_on,
            "coupling_strength": coupling_strength,
            "color": 'black',
            "linestyle": linestyle,
            "linewidth": lw,
            "env": "ood", # indicates all envs except in_dist
        })

# Calculate OOD narrow bar heights and SEMs
for nb in narrow_bar_info:
    ood = df[
        (df["env"] != "in_dist") &
        (df["setting"] == nb["setting"]) &
        (df["trained_on"] == nb["trained_on"]) &
        (df["coupling_strength"] == nb["coupling_strength"])
        ]
    nb["mean"] = ood["bl_avg"].mean()
    nb["sem"] = ood["bl_avg"].std() / np.sqrt(ood["bl_avg"].count()) if ood["bl_avg"].count() > 0 else 0

# Reference horizontal lines
line1 = df_loose[(df_loose['env'] == 'in_dist') & (df_loose['setting'] == '_not_conn_uni')]['bl_avg'].mean()
line2 = df_loose[(df_loose['env'] != 'in_dist') & (df_loose['setting'] == '_not_conn_uni')]['bl_avg'].mean()

# ------------------- PLOTTING -------------------

fig, ax = plt.subplots(figsize=(13, 2.5))

# Plot main bars
for i, (label, trained_on, coupling_strength, color) in enumerate(main_bar_info):
    ax.bar(bar_positions[i], main_avgs[i], color=color, width=bar_width)

# Plot narrow (nested) bars
nested_xs = []
nested_tops = []
for i, nb in enumerate(narrow_bar_info):
    x_pos = bar_positions[nb["main_bar_idx"]] - offset + (0 if nb["setting"] == "_conn_uni" else offset * 2)
    bottom = 0
    bar = ax.bar(
        x_pos, nb["mean"], width=narrow_width,
        bottom=bottom, color='none', edgecolor=nb["color"],
        linestyle=nb["linestyle"], linewidth=nb["linewidth"],
        yerr=nb["sem"], capsize=4, zorder=5
    )
    nested_xs.append(x_pos)
    nested_tops.append(nb["mean"] + nb["sem"])

# ------------------- SIGNIFICANCE TESTING -------------------

# Compact significance bar: reference is _conn_div, div, loose
reference_idx = 3
ref_nb = narrow_bar_info[reference_idx]
ref_label = "ref"
x_indices = [nested_xs[reference_idx]]
bar_tops = [nested_tops[reference_idx]]
p_values = [None]  # No annotation for reference itself

for i, nb in enumerate(narrow_bar_info):
    if i == reference_idx:
        continue
    # Pull data for ref
    ref_mask = (
        (df["env"] != "in_dist") &
        (df["setting"] == ref_nb["setting"]) &
        (df["trained_on"] == ref_nb["trained_on"]) &
        (df["coupling_strength"] == ref_nb["coupling_strength"])
    )
    comp_mask = (
        (df["env"] != "in_dist") &
        (df["setting"] == nb["setting"]) &
        (df["trained_on"] == nb["trained_on"]) &
        (df["coupling_strength"] == nb["coupling_strength"])
    )
    comp_df = pd.concat([
        df[ref_mask].assign(group_label=ref_label),
        df[comp_mask].assign(group_label="comp_"+str(i))
    ], ignore_index=True)
    # Remove rows missing data/IDs
    comp_df = comp_df.dropna(subset=["bl_avg", "id_1", "id_2"])
    if len(comp_df) < 5:
        p_values.append(None)
        x_indices.append(nested_xs[i])
        bar_tops.append(nested_tops[i])
        continue
    # Run mixed-effects model
    model = Lmer("bl_avg ~ group_label + (1|id_1) + (1|id_2)", data=comp_df)
    result = model.fit()
    # Find the row for the group_label predictor (not Intercept)
    pval = result.loc["group_labelref", "P-val"]
    p_values.append(pval)
    x_indices.append(nested_xs[i])
    bar_tops.append(nested_tops[i])

# Position compact significance bar above all nested bars
ymin, ymax = ax.get_ylim()
y_range = ymax - ymin
y_pos = max(bar_tops) + 0.3 * y_range
bracket_height = 0.05 * y_range
add_compact_significance(ax, x_indices, y_pos, p_values, bracket_height, bar_tops, fontsize=22)

# ------------------- FINAL STYLING -------------------

ax.set_xticks([])
ax.set_xticklabels([])
# Draw horizontal reference dotted lines
ax.axhline(line1, color="black", linestyle=":", linewidth=2, zorder=0)
ax.axhline(line2, color="black", linestyle=":", linewidth=2, zorder=0)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_ylabel("Mean distance\n(body lengths)", fontsize=22)
ax.tick_params(axis='y', labelsize=20)
ax.set_xlim(-0.5, 4.5)
sns.despine()

plt.tight_layout()
plt.show()
