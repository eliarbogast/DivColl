import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

sns.set(font_scale=1.5)
sns.set_style('white')
plt.rcParams['font.family'] = 'DejaVu Sans'
df = pd.read_csv("csvs/dtw.csv")
group_true_uni = df[(df["best_perf"] == True) & (df["conn_type"] == "conn_uni")]["DTW"]
group_false_uni = df[(df["best_perf"] == False) & (df["conn_type"] == "conn_uni")]["DTW"]

df_mean = df.groupby(["coll_id", "conn_id", "env", "conn_type"])["DTW"].agg("sum").reset_index()
df_mean["best_perf"] = df_mean["conn_type"]
fig, ax = plt.subplots(figsize=(7.2, 8.2))
ax = sns.barplot(x="best_perf", y="DTW", width=0.85, data=df_mean[df_mean["conn_type"] == "conn_div"], ax=ax,
                 color="#0072b2", edgecolor="black")
ax = sns.barplot(x="best_perf", y="DTW", width=0.85, data=df_mean[df_mean["conn_type"] == "conn_uni"], ax=ax,
                 color="#e69f00", edgecolor="black")

ax = sns.barplot(x="conn_type", y="DTW", width=0.3, data=df[(df["best_perf"] == False) & (df["conn_type"] == "conn_div")], ax=ax,
                 color="#0072b2")
ax = sns.barplot(x="conn_type", y="DTW", width=0.3, data=df[(df["best_perf"] == True) & (df["conn_type"] == "conn_div")], ax=ax,
                 color="#0072b2")
ax = sns.barplot(x="conn_type", y="DTW", width=0.3, data=df[(df["best_perf"] == False) & (df["conn_type"] == "conn_uni")], ax=ax,
                 color="#e69f00")
ax = sns.barplot(x="conn_type", y="DTW", width=0.3, data=df[(df["best_perf"] == True) & (df["conn_type"] == "conn_uni")], ax=ax,
                 color="#e69f00")

for p_idx, patch in enumerate(ax.patches):
    patch.set_linewidth(2)
    if p_idx in [1, 4, 5]:
        patch.set_linestyle('--')

for e_idx, e_bar in enumerate(ax.lines):
    xy_dat = e_bar.get_xydata()
    if e_idx == 2:
        e_bar.set_xdata(-0.2)
    if e_idx == 3:
        e_bar.set_xdata(0.2)
    if e_idx == 4:
        e_bar.set_xdata(0.8)
    if e_idx == 5:
        e_bar.set_xdata(1.2)
for bar_idx, bar in enumerate(ax.patches):
    x = bar.get_x()
    if bar_idx == 2:
        bar.set_x(-0.35)
    if bar_idx == 3:
        bar.set_x(0.05)
    if bar_idx == 4:
        bar.set_x(0.65)
    if bar_idx == 5:
        bar.set_x(1.05)

ax.set_xticks([-0.2, 0.2, 0.8, 1.2],
              ["Worse-\nperforming\nagent", "Better-\nperforming\nagent", "Worse-\nperforming\nagent",
               "Better-\nperforming\nagent"], fontsize=13)
plt.ylabel("Average Dynamic Time Warping Distance")
plt.xlabel("")

# ============================ #
# ADD STATISTICAL ANNOTATION
# ============================ #

# Run t-test
group_true = df[(df["best_perf"] == True) & (df["conn_type"] == "conn_div")]["DTW"]
group_false = df[(df["best_perf"] == False) & (df["conn_type"] == "conn_div")]["DTW"]
group_true_uni = df[(df["best_perf"] == True) & (df["conn_type"] == "conn_uni")]["DTW"]
group_false_uni = df[(df["best_perf"] == False) & (df["conn_type"] == "conn_uni")]["DTW"]

u_stat, pval = mannwhitneyu(group_true, group_false, alternative='less')
_, pval_uni = mannwhitneyu(group_true_uni, group_false_uni, alternative='less')
_, pval_true_true = mannwhitneyu(group_true_uni, group_true, alternative='two-sided')
_, pval_false_false = mannwhitneyu(group_false_uni, group_false, alternative='two-sided')

# Convert p-value to stars
def pval_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# Coordinates for annotation
x1, x2 = -0.2, 0.2  # center positions of bars
y = max(df[(df["best_perf"] == False) & (df["conn_type"] == "conn_div")]["DTW"].mean(),
        df[(df["best_perf"] == True) & (df["conn_type"] == "conn_div")][
            "DTW"].mean()) * 1.05  # vertical position just above bars
h = max(df["DTW"]) * 0.01  # height of annotation line
# Draw the annotation line
ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.0, c='k')
ax.text((x1 + x2) / 2, y + h - 15, pval_to_star(pval),
        ha='center', va='bottom', color='k', fontsize=22)

x1, x2 = 0.8, 1.2  # center positions of bars
y = max(df[(df["best_perf"] == False) & (df["conn_type"] == "conn_uni")]["DTW"].mean(),
        df[(df["best_perf"] == True) & (df["conn_type"] == "conn_uni")][
            "DTW"].mean()) * 1.05  # vertical position just above bars
h = max(df["DTW"]) * 0.01  # height of annotation line
# Draw the annotation line
ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.0, c='k')
ax.text((x1 + x2) / 2, y + h - 15, pval_to_star(pval_uni),
        ha='center', va='bottom', color='k', fontsize=22)

x1, x2 = 0.2, 1.2  # center positions of bars
y = max(df[(df["best_perf"] == True) & (df["conn_type"] == "conn_uni")]["DTW"].mean(),
        df[(df["best_perf"] == True) & (df["conn_type"] == "conn_div")][
            "DTW"].mean()) * 1.33  # vertical position just above bars
h = max(df["DTW"]) * 0.01  # height of annotation line
# Draw the annotation line
ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.0, c='k')
ax.text((x1 + x2) / 2, y + h, pval_to_star(pval_true_true),
        ha='center', va='bottom', color='k', fontsize=20)

x1, x2 = -0.2, 0.8  # center positions of bars
y = max(df[(df["best_perf"] == False) & (df["conn_type"] == "conn_uni")]["DTW"].mean(),
        df[(df["best_perf"] == False) & (df["conn_type"] == "conn_div")][
            "DTW"].mean()) * 1.2  # vertical position just above bars
h = max(df["DTW"]) * 0.01  # height of annotation line
# Draw the annotation line
ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.0, c='k')
ax.text((x1 + x2) / 2, y + h, pval_to_star(pval_false_false),
        ha='center', va='bottom', color='k', fontsize=22)

titles = ["Diverse pairs\nwith connectors\ntrained on\ndiverse pairs",
          "Uniform pairs\nwith connectors\ntrained on\nuniform pairs"]
counter = -1
for bar in ax.patches:
    if np.isclose(bar.get_width(), 0.85):
        counter += 1
        x_center = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()  # Top of the bar
        y_bottom = 0  # Bar baseline
        # Add text above the bar, centered
        ax.text(x_center, y_top + 0.01 * max(df["DTW"]), titles[counter],
                ha='center', va='bottom', fontsize=15)

# Adjust layout to make space for the legend below
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
