import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

plt.rc('font', size=20)  # Sets the default font size to 20
plt.rc('axes', titlesize=24)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
plt.rc('legend', fontsize=18)  # fontsize of the legend

size_2 = pd.read_csv('csvs/larger_colls/scores_2.csv')
size_5 = pd.read_csv('csvs/larger_colls/scores_5.csv')
size_10 = pd.read_csv('csvs/larger_colls/scores_10.csv')
size_20 = pd.read_csv('csvs/larger_colls/scores_20.csv')
envs = size_2["env"].unique()


def get_ci(data):
    # Calculate standard deviation
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    # Calculate sample size
    n = len(data)
    # Calculate Standard Error of the Mean (SEM)
    sem = std_dev / np.sqrt(n)
    # For a 95% confidence interval, the t-critical value (assuming small sample)
    # Degrees of freedom = n - 1
    degrees_freedom = n - 1
    t_critical = stats.t.ppf(0.975, degrees_freedom)  # 0.975 for a two-tailed 95% CI
    # Calculate Margin of Error
    margin_of_error = t_critical * sem
    return margin_of_error


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(20, 15))
gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.1], hspace=0.8)
axes = []

titles = ["Decline", "Incline", "Treadmill", "Wind", "Sticky ground", "Gap", "Sand", "Step down", "Ice", "Step up",
          "Obstacle", "Low gravity"]
for i, env in enumerate(
        ["slope_down", "slope_up", "treadmill", "wind", "sticky", "gap", "sand", "step_down", "ice", "step_up", "obs",
         "low_gravity"]):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    axes.append(ax)
    filtered_df_2_mean = \
        size_2[(size_2["env"] == env) & (size_2["trained_on"] == "div") & (size_2["setting"] == "_conn_div")]["bl_avg"]
    filtered_df_2_mean_uni = \
        size_2[(size_2["env"] == env) & (size_2["trained_on"] == "uni") & (size_2["setting"] == "_conn_uni")]["bl_avg"]

    filtered_df_5_mean = \
    size_5[(size_5["env"] == env) & (size_5["trained_on"] == "div") & (size_5["setting"] == "_conn_div")]["bl_avg"]
    filtered_df_5_mean_uni = \
        size_5[(size_5["env"] == env) & (size_5["trained_on"] == "uni") & (size_5["setting"] == "_conn_uni")]["bl_avg"]

    filtered_df_10_mean = \
    size_10[(size_10["env"] == env) & (size_10["trained_on"] == "div") & (size_10["setting"] == "_conn_div")]["bl_avg"]
    filtered_df_10_mean_uni = \
        size_10[(size_10["env"] == env) & (size_10["trained_on"] == "uni") & (size_10["setting"] == "_conn_uni")]["bl_avg"]

    filtered_df_20_mean = \
        size_20[(size_20["env"] == env) & (size_20["trained_on"] == "div") & (size_20["setting"] == "_conn_div")]["bl_avg"]
    filtered_df_20_mean_uni = \
        size_20[(size_20["env"] == env) & (size_20["trained_on"] == "uni") & (size_20["setting"] == "_conn_uni")]["bl_avg"]

    ax.plot([2, 5, 10, 20], [filtered_df_2_mean.mean(), filtered_df_5_mean.mean(), filtered_df_10_mean.mean(),
                             filtered_df_20_mean.mean()], color='#0072b2', label="avg (connected collective)",
            linewidth=4)
    ax.plot([2, 5, 10, 20],
            [filtered_df_2_mean_uni.mean(), filtered_df_5_mean_uni.mean(), filtered_df_10_mean_uni.mean(),
             filtered_df_20_mean_uni.mean()], label="avg (connected collective) uni", color='#e69f00', linewidth=4,
            linestyle='--')

    ax.errorbar([2, 5, 10, 20], [filtered_df_2_mean.mean(), filtered_df_5_mean.mean(), filtered_df_10_mean.mean(),
                                 filtered_df_20_mean.mean()],
                yerr=[get_ci(filtered_df_2_mean), get_ci(filtered_df_5_mean), get_ci(filtered_df_10_mean),
                      get_ci(filtered_df_20_mean)], color='#0072b2', fmt="o")
    ax.errorbar([2, 5, 10, 20],
                [filtered_df_2_mean_uni.mean(), filtered_df_5_mean_uni.mean(), filtered_df_10_mean_uni.mean(),
                 filtered_df_20_mean_uni.mean()],
                yerr=[get_ci(filtered_df_2_mean_uni), get_ci(filtered_df_5_mean_uni), get_ci(filtered_df_10_mean_uni),
                      get_ci(filtered_df_20_mean_uni)], color='#e69f00', fmt="o", linewidth=4)
    ax.set_title(titles[i], fontsize=26, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=20)
    if i in [0, 4, 8, 12, 16]:
        ax.set_ylabel("Mean distance\n(body lengths)", fontsize=25)
    ax.set_xlabel("Collective size", fontsize=25)

legend_ax = fig.add_subplot(gs[3, :])
legend_ax.axis('off')  # hide the axis

handles = [Line2D([0], [0], color='#0072b2', lw=10), Line2D([0], [0], color='#e69f00', lw=10, linestyle=(0, (2, 1))), ]
labels = ["Connectors trained on diverse pairs,\ntested on diverse collectives ≥ 2",
          "Connectors trained on uniform pairs,\ntested on uniform collectives ≥ 2"]
legend_ax.legend(handles, labels, loc='center', ncol=2, frameon=False, fontsize=23)
plt.tight_layout()
plt.show()
