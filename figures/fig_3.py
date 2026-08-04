import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

og_df = pd.read_csv("csvs/performance.csv")
og_df["morph"] = "diff_shape_diff_spr"
# Load body length data
lengths = pd.read_csv('csvs/body_lengths.csv')
# Merge to get body length for id_1
og_df = og_df.merge(lengths.rename(columns={'id': 'id_1', 'length': 'length_1'}), on='id_1', how='left')
# Merge to get body length for id_2
og_df = og_df.merge(lengths.rename(columns={'id': 'id_2', 'length': 'length_2'}), on='id_2', how='left')
# Compute normalized performance
og_df['bl_perf_1'] = og_df['perf_1'] / og_df['length_1']
og_df['bl_perf_2'] = og_df['perf_2'] / og_df['length_2']
# Compute the average
og_df['bl_avg'] = og_df[['bl_perf_1', 'bl_perf_2']].mean(axis=1)
og_df = og_df[og_df["trained_on"] == "div"]

df_new = pd.read_csv("csvs/morph_var_performance.csv")
df_new["length_1"] = 0.15
df_new["length_2"] = 0.15
df_new['bl_perf_1'] = df_new['perf_1'] / df_new["length_1"]
df_new['bl_perf_2'] = df_new['perf_2'] / df_new["length_2"]
# Compute the average
df_new['bl_avg'] = df_new[['bl_perf_1', 'bl_perf_2']].mean(axis=1)
df = pd.concat([og_df, df_new], ignore_index=True)

df["ood"] = df["env"] != "in_dist"
df["group"] = df["morph"] + df["setting"]

fig = plt.figure(figsize=(14, 6))
plt.subplots_adjust(bottom=0.23, left=0.25, top=0.75)
subfigs = fig.subfigures(nrows=1, ncols=2, wspace=-0.29)
subfig_row1 = subfigs[0]
axes_row1 = subfig_row1.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 2]})
sns.barplot(x="group", y="bl_avg", color="white", edgecolor="black",
            order=["diff_shape_diff_spr_not_conn_uni", "same_shape_diff_spr_not_conn_uni",
                   "same_shape_same_spr_not_conn_uni"],
            data=df[df["ood"] == True],
            ax=axes_row1.flat[0])
for p_idx, patch in enumerate(axes_row1.flat[0].patches):
    patch.set_linewidth(2)
    patch.set_linestyle(':')

sns.barplot(x="group", y="bl_avg", color="#0072B2", edgecolor="black", order=["diff_shape_diff_spr_conn_div",
                                                                              "same_shape_diff_spr_conn_div",
                                                                              "diff_shape_diff_spr_conn_uni",
                                                                              "same_shape_diff_spr_conn_uni",
                                                                              "same_shape_same_spr_conn_div",
                                                                              "same_shape_same_spr_conn_uni"],
            data=df[(df["ood"] == True)],
            ax=axes_row1.flat[1])

for p_idx, patch in enumerate(axes_row1.flat[1].patches):
    patch.set_linewidth(2)
axes_row1.flat[1].set_ylabel("")
axes_row1.flat[1].set_title("Diverse pairs with\nconnectors trained\non diverse pairs", fontsize=18, pad=15)
axes_row1.flat[0].set_title("Unpaired\n(no connectors)", fontsize=18, pad=15)
axes_row1.flat[1].set_xlabel("")
axes_row1.flat[0].set_xlabel("")
axes_row1.flat[0].set_ylabel("Mean distance\n(body lengths)", fontsize=15, labelpad=15)
axes_row1.flat[1].set_xticks([])
axes_row1.flat[0].set_xticks([])
axes_row1.flat[0].tick_params(axis='both', which='major', labelsize=14)
axes_row1.flat[1].tick_params(axis='both', which='major', labelsize=14)
axes_row1.flat[0].set_ylim(axes_row1.flat[1].get_ylim())

subfig_row2 = subfigs[1]
axes_row2 = subfig_row2.subplots()
axes_row2.axis("off")

data1 = [["x", "x", "x"],
         ["x", "x", ""],
         ["x", "", ""],
         ["", "", ""]]
row_labels = ['Diverse behavior', 'Diverse springs', 'Diverse shape', "Diverse pairing"]
table1 = axes_row1.flat[0].table(cellText=data1,
                                 rowLabels=row_labels,
                                 cellLoc='center',
                                 bbox=[0, -0.4, 1, 0.35])
table1.set_fontsize(15)
data2 = [["x", "x", "x", "x", "x", "x"],
         ["x", "x", "x", "x", "", ""],
         ["x", "", "x", "", "", ""],
         ["x", "x", "", "", "x", ""]]
table2 = axes_row1.flat[1].table(cellText=data2,
                                 cellLoc='center',
                                 bbox=[0, -0.4, 1, 0.35])

table2.set_fontsize(15)

data3 = [["", "", ""],
         ["x", "x", ""],
         ["x", "", ""],
         ["", "", ""]]
table3 = axes_row2.table(cellText=data3,
                         cellLoc='center',
                         bbox=[0, -0.4, 1, 0.35])
table3.set_fontsize(15)
for key, cell in table1.get_celld().items():
    if key[1] == -1:
        cell.visible_edges = ''
for key, cell in table3.get_celld().items():
    if key[0] == 0 or key[0] == 3:
        cell.visible_edges = ''
#plt.show()
plt.savefig("fig_3.svg")
