This folder contains scripts for recreating the figures shown in the main paper. Each script uses specific data files and, where applicable, external Python packages. Below are instructions and details for reproducibility.

***

### Figure 2

- **To recreate:** Run `fig_2.py`
- **Data used:**
    - `csvs/performance.csv` – Contains performance metrics of `outputs/og_collectives` with `outputs/og_connectors`
    - `csvs/performance_tight.csv` – Contains performance metrics of `outputs/tight_collectives` with `outputs/tight_connectors`
    - `csvs/body_lengths.csv` – Based on `outputs/og_bots`
- **Notes:**
    - The tightly connected collectives use the same bots as `og_collectives`, but are arranged more closely (offset by 0.02 instead of 0.3).
    - Connector springs for tight collectives have a stiffness value of 5000 versus 2000 in `og_collectives`.
    - Significance calculations require the installation of the [`pymer4`](https://pypi.org/project/pymer4/) package (version 0.8.2)

***

### Figure 3

- **To recreate:** Run `fig_3.py`
- **Data used:**
    - `csvs/morph_var_performance.csv` – Contains performances of two groups:
        - `outputs/same_shape_collectives` (made of `outputs/same_shape_bots`)
        - `outputs/same_springs_collectives` (made of `outputs/same_springs_bots`)
- **Notes:**
    - `same_shape_bots`: All have the same shape but different arrangements of active/passive springs.
    - `same_springs_bots`: All have the same shape and spring distribution, but different controller parameters.

***

### Figure 4 (Left)

- **To recreate:** Run `fig_4_left.py`
- **Data used:**
    - `csvs/dtw.csv` – Contains the trajectory differences (dynamic time warping distance) for each agent in connected vs. unconnected scenarios across environments.
- **How to calculate DTW:**
    - Used the [`similaritymeasures`](https://pypi.org/project/similaritymeasures/) Python package.
    - Input data: the x and y coordinates of each bot at each time step for 2999 time steps (last time step omitted as it is always empty).

***

### Figure 4 (Right)

- **To recreate:** Run `fig_4_right.py`
- **Data used:**
    - NumPy arrays in `pacmap_data/` – PaCMAP embeddings for hidden connector states in both the training and wind environments:
        - Diverse: `outputs/og_collectives` + `outputs/og_connectors/div/0/best.pkl`
        - Uniform: `outputs/og_collectives` + `outputs/og_connectors/uni/2/best.pkl`
- **To run other combinations or environments:**
    - Record connector hidden states (`conn_hidden` from the simulation), omitting the last time step.
- **PacMAP embeddings:**
    - Used implementation: [`PaCMAP`](https://github.com/YingfanWang/PaCMAP)
    - Default settings: `pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)`

***

### Larger Collectives (Supplementary Material)

- **To generate line plots:** Run `larger_colls.py`
- **Data used:**
    - Output folders: `outputs/larger_colls/5`, `outputs/larger_colls/10`, `outputs/larger_colls/20`
- **Note:**
    - When using `evaluate.py` with large collectives, consider reducing the `bundle_size` argument to fit more collectives into GPU memory, otherwise you might see nans or extreme loss values.
