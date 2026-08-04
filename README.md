# DivColl

Code and data for:

> Hein A, Bongard J. **Environmental resilience via morphological diversity
> within machines.**

Everything needed to reproduce the figures, tables and analyses in the paper
is included here, along with the trained agents and connectors used in all
experiments.

## Contents

| Path | Contents |
| :-- | :-- |
| `outputs/og_bots/` | The 100 trained agents used throughout the paper |
| `outputs/` | Trained connectors and all other evolved artifacts |
| `figures/csvs/` | Result data underlying Figs 2-4 and the hidden-state table |
| `figures/*.py` | Scripts regenerating each figure from those CSVs |
| `terrains/` | Definitions of the 12 deployment environments |
| `patches/` | Modifications to the ELDiR simulator (see below) |

## Quick start

```sh
pip install -r requirements.txt
./setup_simulator.sh          # fetches and patches the simulator; see below
```

**To regenerate the figures you do not need the simulator at all** - the
scripts in `figures/` read `figures/csvs/` directly:

```sh
python figures/fig_2.py
python figures/fig_3.py
python figures/fig_4_left.py
python figures/fig_4_right.py
```

## Provenance of the simulator

The differentiable physics simulator and the base evolutionary operators used
in this work derive from ELDiR:

> Strgar L, Matthews D, Hummer T, Kriegman S. **Evolution and learning in
> differentiable robots.** Proceedings of Robotics: Science and Systems, 2024.
> <https://doi.org/10.15607/RSS.2024.XX.100> - <https://github.com/lstrgar/ELDiR>

Seven files descend from that project and are **not redistributed here**.
`patches/eldir-to-divcoll.patch` contains only our modifications to them,
expressed as a diff against ELDiR commit `bcd88b08`. Running
`./setup_simulator.sh` clones ELDiR at that commit, applies the patch, and
installs the result. See `patches/README.md` for the manual equivalent.

## License

The code and data in this repository are released under the MIT License (see
`LICENSE`). That license does not extend to the ELDiR-derived files described
above, which remain subject to their authors' terms.

## Terminology

The paper refers to individual robots as **agents**. Parts of the code use the
earlier term **components** (for example `train_components.py`). The two mean
the same thing.

---

## Script reference

### Training Individual Components

**Script:** `train_components.py`
Trains agent bodies on flat ground by default, using one GPU (**tested on Metal; CUDA should probably also work, but not tested**).

**Arguments:**


| Argument | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `--groundfile` | string | None | Path to custom ground file |
| `--pop_size` | int | 100 | Number of individuals in population |
| `--n_gens` | int | 15 | Number of generations to train |
| `--gpu` | flag | False | Use GPU for training |

- Configure agent bodies in `operators/defaults/config.py`.
- Agents are saved to `outputs/bots`. Original agents from the paper: `outputs/og_bots`.
- Training logs saved in `outputs/gens`.
- Example:

```
python train_components.py --gpu
```


***

### Visualizing Agents

**Script:** `show.py`
Displays static images of one or more agents.


| Argument | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `--from_bot` | int | 0 | First bot to show |
| `--to_bot` | int | 99 | Last bot to show |
| `--bot_dir` | string | outputs/bots | Directory with bot files |
| `--suffix` | string | "" | Suffix for collectives (optional) |

- Example to show all `outputs/og_bots`:

```
python show.py --from_bot 0 --to_bot 99 --bot_dir outputs/og_bots
```


***

### Evaluating Agents

**Script:** `evaluate.py`
Assess agents or collectives in different environments, visualize results, generate videos, or export CSV.


| Argument | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `--from_bot` | int | 0 | First bot to evaluate |
| `--to_bot` | int | 99 | Last bot to evaluate |
| `--env` | string | "in_dist" | Environment to evaluate on |
| `--steps` | int | 1000 | Number of steps to run |
| `--bot_dir` | string | outputs/bots | Directory with bot files |
| `--suffix` | string | "" | Suffix for collectives (optional) |
| `--coll_size` | int | 1 | Number of bots in the collective |
| `--conn_file` | string | None | Path to connector file for evaluation |
| `--bundle_size` | int | 100 | Number of collectives in a bundle for GPU batching |
| `--video` | flag | False | Create video (single agent only) |
| `--vid_dir` | string | outputs/vids | Directory for video files |
| `--show_plot` | flag | False | Show performance plot |
| `--to_csv` | flag | False | Save performance scores to CSV |

**Important Bundle Size Note:**
You must adjust `bundle_size` to match the number of agents you are evaluating. The `bundle_size` must be at most the number of agents you are evaluating (e.g., set `--bundle_size 1` for a single agent, `--bundle_size 40` for 40 agents, etc.). If you have e.g. 100 large collectives, you can use a smaller `bundle_size` (e.g. 50, 25, 10 etc). Taichi will usually not throw errors if the simulation does not fit on the GPU, but you might see nans or excessive loss values (especially for the first agent).

#### Available Environments

`in_dist`, `slope_down`, `slope_up`, `treadmill`, `wind`, `sticky`, `gap`, `sand`, `step_down`, `ice`, `step_up`, `obs`, `low_gravity`

- Example (all original agents for 3000 steps):

```
python evaluate.py --from_bot 0 --to_bot 99 --env in_dist --steps 3000 --bot_dir outputs/og_bots --show_plot --bundle_size 100
```

- For videos (single agent only):

```
python evaluate.py --from_bot 0 --to_bot 0 --video --bundle_size 1
```

#### Note on Video Generation

The code generates videos using `ffmpeg` via ImageIO. By default, it sets:

```python
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
```

This path works for Mac users with Homebrew.

- **If you're on Linux, Windows, or have `ffmpeg` elsewhere:**
    - Make sure `ffmpeg` is installed and available in your system path.
    - Adjust or remove this line if needed to point to your own `ffmpeg` binary.

***

### Creating Collectives

**Script:** `create_collective.py`
Group agents into uniform or diverse collectives.


| Argument | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `--from_bot` | int | 0 | First bot to include |
| `--to_bot` | int | 99 | Last bot to include |
| `--bot_dir` | string | outputs/bots | Directory with bot files |
| `--collective_dir` | string | outputs/collectives | Directory to save collectives |
| `--div` | flag | False | Create diverse collectives |
| `--size` | int | 2 | Size of the collective |
| `--conn_points` | string | None | File with connection points (optional) |

- Example:

```
python create_collective.py --from_bot 0 --to_bot 99 --size 2 --div
```

- Original collectives: `outputs/og_collectives`

***

#### Visualizing \& Evaluating Collectives

After creating collectives, you can use `show.py` and `evaluate.py` as with individual agents, but you must provide additional arguments:

- **show.py for collectives:**
Specify the suffix for the collective type (`_conn_div` for diverse, `_conn_uni` for uniform, `_not_conn_div` or `_not_conn_uni` for unconnected).
Example to visualize first 5 diverse collectives:

```
python show.py --from_bot 0 --to_bot 4 --bot_dir outputs/og_collectives --suffix _conn_div
```

- **evaluate.py for collectives:**
Use the collectives directory as `bot_dir`, set `coll_size` (number of bots in each collective), and specify the correct suffix.
Example to evaluate an *untrained* collectives of size 2:

```
python evaluate.py --from_bot 0 --to_bot 4 --bot_dir outputs/og_collectives --coll_size 2 --suffix _conn_div --bundle_size 5
```


***

### Training Connectors

**Script:** `train_connector.py`
Train connectors for collectives.


| Argument | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `--from_bot` | int | 0 | First collective to train on |
| `--to_bot` | int | 99 | Last collective to train on |
| `--coll_dir` | string | outputs/collectives | Directory with collectives |
| `--conn_dir` | string | outputs/connectors/div | Directory to save connectors |
| `--conn_id` | string | "1" | Connector name/ID |
| `--steps` | int | 1000 | Number of training steps |
| `--suffix` | string | "_conn_div" | Suffix for collective type |
| `--learning_iters` | int | 100 | Number of learning iterations |

- Example:

```
python train_connector.py --from_bot 0 --to_bot 99 --coll_dir outputs/og_collectives --conn_dir outputs/connectors/div --suffix _conn_div --conn_id 0
```

- Trained connectors used in the paper: `outputs/og_connectors`

**Notes:**
Connectors are specifically designed for the agent bodies in `outputs/og_bots`. If you create new bots (even if they have the same number of points and springs), training connectors may yield different results due to variations in the mix of body structures (e.g., upright or flat orientations).

- **Experimental tips:**
    - Try adjusting inter-bot distance (see placement logic in `create_collective.py`).
    - Change connector spring stiffness or activation values.
    - You can design new connector structures manually, or automate their evolution for your custom bots.
    - **Balance connector strength**: Too weak fails to coordinate pairs (both diverse and uniform); too strong can dominate agents, lead to unstable simulations, or "flip" uniform agents, essentially turning uniform pairs into diverse ones.
    - The optimal values and arrangement depend heavily on the specific agent and collective compositions.

***

#### Evaluating Collectives with Trained Connectors

After training connectors, use `evaluate.py` to test collectives with your chosen connector file. Specify all relevant arguments just as you do elsewhere:


| Argument | Type | Example Value | Description |
| :-- | :-- | :-- | :-- |
| `--from_bot` | int | 0 | First collective to evaluate |
| `--to_bot` | int | 4 | Last collective to evaluate |
| `--bot_dir` | string | outputs/og_collectives | Directory with collective files |
| `--conn_file` | string | outputs/og_connectors/div/5/best.pkl | Trained connector file |
| `--suffix` | string | _conn_div | Suffix for collective type (diverse, uniform, or unconnected) |
| `--env` | string | wind | Environment to evaluate on |
| `--steps` | int | 3000 | Number of steps to run |
| `--coll_size` | int | 2 | Number of bots in each collective |
| `--bundle_size` | int | 5 | Number of collectives per bundle |
| `--show_plot` | flag | True | Show performance plot |

- Be sure that `coll_size` matches the number of bots in each collective, and that `bundle_size` matches the number of collectives you’re evaluating.

**Example:**

```
python evaluate.py --from_bot 0 --to_bot 99 \
    --bot_dir outputs/og_collectives \
    --conn_file outputs/og_connectors/div/5/best.pkl \
    --suffix _conn_div \
    --env wind \
    --steps 3000 \
    --coll_size 2 \
    --bundle_size 100 \
    --show_plot
```

This command evaluates collectives 0–99 (each made up of two bots), with the specified trained connector in the "wind" environment for 3000 steps.

***

### Figures

Scripts and data for recreating figures are in the `figures` folder.

***

### Credits

Project based on [ELDiR](https://github.com/lstrgar/ELDiR).


