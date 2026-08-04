from argparse import ArgumentParser
import os
from simulator.sim import forward_visualization
# Set ffmpeg path for video encoding
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy.editor import * # For video creation from images
import numpy as np
from tqdm import tqdm # Progress bar for loops
from matplotlib import pyplot as plt
from io import BytesIO # In-memory byte buffer for images
import math
from utils.eval_utils import collect_bots # Utility for collecting bot files
import seaborn as sns
import pandas as pd

# =============================================================================
# LOAD REQUIRED SIMULATION & ENVIRONMENT FILES FROM DIRECTORY
# =============================================================================
def load_files(dir):
    # Positions of points over time excluding last point
    x = np.load(os.path.join(dir, "x.npy"))[0, :-1]
    spring_actuation = np.load(os.path.join(dir, "spring_actuation.npy"))[0]
    spring_anchor_a = np.load(os.path.join(dir, "spring_anchor_a.npy"))[0]
    spring_anchor_b = np.load(os.path.join(dir, "spring_anchor_b.npy"))[0]
    ground_x = np.load(os.path.join(dir, "ground_x.npy"))
    ground_y = np.load(os.path.join(dir, "ground_y.npy"))
    ground_len = np.load(os.path.join(dir, "ground_len.npy"))
    ground_slope = np.load(os.path.join(dir, "ground_slope.npy"))
    ground_shift = np.load(os.path.join(dir, "ground_shift.npy"))
    # Return all environment & simulation geometry data needed for plotting
    return x, spring_actuation, spring_anchor_a, spring_anchor_b, ground_x, ground_y, \
            ground_len, ground_slope, ground_shift, ground_len, ground_slope, ground_shift

# =============================================================================
# PLOT MASS POINTS ON CURRENT FRAME WITH LAYERED GREY HEX MARKERS
# =============================================================================
def plot_points(x, t):
    for i in range(x.shape[1]):
        grays = plt.colormaps['Greys']
        c0 = grays(0.971968)
        c1 = grays(0.45630044)
        c2 = grays(0.69349504)
        c3 = grays(0.7506251)
        # Draw multiple hexagonal markers with decreasing size and varying gray tones for depth effect
        plt.plot(x[t, i, 0], x[t, i, 1], 'H', color=c2, markersize=10)
        plt.plot(x[t, i, 0], x[t, i, 1], 'H', color=c3, markersize=8)
        plt.plot(x[t, i, 0], x[t, i, 1], 'H', color=c0, markersize=6)
        plt.plot(x[t, i, 0], x[t, i, 1], 'H', color=c1, markersize=4)

# =============================================================================
# SWITCH BACKGROUND IMAGE BASED ON ENVIRONMENT AND SIMULATION STATE
# =============================================================================
def switch_bg(img, img_path, counter, x_max, x_min, t):
    # Moving treadmill: switch background every quarter frame mod 5 for looping effect
    if "treadmill" in img_path:
        img_path = f'background_imgs/treadmill_{(math.floor(counter) % 5) + 1}.png'
        img = plt.imread(img_path)
        counter += 0.25
    # Wind effect: switch between two images depending on wind oscillation sign
    if "wind" in img_path:
        wind = 0.04 * math.sin(t * 0.02)
        if wind >= 0:
            img_path = f'background_imgs/wind_2.png'
        else:
            img_path = f'background_imgs/wind_1.png'
        og_img = plt.imread(img_path)
        num_tiles = int(np.ceil((x_max - x_min)))
        # Tile original image to span entire visible range horizontally
        img = np.tile(og_img, (1, num_tiles, 1))
    return img, counter

# =============================================================================
# EXTEND BACKGROUND IMAGE BY TILING (UNLESS COMPLEX TERRAIN SUCH AS SLOPE OR OBSTACLE)
# =============================================================================
def extend_bg(num_tiles, env, og_img):
    if num_tiles > 1 and "step" not in env and "slope" not in env and "obs" not in env:
        img = og_img[:, :-5]
        for tile in range(num_tiles):
            if "gap" in env:
                img_path = f'background_imgs/in_dist.png'
                og_img = plt.imread(img_path)
            if env in ["sand", "sticky"]:
                # Flip image horizontally for each tile to add variety
                og_img = np.flip(img, axis=1)
            # Concatenate tiles horizontally while overlapping edges slightly for seamlessness
            img = np.concatenate([img, og_img[:, 5:-5]], axis=1)
    else:
        img = og_img
    return img

# =============================================================================
# PREPARE BACKGROUND IMAGE AND TILE INFO BASED ON ENVIRONMENT
# =============================================================================
def get_bg(x_max, x_min, env, ground_y, x):
    num_tiles = int(np.ceil((x_max - x_min))) # Number of tiles to cover horizontal span
    # Adjust vertical positions for slope downward environments
    if env == "slope_down" and num_tiles > 1:
        x[:, :, 1] += 0.2
    if env == "slope_down" and num_tiles > 1:
        ground_y += 0.2
    # Use tiled background images for certain env categories
    if "step" in env or "slope" in env or "obs" in env:
        img_path = f'background_imgs/{env}_{num_tiles}.png'
        og_img = plt.imread(img_path)
        num_tiles = int(np.ceil((x_max - x_min)))
    else:
        img_path = f'background_imgs/{env}.png'
        og_img = plt.imread(img_path)

    img = extend_bg(num_tiles, env, og_img)
    return img, img_path, num_tiles

# =============================================================================
# PLOT SPRINGS BETWEEN POINTS WITH COLOR INDICATING ACTUATION
# =============================================================================
def plot_springs(spring_actuation, x, t, spring_anchor_a, spring_anchor_b):
    for j in range(spring_actuation.shape[0]):
        c = "dimgrey"
        if spring_actuation[j] == 0:
            c = "silver" # Springs inactive (no actuation) shown in silver
        # Draw line between two mass point positions connected by spring
        plt.plot([x[t, spring_anchor_a[j], 0], x[t, spring_anchor_b[j], 0]],
                 [x[t, spring_anchor_a[j], 1], x[t, spring_anchor_b[j], 1]], color=c, linewidth=2)

# =============================================================================
# COLLECT RENDERED FRAMES OF SIMULATION INTO A LIST
# =============================================================================
def collect_frames(steps, img, img_path, x_max, x_min, num_tiles, spring_actuation, x, spring_anchor_a, spring_anchor_b, counter):
    fig = plt.figure(figsize=(10, 5)) # Set figure size
    plt.gca().set_position([0, 0.0, 1, 1]) # Full figure usage for axes
    frames = []
    for t in tqdm(range(0, steps, 5)): # Iterate over timesteps with step of 5
        plt.clf() # Clear figure content
        # Update background image if environment requires animation or tiling
        img, counter = switch_bg(img, img_path, counter, x_max, x_min, t)
        # Show background image in plot region with specified extent
        plt.imshow(img, extent=[0.0, num_tiles, 0.0, 0.5], aspect='auto', zorder=0)
        # Draw springs
        plot_springs(spring_actuation, x, t, spring_anchor_a, spring_anchor_b) # Draw mass points
        plot_points(x, t)
        # Equal axes aspect ratio
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off') # Hide axes
        plt.xlim(x_min - 0.05, x_max + 0.05) # Set x limits slightly beyond extents
        buf = BytesIO() # Create in-memory buffer
        plt.tight_layout()
        plt.savefig(buf, format='raw', dpi=100) # Save current figure as raw RGBA data to buffer
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8) # Convert buffer to numpy array
        # Reshape to image height x width x channels for moviepy consumption
        img_arr = img_arr.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        frames.append(img_arr) # Append frame to list
    plt.close(fig)
    return frames, buf

# =============================================================================
# CREATE VIDEO FROM SIMULATION FRAMES AND SAVE TO FILE
# =============================================================================
def create_video(env, tmp_dir, bot, vid_dir):
    out_path = f"{vid_dir}/{bot}.mp4" # Output file path
    # Load required simulation data files from temporary simulation directory
    state_dir = os.path.join(tmp_dir, "state")
    x, spring_actuation, spring_anchor_a, spring_anchor_b, ground_x, ground_y, \
        ground_len, ground_slope, ground_shift, ground_len, ground_slope, ground_shift = load_files(state_dir)
    x_min, x_max = np.min(x[:, :, 0]), np.max(x[:, :, 0]) # Determine horizontal position range
    steps = x.shape[0] # Number of simulation timesteps
    # Get background image and related parameters
    img, img_path, num_tiles = get_bg(x_max, x_min, env, ground_y, x)
    counter = 0 # For background animation
    # Collect rendered frames from simulation
    frames, buf = collect_frames(steps, img, img_path, x_max, x_min, num_tiles, spring_actuation, x, spring_anchor_a, spring_anchor_b, counter)
    # Create video clip from frames at 50 FPS
    clip = ImageSequenceClip(frames, fps=50)
    clip.write_videofile(out_path) # Write video file to disk

    # Clean up to release memory
    del clip, buf

# =============================================================================
# SETUP SIMULATION FILES AND ENVIRONMENT SETTINGS BASED ON ENVIRONMENT NAME
# =============================================================================
def setup_sim(condition):
    if condition == "in_dist":
        g_file = None
        env_setting = None
    elif condition in ["slope_down", "obs", "slope_up", "gap", "step_down", "step_up"]:
        g_file = f"terrains/{condition}.npy" # Load terrain file for environment
        env_setting = None
    elif condition in ["ice", "sticky", "wind", "sand", "low_gravity", "treadmill"]:
        g_file = None
        env_setting = condition # Special environment settings coded separately
    return g_file, env_setting

# =============================================================================
# SAVE PERFORMANCE METRICS TO CSV FILE
# =============================================================================
def save_to_csv():
    if conn_file is not None:
        save_to = conn_file.replace("best.pkl", "performance.csv")
    else:
        save_to = f"{bot_dir}/performance.csv"
    if os.path.exists(save_to):
        df = pd.read_csv(save_to)
    else:
        df = pd.DataFrame()
    for r in range(bundle_size):
        # Construct dictionary for new row with summary and per-bot metrics
        new_row_dict = {"env": [environment], "avg": [loss[r]], 'min': [loss_per_bot[r].min()], 'max': [loss_per_bot[r].max()],
              "std": [loss_per_bot[r].std()], "setting": [suffix]}
        for sb in range(coll_size):
            new_row_dict[f"perf_{sb+1}"] = [loss_per_bot[r][sb]]
            new_row_dict[f"id_{sb + 1}"] = [ids[r][sb]]
        # Add average normalized performance over sub-robots
        new_row = pd.DataFrame(new_row_dict)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(f"{save_to}", index=False, float_format='%.3f')

# =============================================================================
# MAIN FUNCTION TO RUN EVALUATION AND VISUALIZATION
# =============================================================================
def main():
    print("running evaluation")

# =============================================================================
# COMMAND LINE INTERFACE AND EXECUTION LOGIC
# =============================================================================
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--from_bot', type=int, default=0, help='First bot to evaluate')
    parser.add_argument('--to_bot', type=int, default=99, help='Last bot to evaluate')
    parser.add_argument('--env', type=str, default="in_dist", help='Environment to evaluate on')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to run')
    parser.add_argument('--bot_dir', type=str, default="outputs/bots", help='Directory with bot files')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for collectives (_conn_div, _conn_uni, _not_conn_uni, _not_conn_div)')
    parser.add_argument('--coll_size', type=int, default=1, help='Number of bots in the collective')
    parser.add_argument('--conn_file', type=str, default=None, help='File with connector to evaluate')
    parser.add_argument('--bundle_size', type=int, default=100, help='Number of collectives in a bundle')
    parser.add_argument('--video', action='store_true', help='Create a video')
    parser.add_argument('--vid_dir', type=str, default="outputs/vids", help='Where to store video files')
    parser.add_argument('--show_plot', action='store_true', help='Show plot with bot performance')
    parser.add_argument('--to_csv', action='store_true', help='Save performance as a csv file')
    args = parser.parse_args()

    # Assign command line arguments to variables
    environment = args.env
    n_steps = args.steps
    from_bot = args.from_bot
    to_bot = args.to_bot
    bot_dir = args.bot_dir
    vid_dir = args.vid_dir
    coll_size = args.coll_size
    save_csv = args.to_csv
    conn_file = args.conn_file
    bundle_size = args.bundle_size
    os.makedirs(vid_dir, exist_ok=True)
    show_plot = args.show_plot
    suffix = args.suffix

    # Validation for configuration sanity
    assert not (coll_size > 1 and suffix == ""), "please specify suffix for collectives with more than one bot"
    create_vid = args.video
    state_dir = os.path.join(f"tmp", "state")
    bundles = (to_bot + 1 - from_bot) // bundle_size

    # Collect bot files into bundles for evaluation
    collect_bots(from_bot, bot_dir, suffix, coll_size, bundles, bundle_size)

    # Setup simulation based on environment choice
    ground_file, setting = setup_sim(environment)

    # Run evaluation for each bundle
    for b in range(bundles):
        forward_visualization(f"{bot_dir}/bots_{b}.pkl", "tmp", ground_file, setting, n_steps, coll_size, conn_file)

        # Load losses and positions from simulation state files
        loss = -np.load(os.path.join(state_dir, "loss.npy"))
        loss_per_bot = np.load(os.path.join(state_dir, "loss_per_bot.npy"))
        ids = np.load(f"{bot_dir}/bots_{b}.pkl", allow_pickle=True)["ids"]
        x = np.load(os.path.join(state_dir, "x.npy"))[:, :-1]
        ppb = x.shape[2] // coll_size # Points per bot
        print(loss, loss.mean(), loss.std())
        if save_csv: # Save results to csv if flagged
            save_to_csv()
        # Visualize performance as a box and strip plot using seaborn
        if show_plot:
            ax = sns.boxplot(loss, fliersize=0, color="white")
            sns.stripplot(loss, color="dimgrey", alpha=0.6, jitter=True, ax=ax)
            ax.set_ylabel("Performance")
            plt.show()
            plt.close()
        # Generate visualization video for single bot selection only
        if create_vid:
            assert (to_bot + 1) - to_bot == 1, "video visualization requires choosing only one bot"
            create_video(environment, "tmp", from_bot, vid_dir)