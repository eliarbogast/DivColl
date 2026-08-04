import numpy as np
import os
from argparse import ArgumentParser
import pickle
import pandas as pd
from simulator.sim import forward_visualization
from utils.eval_utils import collect_bots

# =============================================================================
# PREDEFINED OFFSETS AND CONNECTION MAPPINGS FOR COLLECTIVE BOTS
# =============================================================================
# Dictionary mapping sub-robot indices to:
# - horizontal offset to position them side-by-side
# - list of indices of other bots they connect to for springs (connectors)
placing = {0: [0.3, [1, 2]],
               1: [0.0, [3, 5]], 2: [0.0, [4, 6]],
               3: [-0.3, [7, 11]], 4: [-0.3, [8, 12]], 5: [-0.3, [9, 13]], 6: [-0.3, [10, 14]],
               7: [-0.6, [15, 23]], 8: [-0.6, [16, 24]], 9: [-0.6, [17, 25]], 10: [-0.6, [18, 26]],
               11: [-0.6, [19, 27]], 12: [-0.6, [20, 28]], 13: [-0.6, [21, 29]], 14: [-0.6, [22, 30]],
               15: [-0.9, []], 16: [-0.9, []], 17: [-0.9, []], 18: [-0.9, []], 19: [-0.9, []],
               20: [-0.9, []], 21: [-0.9, []], 22: [-0.9, []], 23: [-0.9, []], 24: [-0.9, []]}

# =============================================================================
# GENERATE ALL CYCLIC ROTATIONS OF GROUPS OF ELEMENTS
# =============================================================================
# For diversity, shuffle elements and split into groups of size k.
# For each group, generate all cyclic rotations.
# Returns list of all such rotations, useful for forming robot collectives with varied arrangements.
def generate_cyclic_rotations(elements, k):
    np.random.shuffle(elements) # Randomize element order
    groups = np.array_split(elements, len(elements) // k) # Split into chunks of size k
    rotations = []
    for group in groups:
        group = group.tolist()
        for shift in range(k):
            rotation = group[shift:] + group[:shift] # Rotate group by shift
            rotations.append(rotation)
    return rotations

# =============================================================================
# COMBINE BOT IDS INTO GROUPS FOR COLLECTIVES
# =============================================================================
# If 'div' is True, creates diverse groups using cyclic rotations.
# Otherwise, creates uniform groups repeating the same bot id.
def combine_ids():
    if div:
        groups = generate_cyclic_rotations(list(range(from_bot, to_bot)), coll_size)
    else:
        groups = []
        for j in range(from_bot, to_bot):
            groups.append([j] * coll_size)
    return groups

# =============================================================================
# LOAD AND COMBINE INDIVIDUAL BOTS INTO A COLLECTIVE DATA STRUCTURE
# =============================================================================
# For each bot id in group_ids:
# - loads bot data from file if exists
# - appends weights, points, springs, ownership ids into collective dictionary
# Points are offset slightly to avoid perfect overlap.
def gather_collective(group_ids):
    coll = {"group": list(group_ids),"points": [[]],"springs": [[]],"s_id": [[]],"p_id": [[]],"weights": [[]],}
    for idx, g_id in enumerate(group_ids):
        filename = f"{g_id}.pkl"
        if os.path.isfile(os.path.join(bot_dir, filename)):
            bot = np.load(f'{bot_dir}/{filename}', allow_pickle=True)
            # Append weights for each sub-bot in collective
            coll["weights"][0].append(bot["weights"])
            start_point = len(coll["points"][0])
            # Append mass points offset slightly by 0.05 along x-axis
            for p in bot["points"][0]:
                coll["points"][0].append([p[0] + 1 * 0.05, p[1]])
                coll["p_id"][0].append(idx + 1) # mark ownership of points by robot idx + 1
            # Append springs with updated anchor indices (offset by start_point)
            for s in bot["springs"][0]:
                coll["springs"][0].append((s[0] + start_point, s[1] + start_point, s[2], s[3], s[4]))
                coll["s_id"][0].append(idx + 1) # mark ownership of springs
    return coll

# =============================================================================
# SAVE A COLLECTIVE TO DISK
# =============================================================================
# Saves collective dictionary to a pickle file, naming depends on whether diverse (div) or uniform
def save_collective(group_counter, c, collective):
    if div:
        robot_save_file = os.path.join(coll_dir,
                                       f"{group_counter}_{c}_div.pkl")
    else:
        robot_save_file = os.path.join(coll_dir,
                                       f"{group_counter}_{c}_uni.pkl")
    with open(robot_save_file, "wb") as f:
        pickle.dump(collective, f)

# =============================================================================
# APPLY HORIZONTAL OFFSET TO ALL POINTS IN COLLECTIVE FOR SPATIAL ARRANGEMENT
# =============================================================================
# Moves points of each sub-robot horizontally according to predefined offsets in 'placing'.
# Returns updated collective and number of points per bot.
def add_offset(collective):
    counter = -1
    points_per_bot = int(int(len(collective["points"][0])) // coll_size)
    for p_idx in range(int(len(collective["points"][0]))):
        if p_idx % points_per_bot == 0:
            counter += 1
        collective["points"][0][p_idx][0] += placing[counter][0]
    return collective, points_per_bot

# =============================================================================
# FIND CONNECTOR POINT INDICES BETWEEN TWO BOTS
# =============================================================================
# For a front_bot and child bot in a group, extracts points indices that will be used to create springs connecting them.
# Uses metadata DataFrame df that contains specific point indices for foot and top points.
def find_conn_points(front_bot, group, child):
    hind_bot = df[df["ID"] == group[child]]
    front_bot_left_foot = front_bot["left_foot"].values[0]
    front_bot_left_top = front_bot["left_top"].values[0]
    front_bot_right_top = front_bot["right_top"].values[0]
    hind_bot_right_foot = hind_bot["right_foot"].values[0]
    hind_bot_left_top = hind_bot["left_top"].values[0]
    hind_bot_right_top = hind_bot["right_top"].values[0]
    points_a = [front_bot_left_foot, front_bot_left_foot, front_bot_left_foot,
                front_bot_right_top,
                front_bot_left_top]
    points_b = [hind_bot_right_top, hind_bot_left_top, hind_bot_right_foot,
                hind_bot_right_foot,
                hind_bot_right_foot]
    return points_a, points_b

# =============================================================================
# ADD CONNECTOR SPRINGS BETWEEN BOTS IN A GROUP
# =============================================================================
# For each pair of bots that should be connected according to 'placing',
# computes spring properties (length, stiffness, actuation) and appends connector springs to collective.
def add_connectors(group, collective, points_per_bot):
    for b in range(coll_size - 1):
        for child in placing[b][1]:
            front_bot = df[df["ID"] == group[b]]
            if child < len(group):
                points_a, points_b = find_conn_points(front_bot, group, child)
                for s_i, s_j in zip(points_a, points_b):
                    first = np.array(collective["points"][0][s_i + points_per_bot * b])
                    second = np.array(collective["points"][0][s_j + points_per_bot * child])
                    dist = first - second
                    length = np.linalg.norm(dist)
                    # Append connector spring, negative s_id denotes connector
                    collective["springs"][0].append(
                        [s_i + points_per_bot * b, s_j + points_per_bot * child, length, 2000, 0.6])
                    collective["s_id"][0].append(-(b + 1)) # Negative id indicates connector
    return collective

# =============================================================================
# CREATE COLLECTIVES AND SAVE TO DISK
# =============================================================================
# Iterates over all collective groups, creates both disconnected and connected variants,
# and saves them to disk.
def create_collectives():
    for group_counter, group in enumerate(collective_ids):
        for c in ["not_conn", "conn"]:
            collective = gather_collective(group)
            collective, points_per_bot = add_offset(collective)
            if c == "conn" and coll_size > 1:
                collective = add_connectors(group, collective, points_per_bot)
            save_collective(group_counter, c, collective)

# =============================================================================
# PARTITION MASS POINTS OF A BOT AND COMPUTE SPATIAL METRICS
# =============================================================================
# For a bot index, computes:
# - left and right side average x distances from the center of mass
# - top side normalized y distances
# - center_dist: relative distance from center to each point of mass
# - standardized Euclidean distance to characterize point stability
def partition_bot(xs, bot_in_question):
    x = xs[bot_in_question]
    center = x.mean(1) # Central point of mass
    center_dist = x - center.reshape(-1, 1, 2) # Relative distance to center
    left_side = -center_dist.mean(0)[:, 0] # Points belonging to left side
    right_side = center_dist.mean(0)[:, 0] # Points belonging to right side
    top_side = center_dist.mean(0)[:, 1] # Points belonging to upper bot half
    top_side = (top_side - top_side.min()) / (top_side.max() - top_side.min())
    # Std dev of distances to find most stable points
    euclidean_dist_std = np.sqrt(center_dist[:, :, 0] ** 2 + center_dist[:, :, 1] ** 2).std(0)
    std_normalized = (euclidean_dist_std - euclidean_dist_std.min()) / (
            euclidean_dist_std.max() - euclidean_dist_std.min())
    return left_side, right_side, top_side, center_dist, std_normalized

# =============================================================================
# IDENTIFY LEFT AND RIGHT FOOT POINTS BASED ON TOUCH SENSORS AND SIDE METRICS
# =============================================================================
# Loads touch sensor data (averaged over time and normalized),
# then selects foot points by maximum product of touch intensity and side indicator.
def find_feet(bot_in_question, left_side, right_side):
    touch_sensor = np.load(os.path.join("tmp/state", "touch.npy"))[bot_in_question, :-1].mean(0)
    touch_sensor_normalized = (touch_sensor - touch_sensor.min()) / (touch_sensor.max() - touch_sensor.min())
    x_normalized = touch_sensor_normalized
    left_foot = (x_normalized * left_side).argmax()
    right_foot = (x_normalized * right_side).argmax()
    return left_foot, right_foot

# =============================================================================
# SAVE CONNECTION POINT DATA FOR A BOT TO CSV
# =============================================================================
# Appends a new row with connector anchor information for a bot and saves the CSV.
def save_conn_points(bot_in_question, left_foot, left_top, right_foot, right_top, conn_df):
    new_row = pd.DataFrame(
        {"ID": [bot_in_question], "left_foot": [left_foot],
         "right_foot": [right_foot], "left_top": [left_top],
         "right_top": [right_top]})
    conn_df = pd.concat([conn_df, new_row], ignore_index=True)
    conn_df.to_csv(f"{coll_dir}/conn_points.csv", index=False, float_format='%.3f')
    return conn_df

# =============================================================================
# IDENTIFY CONNECTOR ANCHORS FOR ALL BOTS BY ANALYZING SIMULATION DATA
# =============================================================================
# Runs forward simulation visualization to generate states,
# then computes spatial metrics and foot point indices for all bots,
# saving connector points into a DataFrame and CSV.
def identify_connector_anchors():
    # Collect bot files
    collect_bots(from_bot, bot_dir, "", 1, 1, to_bot-from_bot)
    # Run simulation on standard flat ground with individual bots for 1000 steps
    forward_visualization(f"{bot_dir}/bots_0.pkl", "tmp", None, None, 1000)
    # Load positions from sim
    xs = np.load(os.path.join("tmp/state", "x.npy"))[:, :-1]
    conn_df = pd.DataFrame(columns=["ID", "left_foot", "left_top", "right_foot", "right_top"])
    for bot_in_question in range(xs.shape[0]):
        left_side, right_side, top_side, center_dist, std_normalized = partition_bot(xs, bot_in_question)
        left_foot, right_foot = find_feet(bot_in_question, left_side, right_side)
        # Normalize and combine top side distance with stddev for top point identification
        x_normalized_no_touch = top_side * (1 - std_normalized)
        left_top = (x_normalized_no_touch * left_side).argmax()
        right_top = (x_normalized_no_touch * right_side).argmax()
        conn_df = save_conn_points(bot_in_question, left_foot, left_top, right_foot, right_top, conn_df)
    return conn_df

# =============================================================================
# MAIN ENTRY POINT: COMMAND-LINE ARGUMENTS, PREPARE COLLECTIVES
# =============================================================================
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--from_bot', type=int, default=0, help='First bot to include')
    parser.add_argument('--to_bot', type=int, default=99, help='Last bot to include')
    parser.add_argument('--bot_dir', type=str, default="outputs/bots", help='Directory with bot files')
    parser.add_argument('--collective_dir', type=str, default="outputs/collectives", help='Directory to save collectives to')
    parser.add_argument('--div', action='store_true', help='Create diverse collective(s)')
    parser.add_argument('--size', type=int, default=2, help='Size of the collective')
    parser.add_argument('--conn_points', type=str, default=None, help='Path to file with connection points')
    args = parser.parse_args()

    # Setup global parameters from args
    from_bot = args.from_bot
    to_bot = args.to_bot + 1 # inclusive
    assert from_bot <= to_bot, "first bot must be less than or equal to last bot"
    bot_dir = args.bot_dir
    div = args.div
    coll_size = args.size
    assert (to_bot - from_bot) % coll_size == 0, "number of included bots must be divisible by the size of the collective"
    coll_dir = args.collective_dir
    conn_points = args.conn_points
    os.makedirs(coll_dir, exist_ok=True)

    # Create groups of bots for collectives with or without diversity
    collective_ids = combine_ids()

    # Load existing connector points or compute if not available and collective size > 1
    if conn_points is not None and os.path.isfile(conn_points):
        df = pd.read_csv(conn_points)
    elif coll_size > 1:
        df = identify_connector_anchors()

    # Generate and save collective robots with and without connectors
    create_collectives()




