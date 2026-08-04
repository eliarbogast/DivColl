import pickle, os
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np

if __name__ == '__main__':
    # =============================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # =============================================================================
    parser = ArgumentParser()
    parser.add_argument('--from_bot', type=int, default=0, help='First bot to show')
    parser.add_argument('--to_bot', type=int, default=99, help='Last bot to show')
    parser.add_argument('--bot_dir', type=str, default="outputs/bots", help='Directory with bot files')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for collectives (_conn_div, _conn_uni, or _not_conn)')
    args = parser.parse_args()

    first_bot = args.from_bot
    last_bot = args.to_bot
    suffix = args.suffix
    directory_path = args.bot_dir  # Directory where bot pickle files are stored

    # Ensure the bot range is valid
    assert first_bot <= last_bot, "'from' needs to be larger than 'to'"

    # Number of bots to visualize
    n_bots = (last_bot+1) - first_bot

    # Calculate grid size (rows and columns) to display bots evenly in a square-ish grid
    n_by_n = np.ceil(np.sqrt(n_bots)).astype(int)

    # Create matplotlib figure and subplots grid
    fig, ax = plt.subplots(nrows=n_by_n, ncols=n_by_n, squeeze=False)
    ax = ax.flatten() # Flatten 2D axis array into 1D for easy indexing

    # =============================================================================
    # LOAD EACH ROBOT AND PLOT ITS SPRINGS AND MASS POINTS
    # =============================================================================
    for i in range(first_bot, last_bot + 1):
        full_path = os.path.join(directory_path, f"{i}{suffix}.pkl") # Path to robot file
        if os.path.isfile(full_path) and full_path.endswith(".pkl"):
            with open(full_path, 'rb') as pickle_file:
                robot = pickle.load(pickle_file)
            robot_points = robot['points'][0] # List of mass point coordinates
            robot_springs = robot['springs'][0] # List of springs parameters [anchor_a_index, anchor_b_index, length, stiffness, actuation]
            # Plot springs (lines) connecting mass points
            for j in range(len(robot_springs)):
                # Color depends on whether spring is actuated or passive
                if robot_springs[j][-1] != 0:
                    c = "dimgrey" # Actuated spring
                else:
                    c = "silver" # Passive spring
                ax[i-first_bot].plot(
                    [robot_points[robot_springs[j][0]][0], robot_points[robot_springs[j][1]][0]],
                    [robot_points[robot_springs[j][0]][1], robot_points[robot_springs[j][1]][1]],
                    color=c)
                # Plot mass points as small circles
                for j in range(len(robot_points)):
                    ax[i-first_bot].plot(robot_points[j][0], robot_points[j][1], 'o', color="dimgrey", markersize=1)
                    # Set equal aspect ratio so robot shape is not distorted
                    ax[i-first_bot].set_aspect('equal', adjustable='box')
    # =============================================================================
    # TURN OFF AXES FOR UNUSED SUBPLOTS TO CLEAN UP THE FIGURE
    # =============================================================================
    for i in range(n_by_n**2):
        ax[i].axis('off')
    # Show the figure window with all plotted robots
    fig.show()
