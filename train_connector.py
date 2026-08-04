from argparse import ArgumentParser
from simulator.sim import simulate_conn
from utils.eval_utils import collect_bots
import os
import numpy as np

if __name__ == '__main__':
    # =============================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # =============================================================================
    parser = ArgumentParser()
    parser.add_argument('--from_bot', type=int, default=0, help='First collective to include')
    parser.add_argument('--to_bot', type=int, default=99, help='Last collective to include')
    parser.add_argument('--coll_dir', type=str, default="outputs/collectives", help='Folder with collectives to train on')
    parser.add_argument('--conn_dir', type=str, default="outputs/connectors/div", help='Folder for saving trained connectors')
    parser.add_argument('--conn_id', type=str, default="1", help='Name to save connector under')
    parser.add_argument('--steps', type=str, default=1000, help='Number of steps to train on')
    parser.add_argument('--suffix', type=str, default="_conn_div", help='Suffix for collectives (_conn_div, _conn_uni, _not_conn_uni, _not_conn_div)')
    parser.add_argument('--learning_iters', type=int, default=100, help='Number of learning iterations')
    args = parser.parse_args()

    # =============================================================================
    # ASSIGN ARGUMENTS TO VARIABLES
    # =============================================================================
    coll_dir = args.coll_dir
    conn_dir = args.conn_dir
    steps = args.steps
    suffix = args.suffix
    conn_id = args.conn_id
    learning_iters = args.learning_iters
    from_bot = args.from_bot
    to_bot = args.to_bot

    # Ensure connector directory exists
    os.makedirs(conn_dir, exist_ok=True)
    # =============================================================================
    # COLLECT ALL SELECTED COLLECTIVES INTO A SINGLE FILE FOR TRAINING
    # =============================================================================
    # This bundles the collectives into "bots_0.pkl", treating each pair as a separate robot with two sub-bots
    collect_bots(
        from_bot,
        coll_dir,
        suffix,
        2,  # Each collective consists of 2 bots
        1,
        (to_bot + 1 - from_bot)
    )
    # =============================================================================
    # TRAIN CONNECTORS TO MEDIATE BETWEEN THE BOT PAIRS
    # =============================================================================
    simulate_conn(
        f"{coll_dir}/bots_0.pkl",  # Combined collectives file
        conn_dir,  # Directory to save trained connectors in
        conn_id,  # Connector identifier name
        steps,  # Number of simulation steps
        2,  # Collective size (pairs of bots)
        learning_iters,
        np.random.randint(1000)  # Random seed
    )

