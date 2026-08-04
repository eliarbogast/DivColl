import os, sys, time, shutil, subprocess, numpy as np
from argparse import ArgumentParser
from operators.defaults.geno_pheno import random_geno, geno_2_pheno
from operators.defaults.mutate import mutate
from operators.defaults.select import select
from utils.disk_utils import save_fit, load_pop, save_pop
from simulator.utils import simulate_pop
import pickle

def save_current_bots(generation):
    """
        Load the population robots for the given generation,
        and save each individual robot with its respective weights
        as a separate pickle file in the bots directory.
        """
    with open(os.path.join(output_dir, str(generation), "robots.pkl"), "rb") as f:
        robots = pickle.load(f)
    for i in range(len(robots["points"])):
        best_robot_points = robots['points'][i]
        best_robot_springs = robots['springs'][i]
        best_robot_body_geno = robots['body_geno'][i]
        best_robot_spring_geno = robots['spring_geno'][i]
        best_robot_id = robots['id'][i]
        best_robot = {
                "points": [best_robot_points],
                "springs": [best_robot_springs],
                "body_geno": [best_robot_body_geno],
                "spring_geno": [best_robot_spring_geno],
                "weights": []
            }
        # Extract generation and robot index from ID string like "gen-idx"
        train_gen, idx = best_robot_id.split("-")
        weight_path = os.path.join(output_dir, train_gen, "weights", idx, "best.pkl")
        with open(weight_path, "rb") as f:
            weights = pickle.load(f)
            best_robot["weights"].append(weights)
            robot_save_file = os.path.join(f"{bot_dir}/{i}.pkl")
            with open(robot_save_file, "wb") as f:
                pickle.dump(best_robot, f)

if __name__ == '__main__':

    # Enable or disable Taichi debug mode
    debug = False

    # Define output directories for generations and bot files
    output_dir = os.path.abspath("./outputs/gens")
    bot_dir = os.path.abspath("./outputs/bots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bot_dir, exist_ok=True)

    # Redirect stdout and stderr to log files for capturing output
    log_file = os.path.join(output_dir, f"stdout_{time.strftime('%Y%m%d-%H%M%S')}.txt")
    err_file = os.path.join(output_dir, f"stderr_{time.strftime('%Y%m%d-%H%M%S')}.txt")
    sys.stdout = open(log_file, "w")
    sys.stderr = open(err_file, "w")

    # Parse command line arguments for configuration
    parser = ArgumentParser()
    parser.add_argument('--groundfile', type=str, default=None, help='Path to custom ground file')
    parser.add_argument('--pop_size', type=int, default=100, help='Number of individuals in population')
    parser.add_argument('--n_gens', type=int, default=15, help='Number of individuals in population')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    ground_file = args.groundfile
    pop_size = args.pop_size
    n_gens = args.n_gens

    # Configure GPU usage
    use_cuda = args.gpu
    device_ids = [0,]
    if not use_cuda:
        device_ids = None

    # Create folder for initial generation (gen 0)
    gen_dir = os.path.join(output_dir, "0")
    os.makedirs(gen_dir, exist_ok=True)

    # Simulate initial population and get fitness values
    pop_fpath = random_geno(pop_size, gen_dir)
    geno_2_pheno(pop_fpath)

    ## Evaluate the initial population and save fitness trajectory
    pop_fit, pop_fit_fpath = simulate_pop(pop_fpath, gen_dir, device_ids, ground_file, debug)

    # Prepare generation 1 directory
    gen_dir = os.path.join(output_dir, "1")
    os.makedirs(gen_dir, exist_ok=True)
    # Sort population by max fitness descending and reorder population accordingly
    pop_order = pop_fit.max(1).argsort()[::-1]
    pop_fit = pop_fit[pop_order, :]
    # Save reordered fitness array for gen 1
    np.save(os.path.join(output_dir, "1", os.path.basename(pop_fit_fpath)), pop_fit)
    # Copy population genome file for gen 0 to gen 1 to serve as parent genomes
    shutil.copy(pop_fpath, pop_fpath.replace("0", "1"))
    pop_fpath = pop_fpath.replace("0", "1")
    pop_fit_fpath = pop_fit_fpath.replace("0", "1")
    # Load population genomes and reorder individuals by fitness rank
    pop = load_pop(pop_fpath)
    for k, v in pop.items():
        pop[k] = [v[i] for i in pop_order]
    save_pop(pop, pop_fpath)

    # Main evolutionary loop over generations 1 to n_gens - 1
    for gen in range(1, n_gens):
        print(f"Generation {gen}")

        # Create offspring by mutating current population genomes
        offspring_fpath = mutate(pop_fpath, gen, pop_fit)
        geno_2_pheno(offspring_fpath) # Convert offspring genomes to phenotypes

        # Simulate offspring population to evaluate fitness
        offspring_fit, offspring_fit_fpath = simulate_pop(offspring_fpath, gen_dir, device_ids, ground_file, debug)

        # Create directory for next generation (gen+1)
        gen_dir = os.path.join(output_dir, str(gen+1))
        os.makedirs(gen_dir, exist_ok=True)

        # Select individuals for next generation by combining parents and offspring fitness
        pop_fpath, pop_fit = select(pop_fpath, pop_fit, offspring_fpath, offspring_fit, gen_dir)
        # Save fitness values of the selected population
        save_fit(pop_fit, gen_dir, os.path.basename(pop_fit_fpath))
        # Generate phenotypes from selected genomes for next generation
        geno_2_pheno(pop_fpath)
        # Save robots of current generation as separate pickle files with weights for visualization or analysis
        save_current_bots(gen)