import numpy as np
from copy import deepcopy
from .geno_pheno import body_largest_cc, fill_holes, compute_descriptor, convert
from utils.disk_utils import load_pop, save_pop
from .config import TARGET_N_POM, TARGET_N_SPR


def mutate_geno(geno, check_nonzero=False, p=None, target_sum=None):
    """
        Mutate a binary genotype by flipping bits with probability p.
        Ensures at least one bit is set if check_nonzero is True.
        Optionally adjusts mutation to achieve target number of bits set (target_sum).
        """
    geno_cpy = deepcopy(geno)
    if p is None:
        p = 1 / len(geno_cpy) # Default: flip on average one bit
    geno_flip_mask = np.random.binomial(1, p, len(geno_cpy))
    mut_geno = np.logical_xor(geno_cpy, geno_flip_mask).astype(int)

    if check_nonzero:
        # Ensure mutated genotype is not empty
        while np.sum(mut_geno) == 0:
            geno_flip_mask = np.random.binomial(1, p, len(geno_cpy))
            mut_geno = np.logical_xor(geno_cpy, geno_flip_mask).astype(int)
    if target_sum is not None:
        current_sum = np.sum(mut_geno)
        difference = target_sum - current_sum
        if difference > 0:
            # Add 1s to reach target sum
            zeros = np.where(mut_geno == 0)[0]
            to_flip = np.random.choice(zeros, min(difference, len(zeros)), replace=False)
            mut_geno[to_flip] = 1
        elif difference < 0:
            # Remove 1s to reach target sum
            ones = np.where(mut_geno == 1)[0]
            to_flip = np.random.choice(ones, min(-difference, len(ones)), replace=False)
            mut_geno[to_flip] = 0
    return mut_geno

def mutate_body_geno(body_geno, spring_geno, check_nonzero, require_change, target_sum):
    """
        Mutate the body genotype with constraints.
        Ensures validity of the new genotype (correct number of points and springs),
        optionally requiring it to differ from the original.
        Attempts up to 1000 mutations to find a valid one, else returns original.
        """
    def valid_mutation(geno):
        points, springs = convert(geno, spring_geno)  # Uses dummy spring_geno to get points, springs
        return len(points) == target_sum and len(springs) == TARGET_N_SPR and (not require_change or not body_geno_unchanged(body_geno, geno))

    mut_body_geno = body_geno.copy()
    max_attempts = 1000
    for _ in range(max_attempts):
        temp_geno = mutate_geno(mut_body_geno, check_nonzero, target_sum=target_sum)
        temp_geno = fill_holes(temp_geno)
        temp_geno = body_largest_cc(temp_geno)

        if valid_mutation(temp_geno):
            return temp_geno

        mut_body_geno = temp_geno

    return body_geno # fallback if no valid mutation found

def mutate_spring_geno(spring_geno, check_nonzero):
    """Mutate spring genotype using mutate_geno helper."""
    return mutate_geno(spring_geno, check_nonzero)

def body_geno_unchanged(body_geno, mut_body_geno):
    """Check if body genotype is unchanged after mutation."""
    return (body_geno == mut_body_geno).all()

def mutate(pop_file, gen_idx, fitness, target_sum=TARGET_N_POM):
    """
        Load population from file, mutate each individual's body and spring genotypes
        with constraints and saving new population as offspring.
        """
    pop = load_pop(pop_file)
    n = len(pop["body_geno"])
    offspring = {
        "body_geno": [],
        "spring_geno": [],
        "id": [],
        "descriptor": []
    }
    for i in range(n):
        bg, sg = pop["body_geno"][i], pop["spring_geno"][i]
        mut_bg = mutate_body_geno(bg, sg, check_nonzero=True, require_change=True, target_sum=target_sum)
        mut_sg = mutate_spring_geno(sg, check_nonzero=True)
        offspring["body_geno"].append(mut_bg)
        offspring["spring_geno"].append(mut_sg)
        offspring["id"].append(f"{gen_idx}-{i}")
        offspring["descriptor"].append(compute_descriptor(mut_bg, mut_sg))
    return save_pop(offspring, pop_file.replace("robots", "offspring"))

