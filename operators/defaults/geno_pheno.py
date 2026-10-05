import numpy as np, os
from copy import deepcopy
from skimage.measure import label
from utils.disk_utils import save_pop, load_pop
from .config import *

# Grid size for body genotype (2D grid)
row = ROW
col = int(np.ceil(row * np.sqrt(3)))  # Approximately square genotype space

# Phenotype scaling and shifts for body mass locations
pheno_scale = PHENO_SCALE
pheno_x_shift = PHENO_X_SHIFT
pheno_y_shift = PHENO_Y_SHIFT


def calculate_symmetry(body_array):
    """
        Calculate vertical and horizontal symmetry of the robot body array.
        Symmetry is measured relative to the central core of the body.
        """
    core_x, core_y = np.mean(np.argwhere(body_array == 1), axis=0)

    vertical_symmetry = 0
    horizontal_symmetry = 0

    # Vertical symmetry: count how many modules have a symmetric counterpart on vertical axis
    for x in range(body_array.shape[1]):
        for y in range(body_array.shape[0]):
            if body_array[y, x] == 1:
                mirrored_x = int(2 * core_x - x)
                if 0 <= mirrored_x < body_array.shape[1] and body_array[y, mirrored_x] == 1:
                    vertical_symmetry += 1

    # Horizontal symmetry: count how many modules have symmetric counterpart on horizontal axis
    for y in range(body_array.shape[0]):
        for x in range(body_array.shape[1]):
            if body_array[y, x] == 1:
                mirrored_y = int(2 * core_y - y)
                if 0 <= mirrored_y < body_array.shape[0] and body_array[mirrored_y, x] == 1:
                    horizontal_symmetry += 1

    total_modules = np.sum(body_array)
    vertical_symmetry /= total_modules
    horizontal_symmetry /= total_modules

    return vertical_symmetry, horizontal_symmetry


def compute_descriptor(body_geno, spring_geno):
    """
        Compute morphological descriptor vector from body and spring genotypes.
        Descriptors include normalized height, width, aspect ratio, balance, compactness, symmetry, and centroid.
        """
    body_array = body_geno.reshape(row, col)

    # Height and width based on occupied cells
    height = np.max(np.where(body_array == 1)[0]) + 1
    width = np.max(np.where(body_array == 1)[1]) + 1
    aspect_ratio = width / height
    bounding_box_area = height * width
    compactness = np.sum(spring_geno) / bounding_box_area
    symmetry_x, symmetry_y = calculate_symmetry(body_array)

    # Centroid normalized by width and height
    module_positions = np.argwhere(body_array == 1)
    centroid_x = np.mean(module_positions[:, 1]) / width
    centroid_y = np.mean(module_positions[:, 0]) / height

    # Balance measured by standard deviation of vertical module positions (lower is balanced)
    balance = np.std(module_positions[:, 0])
    return np.array(
        [height / row, width / col, aspect_ratio, balance, compactness, symmetry_x, symmetry_y, centroid_x, centroid_y])


def triangle_coord_to_idx(x, y):
    return x + y * row


def triangle_idx_to_coord(idx):
    return idx % col, idx // col


def all_node_coords():
    """
        Generate all node coordinates in a hex-like grid pattern,
        interleaving even and odd rows with appropriate horizontal offsets.
        """
    coords = []
    for x in range(0, col + 1, 2):
        for y in range(0, row + 1, 2):
            coords.append((x, y))
    for x in range(-1, col + 1, 2):
        for y in range(1, row + 1, 2):
            coords.append((x, y))
    coords = sorted(list(set(coords)))
    return coords


node_coords = all_node_coords()


def node_to_idx(node):
    """Get index of a node in the node_coords list."""
    global node_coords
    return node_coords.index(node)


def idx_to_node(idx):
    """Get node coordinate from index."""
    global node_coords
    return node_coords[idx]


def all_springs():
    """
        Compute all valid springs (connections) between nodes in the hex grid.
        Returns sorted unique list of tuple node index pairs.
        """
    global node_coords
    springs = []
    for coord in node_coords:
        x, y = coord
        neighbors = [
            (x + 2, y),
            (x - 2, y),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1)
        ]
        neighbors = [n for n in neighbors if n in node_coords]
        for n in neighbors:
            idx0 = node_to_idx(coord)
            idx1 = node_to_idx(n)
            springs.append((min(idx0, idx1), max(idx0, idx1)))
    springs = sorted(list(set(springs)))
    return springs


spring_order = all_springs()
n_springs = len(spring_order)


def pointing_up(idx):
    """Check if the triangle at given index points up in the triangular lattice."""
    x, y = triangle_idx_to_coord(idx)
    return ((x % 2 == 1) & (y % 2 == 0)) | ((x % 2 == 0) & (y % 2 == 1))


def pointing_down(idx):
    """Check if the triangle at given index points down."""
    x, y = triangle_idx_to_coord(idx)
    return ((x % 2 == 1) & (y % 2 == 1)) | ((x % 2 == 0) & (y % 2 == 0))


def points_in_triangle(idx):
    """
        Return node coordinates of the 3 nodes forming the triangular cell with given index.
        """
    x, y = triangle_idx_to_coord(idx)
    points = []
    if pointing_up(idx):
        points.append((x + 1, y))
        points.append((x - 1, y))
        points.append((x, y + 1))
    if pointing_down(idx):
        points.append((x, y))
        points.append((x - 1, y + 1))
        points.append((x + 1, y + 1))
    assert len(points) == 3
    return points


def decompose_triangle(idx):
    """
        Decompose a triangle into its springs (edges) and points (nodes).
        Returns unique springs and points for the triangle.
        """
    points = points_in_triangle(idx)
    springs = []
    for i, p in enumerate(points):
        for j, pj in enumerate(points):
            if i == j:
                continue
            springs.append((node_to_idx(p), node_to_idx(pj)))
    springs = [(min(s[0], s[1]), max(s[0], s[1])) for s in springs]
    return springs, points


triangle_to_points = {}
triangle_to_springs = {}
for i in range(row * col):
    springs, points = decompose_triangle(i)
    triangle_to_points[i] = points
    triangle_to_springs[i] = springs


def body_largest_cc(body_geno):
    """
        Find and retain largest connected component in the body genotype.
        Smaller disconnected components are removed.
        """
    ## Find the largest connected component in the body genotype
    body_geno_arr = deepcopy(body_geno).reshape(row, col)
    labeled, ncomponents = label(body_geno_arr, connectivity=1, return_num=True)
    sizes = [np.sum(labeled == i) for i in range(1, ncomponents + 1)]
    largest = np.argmax(sizes) + 1
    body_geno_arr[labeled != largest] = 0
    body_geno_arr[labeled == largest] = 1
    return body_geno_arr.reshape(row * col)


def fill_holes(body_geno):
    """
        Fill holes in the body genotype.
        If all springs of a triangle are present, mark triangle as present.
        """
    global triangle_to_springs
    body_geno_filled = deepcopy(body_geno)
    springs = []
    idx = np.where(body_geno_filled == 1)[0]
    for i in idx:
        s = triangle_to_springs[i]
        springs.extend(s)
    springs = sorted(list(set(springs)))
    idx = np.where(body_geno_filled == 0)[0]
    for i in idx:
        s = triangle_to_springs[i]
        valid = True
        for si in s:
            if si not in springs:
                valid = False
        if valid:
            body_geno_filled[i] = 1
    return body_geno_filled


def body_to_triangles(body_geno):
    """Convert body genotype bit array into list of triangle indices where bits are 1."""
    triangles = []
    idx = np.where(body_geno == 1)[0]
    for i in idx:
        triangles.append(i)
    return triangles


def rescale_points(points):
    """
        Rescale and shift points coordinates from genotype grid to phenotype space.
        - Translate min x,y to zero
        - Vertically scale y by sqrt(3) for equilateral triangles
        - Scale by pheno_scale
        - Shift by pheno_x_shift and pheno_y_shift
        """
    points = np.array(points)
    minx = np.min(points[:, 0])
    miny = np.min(points[:, 1])
    points[:, 0] -= minx
    points[:, 1] -= miny
    points = points.astype(np.float64)
    ## Make the body mass locations equilateral triangles
    points[:, 1] *= np.sqrt(3)
    ## Rescale all phenotype body mass locations by a factor of 0.025
    points *= pheno_scale
    ## Shift all phenotype body mass locations by (0.025, 0.1)
    points[:, 0] += pheno_x_shift
    points[:, 1] += pheno_y_shift
    return points.tolist()


def random_spring_geno():
    """Random binary spring genotype."""
    return np.random.randint(0, 2, n_springs)


def random_body_geno():
    """Generate random body genotype with holes filled and largest connected component."""
    bg = np.random.randint(0, 2, row * col)
    bg = fill_holes(bg)
    bg = body_largest_cc(bg)
    return bg


def sample_geno():
    """
        Sample random body and spring genotypes,
        convert to phenotype points and springs until desired point and spring count reached.
        """
    rbg = random_body_geno()
    rsg = random_spring_geno()
    p, s = convert(rbg, rsg)
    while len(p) != TARGET_N_POM or len(s) != TARGET_N_SPR:
        rbg = random_body_geno()
        p, s = convert(rbg, rsg)
    return rbg, rsg


def random_geno(n, outdir):
    """
        Generate population of n random genotypes,
        compute descriptors and save as a population pickle file at outdir/robots.pkl.
        """
    pop = {
        "body_geno": [],
        "spring_geno": [],
        "descriptor": [],
        "id": []
    }
    for i in range(n):
        bg, sg = sample_geno()
        pop["body_geno"].append(bg)
        pop["spring_geno"].append(sg)
        pop["descriptor"].append(compute_descriptor(bg, sg))
        pop["id"].append(f"0-{i}")
    return save_pop(pop, os.path.join(outdir, "robots.pkl"))


def convert(body_geno, spring_geno):
    """
        Convert body and spring genotypes to phenotype points and springs list.
        Points and springs are sorted and duplicates removed.
        Springs include physical parameters from config.
        """
    global spring_order, triangle_to_points, triangle_to_springs
    spring_length = SPRING_LENGTH
    spring_K = SPRING_K
    springs = []
    points = []
    triangles = body_to_triangles(body_geno)
    for t in triangles:
        s = triangle_to_springs[t]
        p = triangle_to_points[t]
        points.extend(p)
        springs.extend(s)
    points = sorted(list(set(points)))
    springs = sorted(list(set(springs)))
    mesh_springs = []
    for s in springs:
        a, b = s
        idx = spring_order.index((min(a, b), max(a, b)))
        a, b = idx_to_node(a), idx_to_node(b)
        spr = (
            points.index(a),
            points.index(b),
            spring_length,
            spring_K,
            spring_geno[idx] * SPRING_ACT
        )
        mesh_springs.append(spr)
    points = rescale_points(points)
    return points, mesh_springs


def geno_2_pheno(pop_file):
    """
        Convert saved genotype population file to phenotype representation,
        updating with points, springs, and descriptors and saving back.
        """
    pop = load_pop(pop_file)
    pop_pheno = {
        "points": [],
        "springs": [],
        "descriptor": []
    }
    n = len(pop["body_geno"])
    for i in range(n):
        bg, sg = pop["body_geno"][i], pop["spring_geno"][i]
        points, springs = convert(bg, sg)
        pop_pheno["points"].append(points)
        pop_pheno["springs"].append(springs)
        pop_pheno["descriptor"].append(compute_descriptor(bg, sg))
    pop.update(pop_pheno)
    save_pop(pop, pop_file)
