import os, taichi as ti, math, numpy as np, pickle, sys
from operators.defaults.config import TARGET_N_SPR

# =============================================================================
# CONFIGURATION AND GLOBAL PARAMETERS
# =============================================================================
# Time and optimization parameters

sim_steps = 1000        # Number of simulation time steps per episode
dt = 0.004              # Simulation time constant (step size)
learning_iters = 15     # Number of training/learning iterations

# Physical simulation parameters for terrain and robot
base_ground_height = 0.1
gravity = -4.8          # Downward gravity
spring_omega = 10       # Frequency for periodic (oscillator) NN input
damping_x = 15          # Damping along X direction
damping_y = 15          # Damping along Y direction
friction = 1.0          # Terrain friction
n_ground_segs = 0       # Number of defined ground segments

# Environmental factors and modifiers
wind_factor = 0.0       # Wind intensity
absorb_factor = 1       # Ground absorption during contact
rest_factor = 1.0       # Post-impact rest height
treadmill_vel = 0.0     # Terrain movement simulating treadmill

# Robot architecture parameters
max_springs = 0         # Maximum number of springs per robot
max_objects = 0         # Maximum number of mass points per robot
n_robots = 0            # Number of robots to simulate
n_sub_bots = 1          # Number of subcomponents per robot (for collectives)

# Connector parameters for multi-robot collectives
n_conn_springs = 5      # Springs used for connectors between robots
n_conn_inputs = 12      # NN input features for connectors

# Neural Network architecture settings
n_sin_waves = 10        # How many oscillatory features for "brain" input
n_hidden = 32           # Hidden units in each layer
gradient_clip = 0.16    # Gradient clipping value for training stability

# Inter-level communication (off by default; used by experiments/). Sub-bots are grouped into modules of
# two (s // 2). Level-3 message m3 and per-module level-2 messages m2 are computed from pooled
# bottom-up summaries (mean x-velocity, touch fraction) and fed back into every sub-bot's hidden layer.
comm_enabled = False
n_comm_mod = 1

MAX_INT = np.iinfo(np.int32).max

# =============================================================================
# GLOBAL FIELDS INITIALIZATION
# =============================================================================
# This function creates all Taichi fields, which store simulation state
# and NN parameters. Dimensions and intended meanings for each:
#  - r: robot index
#  - s: sub-bot index (for collectives)
#  - t: simulation step/time
#  - i/j/k: spring or object indices

def set_ti_globals():
    scalarf32 = lambda: ti.field(dtype=ti.f32)
    scalari32 = lambda: ti.field(dtype=ti.i32)
    vecf32 = lambda: ti.Vector.field(2, dtype=ti.f32)
    global x, v, v_inc, center, loss, update_scale, n_objects, n_springs, \
        spring_anchor_a, spring_anchor_b, spring_length, spring_stiffness, spring_actuation, \
            weights1, bias1, weights2, bias2, hidden, act, is_free_fall, touch_sensor, \
            conn_weights1, conn_bias1, conn_weights2, conn_bias2, conn_hidden, conn_update_scale, \
    ground_segs_x0, ground_segs_y0, ground_segs_slope, ground_segs_shift, ground_segs_len, \
        s_ids, p_ids, sub_robot_loss, actual_spring_sub_count, sub_robot_spring_count, sub_robot_obj_count, sub_robot_obj_sub, \
        pool_v, pool_c, msg, comm_in, comm_w2, comm_b2, comm_w3, comm_b3, comm_update_scale
    # Simulation physical state fields
    loss = scalarf32()
    actual_spring_sub_count = scalari32()
    x = vecf32()  # Positions of objects
    v = vecf32()  # Velocities of objects
    v_inc = vecf32()  # Velocity increments for physics updates
    n_objects = scalari32()  # Number of objects per robot
    sub_robot_spring_count = scalari32()
    sub_robot_obj_count = scalari32()
    sub_robot_obj_sub = scalari32()
    p_ids = scalari32()  # IDs mapping object to sub-bot
    n_springs = scalari32()  # Number of springs per robot
    is_free_fall = ti.field(dtype=ti.i32)  # Free-fall status
    spring_anchor_a = scalari32()
    spring_anchor_b = scalari32()
    spring_length = scalarf32()
    spring_stiffness = scalarf32()
    spring_actuation = scalarf32()
    sub_robot_loss = scalarf32()
    hidden = scalarf32()  # NN hidden state
    conn_hidden = scalarf32()  # Connector NN hidden state
    center = vecf32()  # Centers of mass
    act = scalarf32()  # Spring activations (NN output)
    update_scale = scalarf32()
    conn_update_scale = scalarf32()
    ground_segs_x0 = scalarf32()    # Terrain configuration
    ground_segs_y0 = scalarf32()
    ground_segs_slope = scalarf32()
    ground_segs_shift = scalarf32()
    ground_segs_len = scalarf32()
    weights1 = scalarf32()       # NN weights and biases for robot controllers
    bias1 = scalarf32()
    weights2 = scalarf32()
    bias2 = scalarf32()
    conn_weights1 = scalarf32()  # NN weights and biases for connector controllers
    conn_bias1 = scalarf32()
    conn_weights2 = scalarf32()
    conn_bias2 = scalarf32()
    s_ids = scalari32()  # IDs mapping spring to sub-bot
    touch_sensor = scalarf32()  # Contact sensor value per object
    pool_v = scalarf32()  # Pooled mean x-velocity per module (communication)
    pool_c = scalarf32()  # Pooled touch fraction per module (communication)
    msg = scalarf32()  # Messages: [0..n_comm_mod) level-2, index n_comm_mod is level-3
    comm_in = scalarf32()  # Sub-bot hidden-layer weights for incoming messages
    comm_w2 = scalarf32()  # Level-2 controller weights: [module, (pool_v, pool_c, m3)]
    comm_b2 = scalarf32()
    comm_w3 = scalarf32()  # Level-3 controller weights: [module, (pool_v, pool_c)]
    comm_b3 = scalarf32()
    comm_update_scale = scalarf32()

# =============================================================================
# NN INPUT DIMENSION FUNCTIONS
# =============================================================================

# Population maximum NN input dimension (used for memory allocation)
def max_input_states():
    return n_sin_waves + 4 * max_objects

@ti.func
def max_input_states_ti():
    return n_sin_waves + 4 * max_objects

# Individual robot NN input dimension
def n_input_states(robot_id):
    return n_sin_waves + 4 * n_objects[robot_id]

@ti.func
def n_input_states_ti(robot_id):
    return n_sin_waves + 4 * n_objects[robot_id]

# =============================================================================
# MEMORY ALLOCATION: TAICHI FIELD LAYOUTS
# =============================================================================
# This allocates field memory for all global fields, with
# appropriate pre-allocated sizes for efficient kernel operations.
def allocate_fields():
    ti.root.dense(ti.ijk, (n_robots, sim_steps, max_objects)).place(x, v, v_inc)
    ti.root.dense(ti.ij, (n_robots, max_springs)).place(spring_anchor_a, spring_anchor_b, spring_length, spring_stiffness, spring_actuation)
    ti.root.dense(ti.i, n_robots).place(n_objects, n_springs)
    ti.root.dense(ti.ijk, (n_robots, n_sub_bots, sim_steps)).place(is_free_fall)
    ti.root.dense(ti.ij, (n_robots, n_sub_bots)).place(sub_robot_loss, sub_robot_spring_count, sub_robot_obj_count, actual_spring_sub_count, sub_robot_obj_sub)
    ti.root.dense(ti.ijkl, (n_robots, n_sub_bots, n_hidden, max_input_states())).place(weights1)
    ti.root.dense(ti.ijk, (n_robots, n_sub_bots, n_hidden)).place(bias1)
    ti.root.dense(ti.ijkl, (n_robots, n_sub_bots, max_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.ijk, (n_robots, n_sub_bots, max_springs)).place(bias2)

    ti.root.dense(ti.ij, (n_robots, max_springs)).place(s_ids)
    ti.root.dense(ti.ij, (n_robots, max_objects)).place(p_ids)
    ti.root.dense(ti.ijk, (n_robots, sim_steps, max_springs)).place(act)
    ti.root.dense(ti.ijkl, (n_robots, n_sub_bots, sim_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ijk, (n_robots, n_sub_bots, sim_steps)).place(center)
    ti.root.dense(ti.i, n_robots).place(loss, update_scale)
    ti.root.dense(ti.i, (n_ground_segs)).place(ground_segs_x0, ground_segs_y0, ground_segs_slope, ground_segs_shift, ground_segs_len)
    ti.root.dense(ti.ijk, (n_robots, sim_steps, max_objects)).place(touch_sensor)

    ti.root.dense(ti.ijkl, (n_robots, sim_steps, max_springs, n_hidden)).place(conn_hidden)
    ti.root.dense(ti.ij, (n_hidden, n_conn_inputs * max_springs)).place(conn_weights1)
    ti.root.dense(ti.i, (n_hidden)).place(conn_bias1)
    ti.root.dense(ti.ijk, (1, max_springs, n_hidden)).place(conn_weights2)
    ti.root.dense(ti.ij, (1, max_springs)).place(conn_bias2)
    ti.root.dense(ti.i, 1).place(conn_update_scale)
    global n_comm_mod
    n_comm_mod = max(1, n_sub_bots // 2)
    ti.root.dense(ti.ijk, (n_robots, sim_steps, n_comm_mod)).place(pool_v, pool_c)
    ti.root.dense(ti.ijk, (n_robots, sim_steps, n_comm_mod + 1)).place(msg)
    ti.root.dense(ti.ijk, (n_sub_bots, n_hidden, 2)).place(comm_in)
    ti.root.dense(ti.ij, (n_comm_mod, 3)).place(comm_w2)
    ti.root.dense(ti.i, n_comm_mod).place(comm_b2)
    ti.root.dense(ti.ij, (n_comm_mod, 2)).place(comm_w3)
    ti.root.dense(ti.i, 1).place(comm_b3, comm_update_scale)
    # Pre-allocate lazy gradient storage for all fields used in backward pass
    ti.root.lazy_grad()

# =============================================================================
# COMPUTE CENTER OF MASS FOR EACH ROBOT SUBCOMPONENT
# =============================================================================
# Calculates center of mass for each sub-robot within each robot at a given timestep.
# This is used for NN inputs and tracking overall robot translation.
# - r: robot index
# - i: object index within robot
# - s: sub-robot index
# - t: simulation timestep
@ti.kernel
def compute_center(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            bot_id = p_ids[r, i] - 1 # Map object to sub-robot id (0-indexed)
            if bot_id >= 0:
                center[r, bot_id, t] += x[r, t, i] # Accumulate position for center calc
    for r, s in ti.ndrange(n_robots, n_sub_bots):
        # Average accumulated positions to get center of mass for sub-robot s
        center[r, s, t] += (1.0 / sub_robot_obj_count[r, s]) * center[r, s, t] - center[r, s, t]

# =============================================================================
# INTER-LEVEL COMMUNICATION: POOLED SUMMARIES AND MESSAGES
# =============================================================================
# Bottom-up: each module (pair of sub-bots) pools mean x-velocity and touch fraction of its points.
# The level-3 controller reads both module pools and emits m3; each level-2 controller reads its
# module pool plus m3 (top-down) and emits m2. nn1 feeds (m2 of own module, m3) into every sub-bot.
@ti.kernel
def comm_signals(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            s = p_ids[r, i] - 1
            if s >= 0:
                w = 1.0 / (2.0 * sub_robot_obj_count[r, s])
                pool_v[r, t, s // 2] += v[r, t, i][0] * 10 * w
                pool_c[r, t, s // 2] += touch_sensor[r, t, i] * w
    for r, m in ti.ndrange(n_robots, n_comm_mod):
        msg[r, t, n_comm_mod] += comm_w3[m, 0] * pool_v[r, t, m] + comm_w3[m, 1] * pool_c[r, t, m]
    for r in range(n_robots):
        msg[r, t, n_comm_mod] += comm_b3[0]
        msg[r, t, n_comm_mod] += ti.tanh(msg[r, t, n_comm_mod]) - msg[r, t, n_comm_mod]
    for r, m in ti.ndrange(n_robots, n_comm_mod):
        msg[r, t, m] += comm_w2[m, 0] * pool_v[r, t, m] + comm_w2[m, 1] * pool_c[r, t, m] \
            + comm_w2[m, 2] * msg[r, t, n_comm_mod]
        msg[r, t, m] += comm_b2[m]
        msg[r, t, m] += ti.tanh(msg[r, t, m]) - msg[r, t, m]

# =============================================================================
# NEURAL NETWORK PART 1: COMPUTE HIDDEN LAYER ACTIVATIONS
# =============================================================================
# Performs forward pass first layer computation of activations for each sub-robot's NN,
# at a given timestep.
# Inputs include periodic sine waves (oscillator) proprioceptive features (positions, velocities etc).
#
# Hidden state is accumulated and then passed through nonlinear activation (modified tanh).
@ti.kernel
def nn1(t: ti.i32):
    # Phase-offset sinusoidal input features for generating rhythmic patterns
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, n_hidden, n_sin_waves):
        hidden[r, s, t, i] += weights1[r, s, i, j] * ti.sin(spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)

    # Proprioceptive input features for mass points: relative position offsets and velocities
    for r, i, j in ti.ndrange(n_robots, n_hidden, max_objects):
        if j < n_objects[r]:
            s = p_ids[r, j] - 1
            if s >= 0:
                to_sub = sub_robot_obj_sub[r, s]
                offset = x[r, t, j] - center[r, s, t]
                # Add weighted position offsets (x,y) * 10 for feature scale
                hidden[r, s, t, i] += weights1[r, s, i, (j - to_sub) * 4 + n_sin_waves] * offset[0] * 10
                hidden[r, s, t, i] += weights1[r, s, i, (j - to_sub) * 4 + n_sin_waves + 1] * offset[1] * 10
                # Add weighted velocity features (vx, vy)
                hidden[r, s, t, i] += weights1[r, s, i, (j - to_sub) * 4 + n_sin_waves + 2] * v[r, t, j][0]
                hidden[r, s, t, i] += weights1[r, s, i, (j - to_sub) * 4 + n_sin_waves + 3] * v[r, t, j][1]

    # Connector proprioceptive inputs for springs that connect sub-robots
    for r, s, i in ti.ndrange(n_robots, max_springs, n_hidden):
        if s_ids[r, s] < 0: # Connector springs have negative s_id
            a = spring_anchor_a[r, s]
            b = spring_anchor_b[r, s]
            # Ensure order of anchors (a < b) for consistent input representation
            if x[r, t, b][0] < x[r, t, a][0]:
                b = spring_anchor_a[r, s]
                a = spring_anchor_b[r, s]
            if x[r, t, b][0] >= x[r, t, a][0]:
                a = spring_anchor_a[r, s]
                b = spring_anchor_b[r, s]
            pos_a = x[r, t, a]
            pos_b = x[r, t, b]
            center_i = (pos_a + pos_b) / 2
            rel_pos_a = pos_a - center_i
            rel_pos_b = pos_b - center_i
            v_a = v[r, t, a]
            v_b = v[r, t, b]
            # Compose feature vector of relative positions, velocities, touch sensors and actuation
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 0] * rel_pos_a[0]
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 1] * rel_pos_a[1]
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 2] * rel_pos_b[0]
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 3] * rel_pos_b[1]
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 4] * v_a[0] / 10
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 5] * v_a[1] / 10
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 6] * v_b[0] / 10
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 7] * v_b[1] / 10
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 8] * touch_sensor[r, t, a]
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 9] * touch_sensor[r, t, b]
            # This input will not actually do anything, because act[r, t, s] is always zero at this point. I noticed
            # this too late, my intent was to use act[r, t-1, s]. Keeping this in here for shape compatibility.
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 10] * act[r, t, s]
            conn_hidden[r, t, s, i] += conn_weights1[i, s * n_conn_inputs + 11] * spring_length[r, s]

    # Incoming inter-level messages (only when communication is enabled)
    if ti.static(comm_enabled):
        for r, s, i in ti.ndrange(n_robots, n_sub_bots, n_hidden):
            hidden[r, s, t, i] += comm_in[s, i, 0] * msg[r, t, s // 2] + comm_in[s, i, 1] * msg[r, t, n_comm_mod]

    # Apply bias and nonlinear activation (modified tanh) for sub-robots
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, n_hidden):
        hidden[r, s, t, i] += bias1[r, s, i]
        hidden[r, s, t, i] += ti.tanh(hidden[r, s, t, i]) - hidden[r, s, t, i]
    # Apply bias and activation for connectors
    for r, s, i in ti.ndrange(n_robots, max_springs, n_hidden):
        if s_ids[r, s] < 0:
            conn_hidden[r, t, s, i] += conn_bias1[i]
            conn_hidden[r, t, s, i] += ti.tanh(conn_hidden[r, t, s, i]) - conn_hidden[r, t, s, i]

# =============================================================================
# NEURAL NETWORK PART 2: COMPUTE SPRING ACTIVATIONS
# =============================================================================
# Calculates final output activations controlling spring actuations,
# using hidden activations from nn1.
@ti.kernel
def nn2(t: ti.i32):
    # Weighted sum for bot springs and connector springs
    for r, i, j in ti.ndrange(n_robots, max_springs, n_hidden):
        if i < n_springs[r]:
            s = s_ids[r, i] - 1
            if s >= 0: # bot springs
                to_sub = sub_robot_spring_count[r, s]
                act[r, t, i] += weights2[r, s, i - to_sub, j] * hidden[r, s, t, j]
            else: # connector springs
                act[r, t, i] += conn_weights2[0, i, j] * conn_hidden[r, t, i, j]

    # Add bias and apply activation with smoothing
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            s = s_ids[r, i] - 1
            if s >= 0: # bot spring outputs
                to_sub = sub_robot_spring_count[r, s]
                act[r, t, i] += bias2[r, s, i - to_sub]
                act[r, t, i] += ti.tanh(act[r, t, i]) - act[r, t, i]
                # Temporal smoothing for stability
                act[r, t, i] *= 0.15
                act[r, t, i] += 0.85 * act[r, t-1, i]
            else: # connector outputs
                act[r, t, i] += conn_bias2[0, i]
                act[r, t, i] += ti.tanh(act[r, t, i]) - act[r, t, i]
                act[r, t, i] *= 0.1
                act[r, t, i] += 0.9 * act[r, t - 1, i]

# =============================================================================
# APPLY SPRING FORCES TO OBJECT VELOCITIES
# =============================================================================
# Calculate impulses from spring deformation and neural activation,
# then apply to object velocity increments for physics integration.
@ti.kernel
def apply_spring_force(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_springs):
        if i < n_springs[r]:
            a = spring_anchor_a[r, i]
            b = spring_anchor_b[r, i]
            pos_a = x[r, t, a]
            pos_b = x[r, t, b]
            dist = pos_a - pos_b
            length = dist.norm() + 1e-4 # Avoid division by zero
            # Target spring length modulated by neural actuation
            target_length = spring_length[r, i] * (1.0 + spring_actuation[r, i] * act[r, t, i])
            impulse = dt * (length - target_length) * spring_stiffness[r, i] / length * dist
            ti.atomic_add(v_inc[r, t + 1, a], -impulse)
            ti.atomic_add(v_inc[r, t + 1, b], impulse)

# =============================================================================
# UTILITY FUNCTIONS FOR SPRINGS, GROUNDS, COLLISIONS
# =============================================================================
@ti.func
def compute_spring_state(r: ti.i32, t: ti.i32, i: ti.i32):
    a = spring_anchor_a[r, i]
    b = spring_anchor_b[r, i]
    pos_a = x[r, t, a]
    pos_b = x[r, t, b]
    return ((pos_a - pos_b).norm() / spring_length[r, i]) - 1

@ti.func
def get_wind(t: ti.i32):
    # Returns wind vector at timestep t
    return ti.Vector([wind_factor * ti.sin(t * 0.02), 0.0])

@ti.func
def compute_spring_center(r: ti.i32, t: ti.i32, i: ti.i32) -> ti.Vector:
    a = spring_anchor_a[r, i]
    b = spring_anchor_b[r, i]
    return (x[r, t, a] + x[r, t, b]) / 2


@ti.func
def compute_relative_position(r: ti.i32, t: ti.i32, i: ti.i32, j: ti.i32) -> ti.Vector:
    center_i = compute_spring_center(r, t, i)
    center_j = compute_spring_center(r, t, j)
    return center_j - center_i

def ground_height_at_(x_val, seg):
    ground_height = base_ground_height
    if x_val >= 0 and seg >= 0:
        slope = ground_segs_slope[seg]
        shift = ground_segs_shift[seg]
        ground_height = x_val * slope + shift
    return ground_height

@ti.func
def ground_height_at(x_val: ti.f32, seg: ti.i32):
    ground_height = base_ground_height
    if x_val >= 0 and seg >= 0:
        slope = ground_segs_slope[seg]
        shift = ground_segs_shift[seg]
        ground_height = x_val * slope + shift
    return ground_height

def ground_seg_at_(x_val):
    seg = 0
    if x_val >= 0:
        for i in range(n_ground_segs):
            if x_val >= ground_segs_x0[i]:
                seg = i
    return seg

@ti.func
def ground_seg_at(x_val: ti.f32):
    seg = 0
    if x_val >= 0:
        ti.loop_config(serialize=True)
        for i in ti.static(range(n_ground_segs)):
            if x_val >= ground_segs_x0[i]:
                seg = i
    return seg

@ti.func
def distance_to_ground_at(x_val: ti.f32, y_val: ti.f32, seg: ti.i32):
    slope = ground_segs_slope[seg]
    shift = ground_segs_shift[seg]
    return ti.abs(-slope * x_val + y_val - shift) / ti.sqrt(1 + slope**2)

@ti.func
def normal_vec(seg: ti.i32):
    slope = ground_segs_slope[seg]
    return ti.Vector([-slope / ti.sqrt(1 + slope**2), 1 / ti.sqrt(1 + slope**2)])

@ti.func
def compute_toi(seg: ti.i32, x_val: ti.f32, y_val: ti.f32, vx: ti.f32, vy: ti.f32):
    dist = distance_to_ground_at(x_val, y_val, seg)
    norm_vec = normal_vec(seg)
    v = ti.Vector([vx, vy])
    norm_vec_mag = ti.abs(v.dot(norm_vec))
    toi = dist / (norm_vec_mag + 1e-10)
    return toi

@ti.func
def custom_sign(var):
    ret = 1.0
    if var < 0:
        ret = -1.0
    if var == 0:
        ret = 0.0
    return ret

@ti.func
def new_v_on_contact(seg: ti.i32, vx: ti.f32, vy: ti.f32):
    norm_vec = normal_vec(seg)
    v = ti.Vector([vx, vy])
    v[0] += treadmill_vel
    norm_vec_scale = v.dot(norm_vec)
    norm_vec = norm_vec * norm_vec_scale
    norm_vec_mag = norm_vec.norm()
    tan_vec = v - norm_vec * absorb_factor

    tan_vec_mag = tan_vec.norm()
    friction_vec = tan_vec * -1
    friction_vec = friction_vec.normalized() * friction * ti.min(norm_vec_mag, tan_vec_mag)
    new_v = tan_vec + friction_vec
    if ti.abs(ground_segs_slope[seg]) > 0.8:
        new_v = tan_vec
    new_v[0] -= treadmill_vel
    return new_v, norm_vec

# =============================================================================
# ADVANCE PHYSICS SIMULATION ONE STEP
# =============================================================================
# Update positions and velocities of all mass points in all robots at timestep tm.
# This integrates velocity increments, applies damping, gravity, wind, and processes
# collisions with the terrain. It calculates new positions and velocities, updating
# touch sensors and free-fall states accordingly.
@ti.kernel
def advance(tm: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            touch_sensor[r, tm, i] = 0.0
            sb = p_ids[r, i] - 1 # sub-robot id for this object
            s_x = ti.exp(-dt * damping_x) # damping factors
            s_y = ti.exp(-dt * damping_y)
            v_x = s_x * v[r, tm - 1, i][0] + v_inc[r, tm, i][0]
            # Special velocity boost on first timestep (mostly relevant for the low gravity environment, as the
            # bot(s) would just float and not really move much without this)
            if tm == 1:
                v_x += 0.15
            # If in free fall, disable y damping
            if sb > 0 and is_free_fall[r, sb, tm-1] == sub_robot_obj_count[r, sb]:
                s_y = ti.f32(1.0)
            v_y = s_y * v[r, tm - 1, i][1] + dt * gravity + v_inc[r, tm, i][1]
            v_ = ti.Vector([v_x, v_y]) + get_wind(tm)
            old_x = x[r, tm - 1, i]
            new_x = old_x + dt * v_
            new_x[0] -= dt * treadmill_vel

            seg_new_x = ground_seg_at(new_x[0])
            # Adjust segment id if slope < 1 with very short segment length
            if ground_segs_slope[seg_new_x] < 1 and ground_segs_len[seg_new_x] < 0.01:
                seg_new_x += 1
                new_x[0] = ground_segs_x0[seg_new_x]
            ground_height = ground_height_at(new_x[0], seg_new_x)
            # Check if object is above ground + small margin
            if new_x[1] > ground_height + 0.03:
                is_free_fall[r, sb, tm] += 1
            # Handle collision with ground if below ground height
            if new_x[1] < ground_height:
                seg_old_x = ground_seg_at(old_x[0])
                s0 = ti.min(seg_old_x, seg_new_x)
                s1 = ti.max(seg_old_x, seg_new_x)
                toi = compute_toi(s0, old_x[0], old_x[1], v_[0], v_[1])
                for j in ti.static(range(n_ground_segs)):
                    if j > s0 and j <= s1:
                        toi = ti.min(toi, compute_toi(j, old_x[0], old_x[1], v_[0], v_[1]))

                toi = ti.min(ti.max(0, toi), dt)
                new_x = old_x + toi * v_
                if new_x[1] > ground_height + 0.03:
                    is_free_fall[r, sb, tm] += 1
                seg_new_x = ground_seg_at(new_x[0])
                # Adjust velocity on contact with ground, add friction
                v_, norm_v = new_v_on_contact(seg_new_x, v_[0], v_[1])
                touch_sensor[r, tm, i] = 1.0
                if absorb_factor > 1:
                    new_x[1] += norm_v[1] * dt
                    if new_x[1] > ground_height + 0.03:
                        is_free_fall[r, sb, tm] += 1
                ground_height = ground_height_at(new_x[0], seg_new_x)
                if toi < dt:
                    new_x = new_x + (dt - toi) * v_
                    if new_x[1] > ground_height + 0.03:
                        is_free_fall[r, sb, tm] += 1
                    seg_new_x = ground_seg_at(new_x[0])
                    ground_height = ground_height_at(new_x[0], seg_new_x)

                if new_x[1] < ground_height:
                    new_x_1 = ground_height
                    if absorb_factor > 1:
                        new_x_1 = new_x[1] + (ground_height - new_x[1]) * rest_factor
                    new_x[1] = new_x_1
                    if new_x[1] > ground_height + 0.03:
                        is_free_fall[r, sb, tm] += 1
            v[r, tm, i] = v_
            x[r, tm, i] = new_x

# =============================================================================
# LOSS COMPUTATION
# =============================================================================
# Compute the loss as the negative horizontal displacement of the center of mass
# for each robot at time t. This encourages forward motion.
@ti.kernel
def compute_loss(t: ti.i32):
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            loss[r] += x[r, t, i][0] - x[r, 0, i][0]
    for r in range(n_robots):
        loss[r] += (-1.0 / n_objects[r]) * loss[r] - loss[r]
        if abs(loss[r]) > 10:
            loss[r] /= 1000

# Compute loss per sub-robot for detailed evaluation
@ti.kernel
def compute_loss_sub(t: ti.i32):
    # Reset sub-robot losses
    objects_to_count = 0.0
    for r, b in ti.ndrange(n_robots, n_sub_bots):
        sub_robot_loss[r, b] = 0.0
    # Assign displacement per mass point to corresponding sub-robot loss
    for r, i in ti.ndrange(n_robots, max_objects):
        if i < n_objects[r]:
            start_x = x[r, 0, i][0]
            end_x = x[r, t, i][0]
            displacement = end_x - start_x
            bot_id = p_ids[r, i] - 1
            if bot_id >= 0:
                ti.atomic_add(sub_robot_loss[r, bot_id], displacement)
                objects_to_count += 1
    for r in range(n_robots):
        for b in range(n_sub_bots):
            if sub_robot_obj_count[r, b] > 0:
                sub_robot_loss[r, b] /= sub_robot_obj_count[r, b]

# =============================================================================
# CLEAR AND RESET FUNCTIONS
# =============================================================================
@ti.kernel
def clear_states():
    conn_update_scale[0] = 0.0
    comm_update_scale[0] = 0.0
    for r, t, m in ti.ndrange(n_robots, sim_steps, n_comm_mod):
        pool_v[r, t, m] = 0.0
        pool_c[r, t, m] = 0.0
    for r, t, m in ti.ndrange(n_robots, sim_steps, n_comm_mod + 1):
        msg[r, t, m] = 0.0
    for r, s, t in ti.ndrange(n_robots, n_sub_bots, sim_steps):
        center[r, s, t] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_objects):
        v_inc[r, t, i] = ti.Vector([0.0, 0.0])
        v[r, t, i] = ti.Vector([0.0, 0.0])
        if t > 0:
            x[r, t, i] = ti.Vector([0.0, 0.0])
        touch_sensor[r, t, i] = 0.0
    for r, s, t, i in ti.ndrange(n_robots, n_sub_bots, sim_steps, n_hidden):
        hidden[r, s, t, i] = 0.0
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_springs):
        act[r, t, i] = 0.0
    for r in ti.ndrange(n_robots):
        update_scale[r] = 0.0
    for r, t, i, k in ti.ndrange(n_robots, sim_steps, max_springs, n_hidden):
        conn_hidden[r, t, i, k] = 0.0

## Reset gradients
@ti.kernel
def clear_grad():
    for r, t, i, k in ti.ndrange(n_robots, sim_steps, max_springs, n_hidden):
        conn_hidden.grad[r, t, i, k] = 0.0
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, n_hidden):
        bias1.grad[r, s, i] = 0.0
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, n_hidden, max_springs):
        weights2.grad[r, s, j, i] = 0.0
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, n_hidden, max_input_states_ti()):
        weights1.grad[r, s, i, j] = 0.0
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, max_springs):
        bias2.grad[r, s, i] = 0.0

    for r, t, m in ti.ndrange(n_robots, sim_steps, n_comm_mod):
        pool_v.grad[r, t, m] = 0.0
        pool_c.grad[r, t, m] = 0.0
    for r, t, m in ti.ndrange(n_robots, sim_steps, n_comm_mod + 1):
        msg.grad[r, t, m] = 0.0
    for s, i, k in ti.ndrange(n_sub_bots, n_hidden, 2):
        comm_in.grad[s, i, k] = 0.0
    for m, k in ti.ndrange(n_comm_mod, 3):
        comm_w2.grad[m, k] = 0.0
    for m in ti.ndrange(n_comm_mod):
        comm_b2.grad[m] = 0.0
    for m, k in ti.ndrange(n_comm_mod, 2):
        comm_w3.grad[m, k] = 0.0
    comm_b3.grad[0] = 0.0

    for i in ti.ndrange(n_hidden):
        conn_bias1.grad[i] = 0.0
    for i, j, k in ti.ndrange(1, max_springs, n_hidden):
        conn_weights2.grad[i, j, k] = 0.0
    for i, j in ti.ndrange(n_hidden, n_conn_inputs*max_springs):
        conn_weights1.grad[i, j] = 0.0
    for i, j in ti.ndrange(1, max_springs):
        conn_bias2.grad[i, j] = 0.0

    for r, i in ti.ndrange(n_robots, max_springs):
        spring_length.grad[r, i] = 0.0
        spring_stiffness.grad[r, i] = 0.0
        spring_actuation.grad[r, i] = 0.0
    for r, s, t in ti.ndrange(n_robots, n_sub_bots, sim_steps):
        center.grad[r, s, t] = ti.Vector([0.0, 0.0])
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_objects):
        v_inc.grad[r, t, i] = ti.Vector([0.0, 0.0])
        touch_sensor.grad[r, t, i] = 0.0
        v.grad[r, t, i] = ti.Vector([0.0, 0.0])
        if t > 0:
            x.grad[r, t, i] = ti.Vector([0.0, 0.0])
    for r, s, t, i in ti.ndrange(n_robots, n_sub_bots, sim_steps, n_hidden):
       hidden.grad[r, s, t, i] = 0.0
    for r, t, i in ti.ndrange(n_robots, sim_steps, max_springs):
        act.grad[r, t, i] = 0.0
    for i in ti.ndrange(n_ground_segs):
        ground_segs_x0.grad[i] = 0.0
        ground_segs_y0.grad[i] = 0.0
        ground_segs_slope.grad[i] = 0.0
        ground_segs_shift.grad[i] = 0.0
        ground_segs_len.grad[i] = 0.0

## Reset loss history
@ti.kernel
def clear_loss():
    loss.fill(0.0)

# =============================================================================
# RESET ALL SIMULATION AND TRAINING STATE
# =============================================================================
def clear():
    clear_states()
    clear_loss()
    clear_grad()

# =============================================================================
# ADJUST INITIAL HEIGHT OF ROBOT MASS POINTS TO ENSURE ABOVE GROUND START
# =============================================================================
# Checks each mass point's initial y-position against ground height at their x-position.
# Computes maximum offset needed to raise all points above ground and applies vertical offset.
@ti.kernel
def adjust_initial_height(id: ti.i32, n_obj: ti.i32):
    max_offset = 0.0
    # Loop over all objects for this robot to find largest downward offset
    for i in range(n_obj):
        x_val = x[id, 0, i][0] # x-position of mass point at initial timestep
        height = x[id, 0, i][1] # y-position
        seg = ground_seg_at(x_val) # find ground segment index at this x
        ground_height = ground_height_at(x_val, seg) # ground height there
        offset = ground_height - height # difference below ground
        if height < ground_height and offset > max_offset:
            max_offset = offset # track largest needed upward offset
    # Increase y of all mass points by max_offset to ensure clearance over ground
    for i in range(n_obj):
        x[id, 0, i][1] = x[id, 0, i][1] + max_offset

# =============================================================================
# SETUP INDIVIDUAL ROBOT IN TAICHI FIELD MEMORY
# =============================================================================
# Populate global Taichi arrays with initial mass point positions, spring connectivity, and properties.
# Also assign mass points and springs to sub-robots (parts of the robot).
# Arguments:
# - id: robot index
# - n_obj: number of mass points in robot
# - objects: ndarray of shape (n_obj, 2) containing initial positions
# - n_spr: number of springs in robot
# - springs: ndarray of shape (n_spr, 5) with spring info: anchors, length, stiffness, actuation
# - bot_ids: array specifying sub-robot id ownership for springs
# - bot_p_ids: array specifying sub-robot id ownership for mass points
@ti.kernel
def setup_robot(id: ti.i32, n_obj: ti.i32, objects: ti.types.ndarray(), n_spr: ti.i32, springs: ti.types.ndarray(), bot_ids: ti.types.ndarray(), bot_p_ids: ti.types.ndarray()): # type: ignore
    n_objects[id] = n_obj
    for i in range(n_obj):
        x[id, 0, i] = ti.Vector([objects[i, 0], objects[i, 1]]) # set initial position

    n_springs[id] = n_spr
    active_springs = 0.0
    for i in range(n_obj):
        p_ids[id, i] = bot_p_ids[i] # assign sub-robot id to mass point
        if bot_p_ids[i] > 0:
            sub_robot_obj_count[id, bot_p_ids[i]-1] += 1 # count objects per sub-robot
    # Compute cumulative object index offsets per sub-robot for NN indexing
    for i in range(1, n_sub_bots):
        to_add = sub_robot_obj_count[id, i - 1]
        for j in range(i, n_sub_bots):
            sub_robot_obj_sub[id, j] += to_add
    for i in range(n_spr):
        spring_anchor_a[id, i] = ti.cast(springs[i, 0], ti.i32)
        spring_anchor_b[id, i] = ti.cast(springs[i, 1], ti.i32)
        spring_length[id, i] = springs[i, 2]
        spring_stiffness[id, i] = springs[i, 3]
        spring_actuation[id, i] = springs[i, 4]
        s_ids[id, i] = bot_ids[i] # sub-robot or connector id for spring
        if bot_ids[i] > 0:
            actual_spring_sub_count[id, bot_ids[i] - 1] += 1 # count springs in sub-robot
            # Compute cumulative spring count offsets for NN indexing
            for j in range(bot_ids[i], n_sub_bots):
               sub_robot_spring_count[id, j] += 1
        if spring_actuation[id, i] > 0:
            active_springs += 1
        x[id, 0, i] = ti.Vector([objects[i, 0], objects[i, 1]])

# =============================================================================
# SETUP ALL ROBOTS (AND CONNECTORS IF ANY) FROM FILES INTO TAICHI
# =============================================================================
# Loads robot data from pickle files, computes limits for objects and springs,
# prepares ground terrain data, allocates memory, sets initial robot states,
# and loads NN weights (either provided or random initialization).
# Supports partial loading via idx0 and idx1 (range in robot dataset).
def setup(robots_file, ground_file, conn_file=None, idx0=None, idx1=None):
    global n_robots, max_objects, max_springs, n_ground_segs

    with open(robots_file, "rb") as f:
        robots = pickle.load(f)

    if conn_file is not None:
        with open(conn_file, "rb") as f:
            connector = pickle.load(f)

    if idx0 is None:
        idx0 = 0
    if idx1 is None:
        idx1 = len(robots['points'])

    print(f"Loading robots [{idx0}, {idx1}) into Taichi from {robots_file}...", flush=True)

    # Extract robot spring and point data subset
    all_springs = robots['springs'][idx0:idx1]
    all_objects = robots['points'][idx0:idx1]

    # Default bot spring and point id assignments (all 1)
    bot_s_ids = []
    bot_p_ids = []
    for r in range(idx0, idx1):
        bot_s_ids.append([1 for _ in range(len(all_springs[r]))])
        bot_p_ids.append([1 for _ in range(len(all_objects[r]))])
    # Overwrite if 's_id' or 'p_id' keys exist for collective assignment
    if "s_id" in robots.keys():
        bot_s_ids = robots['s_id'][idx0:idx1]
    if "p_id" in robots.keys():
        bot_p_ids = robots['p_id'][idx0:idx1]
    n_robots = len(all_objects)
    max_objects = max([len(o) for o in all_objects])
    max_springs = max([len(s) for s in all_springs])
    print(f"num robots: {n_robots}, max num points: {max_objects}, max num springs: {max_springs}", flush=True)
    # Load and prepare ground segments, or use default flat ground
    if ground_file is None:
        n_ground_segs = 3
        print(f"Using flat terrain...", flush=True)
    else:
        ground = np.load(ground_file)
        xs, ys, lens, slopes, shifts = ground
        n_ground_segs = len(xs)
        print(f"Loading terrain from {ground_file}...", flush=True)
        extension_length = 0.25  # extra length added to first flat segment
        first_flat_segment_index = 1
        # Extend the length of the first flat segment
        lens[first_flat_segment_index] += extension_length
        # Shift x-coordinates of all subsequent segments
        for i in range(first_flat_segment_index + 1, n_ground_segs):
            xs[i] += extension_length
        # Recalculate shifts for all segments after the extended one
        for i in range(first_flat_segment_index + 1, n_ground_segs):
            # Calculate new shift for this segment
            shifts[i] = ys[i] - slopes[i] * xs[i]
    print(f"n_ground_segs: {n_ground_segs}", flush=True)

    allocate_fields()

    # Set ground segment data into Taichi fields
    if ground_file is None:
        # Default flat ground segments
        ground_segs_x0[0] = -20.0
        ground_segs_y0[0] = base_ground_height
        ground_segs_len[0] = 0.2
        ground_segs_slope[0] = 0.0
        ground_segs_shift[0] = base_ground_height
        ground_segs_x0[1] = 0.2
        ground_segs_y0[1] = base_ground_height
        ground_segs_len[1] = 0.2
        ground_segs_slope[1] = 0.0
        ground_segs_shift[1] = base_ground_height
        ground_segs_x0[2] = 0.4
        ground_segs_y0[2] = base_ground_height
        ground_segs_len[2] = 4.6
        ground_segs_slope[2] = 0.0
        ground_segs_shift[2] = base_ground_height
    else:
        for i in range(n_ground_segs):
            ground_segs_x0[i] = xs[i]
            ground_segs_y0[i] = ys[i]
            ground_segs_slope[i] = slopes[i]
            ground_segs_shift[i] = shifts[i]
            ground_segs_len[i] = lens[i]

    # Setup each robot's initial states and offsets
    for robot_id in range(0, n_robots):
        obj = np.array(all_objects[robot_id], dtype=np.float32)
        spr = np.array(all_springs[robot_id], dtype=np.float32)
        bot_id = np.array(bot_s_ids[robot_id], dtype=np.int32)
        bot_id_p = np.array(bot_p_ids[robot_id], dtype=np.int32)
        n_obj, n_spr = len(obj), len(spr)
        setup_robot(robot_id, n_obj, obj, n_spr, spr, bot_id, bot_id_p)
    print("Robot states loaded...", flush=True)

    # Adjust initial heights of each robot so masses start above ground
    for robot_id in range(0, n_robots):
        obj = np.array(all_objects[robot_id], dtype=np.float32)
        n_obj = len(obj)
        adjust_initial_height(robot_id, n_obj)

    # Load or initialize neural network weights
    if "weights" in robots:
        ## Load provided weights for visualization
        print("Loading provided weights...", flush=True)
        init_weights()
        for robot_id in range(0, n_robots):
            weights = robots["weights"][robot_id]
            # If collective: load sub-bot weights and connector weights if any
            if n_sub_bots > 1:
                for sub_bot_id in range(n_sub_bots):
                    weights_1 = weights[sub_bot_id][0]["w1"]
                    ipt_dim = min(weights1.to_numpy().shape[-1], weights_1.shape[-1])
                    fill_weights_ti(robot_id, sub_bot_id, weights[sub_bot_id][0]["w1"], weights[sub_bot_id][0]["b1"], weights[sub_bot_id][0]["w2"], weights[sub_bot_id][0]["b2"], ipt_dim)
                if conn_file is not None:
                    fill_weights_c(connector["conn_w1"], connector["conn_b1"], connector["conn_w2"], connector["conn_b2"])
            else:
                # Single robot weights
                weights_1 = weights["w1"]
                ipt_dim = min(weights1.to_numpy().shape[-1], weights_1.shape[-1])
                fill_weights_ti(robot_id, 0, weights["w1"], weights["b1"], weights["w2"], weights["b2"], ipt_dim)
    else:
        ## Random weights initialization
        print("Initializing weights...", flush=True)
        init_weights()

# =============================================================================
# LOAD PROVIDED WEIGHTS INTO TAICHI FIELDS FOR A SINGLE ROBOT AND SUB-ROBOT
# =============================================================================
# This kernel copies numpy arrays containing NN weights and biases into the
# corresponding Taichi fields for a particular robot (r) and sub-robot (s).
# - w1, b1: weights and biases for first NN layer
# - w2, b2: weights and biases for second NN layer (output layer)
# - ipt_dim: number of input features for the NN (variable per robot)
@ti.kernel
def fill_weights_ti(r: ti.i32, s: ti.i32, w1: ti.types.ndarray(), b1: ti.types.ndarray(), w2: ti.types.ndarray(), b2: ti.types.ndarray(), ipt_dim: ti.i32): # type: ignore
    for i, j in ti.ndrange(n_hidden, ipt_dim):
        weights1[r, s, i, j] = w1[0, i, j] # Copy first layer weights
    for i in range(n_hidden):
        bias1[r, s, i] = b1[0, i] # Copy first layer biases
    for i, j in ti.ndrange(actual_spring_sub_count[r, s], n_hidden):
        weights2[r, s, i, j] = w2[0, i, j] # Copy second layer weights
    for i in range(actual_spring_sub_count[r, s]):
        bias2[r, s, i] = b2[0, i] # Copy second layer biases


# =============================================================================
# LOAD PROVIDED CONNECTOR WEIGHTS INTO TAICHI FIELDS
# =============================================================================
# This kernel copies connector neural network weights and biases into the
# connector-specific Taichi fields. Connectors are shared across multiple sub-robot pairs.
# Assumes connectors were trained on pairs of robots with TARGET_N_SPR springs.
@ti.kernel
def fill_weights_c(w1: ti.types.ndarray(), b1: ti.types.ndarray(), w2: ti.types.ndarray(),
                   b2: ti.types.ndarray()):
    non_connectors = max_springs - (n_sub_bots - 1) * n_conn_springs
    for i, j, k in ti.ndrange(n_hidden, n_sub_bots - 1, n_conn_inputs * n_conn_springs):
        # Copy connector weights block from provided weights to correct location
        conn_weights1[i, non_connectors * n_conn_inputs + j * n_conn_springs * n_conn_inputs + k] = w1[
            i, TARGET_N_SPR * 2 * n_conn_inputs + k] # Assuming connector trained on 2 bots (pairs)
    for i in range(n_hidden):
        conn_bias1[i] = b1[i] # Copy connector biases for first layer
    # Copy second layer weights for connector
    for i, j, l, k in ti.ndrange(1, n_sub_bots - 1, n_conn_springs, n_hidden):
        conn_weights2[i, non_connectors + j * n_conn_springs + l, k] = w2[
            i, TARGET_N_SPR * 2 + l, k]
    # Connector second layer biases
    for i, j, k in ti.ndrange(1, n_sub_bots - 1, n_conn_springs):
        conn_bias2[i, non_connectors + j * n_conn_springs + k] = b2[i, TARGET_N_SPR * 2 + k]

# =============================================================================
# RANDOM INITIALIZATION OF NETWORK WEIGHTS AND BIASES
# =============================================================================
# Xavier normal initialization for all weights and zero initialization for biases.
# Includes weights for robot neural networks and connector neural networks.
@ti.kernel
def init_weights():
    ## Weights initialized with Xavier normal initialization
    ## Bias initialized with zeros
    ti.loop_config(serialize=True)
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            # Xavier normal initialization for first layer weights
            weights1[r, s, i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_input_states_ti(r)))
    ti.loop_config(serialize=True)
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, n_hidden):
        # Initialize first layer biases to zero
        bias1[r, s, i] = 0.0
    ti.loop_config(serialize=True)
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, max_springs, n_hidden):
        if i < n_springs[r]:
            # Xavier normal initialization for second layer weights
            weights2[r, s, i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_springs[r]))
    ti.loop_config(serialize=True)
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, max_springs):
        if i < n_springs[r]:
            # Initialize second layer biases to zero
            bias2[r, s, i] = 0.0

    ti.loop_config(serialize=True)
    for i, j in ti.ndrange(n_hidden, n_conn_inputs * max_springs):
        # Xavier normal initialization for connector first layer weights
        conn_weights1[i, j] = ti.randn() * ti.sqrt(2 / (n_hidden + n_conn_inputs * max_springs))
    ti.loop_config(serialize=True)
    for i in ti.ndrange(n_hidden):
        # Initialize connector first layer biases to zero
        conn_bias1[i] = 0.0
    ti.loop_config(serialize=True)
    for i, j, k in ti.ndrange(1, max_springs, n_hidden):
        # Xavier normal initialization for connector second layer weights
        conn_weights2[i, j, k] = ti.randn() * ti.sqrt(2 / (2 + max_springs + n_hidden))
    ti.loop_config(serialize=True)
    for i, j in ti.ndrange(1, max_springs):
        # Initialize connector second layer biases to zero
        conn_bias2[i, j] = 0.0

# =============================================================================
# PERFORM GRADIENT-BASED WEIGHT UPDATES
# =============================================================================
# Calculates per-robot update scale inversely proportional to gradient root-sum-square norm,
# performs gradient descent weight update on all robot NN weights and biases.
@ti.kernel
def update_weights():
    # Accumulate squared gradient values for first layer weights
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            update_scale[r] += weights1.grad[r, s, i, j] ** 2
    # Accumulate squared gradient values for first layer biases
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, n_hidden):
        update_scale[r] += bias1.grad[r, s, i] ** 2
    # Accumulate squared gradient values for second layer weights
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, max_springs, n_hidden):
        if i < n_springs[r]:
            update_scale[r] += weights2.grad[r, s, i, j] ** 2
    # Accumulate squared gradient values for second layer biases
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, max_springs):
        if i < n_springs[r]:
            update_scale[r] += bias2.grad[r, s, i] ** 2
    # Compute per-robot adaptive learning rate based on gradient norm
    for r in ti.ndrange(n_robots):
        update_scale[r] += gradient_clip / (update_scale[r] ** 0.5 + 1e-6) - update_scale[r]

    # Gradient descent weight and bias update for first layer
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, n_hidden, max_input_states_ti()):
        if j < n_input_states_ti(r):
            weights1[r, s, i, j] -= update_scale[r] * weights1.grad[r, s, i, j]
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, n_hidden):
        bias1[r, s, i] -= update_scale[r] * bias1.grad[r, s, i]
    # Gradient descent update for second layer
    for r, s, i, j in ti.ndrange(n_robots, n_sub_bots, max_springs, n_hidden):
        if i < n_springs[r]:
            weights2[r, s, i, j] -= update_scale[r] * weights2.grad[r, s, i, j]
    for r, s, i in ti.ndrange(n_robots, n_sub_bots, max_springs):
        if i < n_springs[r]:
            bias2[r, s, i] -= update_scale[r] * bias2.grad[r, s, i]

# =============================================================================
# SAVE INDIVIDUAL ROBOT WEIGHTS TO DISK
# =============================================================================
# Saves current NN weights and biases for specified robots to a directory on disk
def save_weights(dir, robot_idx, offset, name):
    w1 = weights1.to_numpy()
    b1 = bias1.to_numpy()
    w2 = weights2.to_numpy()
    b2 = bias2.to_numpy()
    for r in robot_idx:
        if not os.path.exists(os.path.join(dir, str(r+offset))):
            os.makedirs(os.path.join(dir, str(r+offset)))
        with open(os.path.join(dir, str(r+offset), f'{name}.pkl'), "wb") as f:
            pickle.dump({'w1': w1[r], 'b1': b1[r], 'w2': w2[r], 'b2': b2[r]}, f)

# =============================================================================
# SAVE CONNECTOR WEIGHTS TO DISK
# =============================================================================
# Saves neural network weights and biases for connectors to disk
def save_conn_weights(dir, offset, name):
    conn_w1 = conn_weights1.to_numpy()
    conn_b1 = conn_bias1.to_numpy()
    conn_w2 = conn_weights2.to_numpy()
    conn_b2 = conn_bias2.to_numpy()
    if not os.path.exists(os.path.join(dir, str(offset))):
        os.makedirs(os.path.join(dir, str(offset)))
    with open(os.path.join(dir, str(offset), f'{name}.pkl'), "wb") as f:
        pickle.dump({'conn_w1': conn_w1, 'conn_b1': conn_b1, 'conn_w2': conn_w2, 'conn_b2': conn_b2}, f)

# =============================================================================
# RUN FULL FORWARD SIMULATION OF THE ROBOT(S)
# =============================================================================
# Performs the physics and neural net simulation for sim_steps timesteps.
# At each timestep:
#  - Compute center of mass for each sub-robot
#  - Perform NN forward passes (nn1 and nn2)
#  - Apply spring forces based on NN activations
#  - Advance the physical state: update positions, velocities, process collisions
# Optionally saves simulation states (positions, velocities, activations, losses, sensors, ground info) to disk.
def forward(save_state=False, outdir=None):
    global sim_steps
    for t in range(1, sim_steps):
        compute_center(t - 1)  # Compute center of mass at previous timestep
        if comm_enabled:
            comm_signals(t - 1)  # Pooled summaries and inter-level messages
        nn1(t - 1)  # NN layer 1 forward pass
        nn2(t - 1)  # NN layer 2 forward pass to compute spring activations
        apply_spring_force(t - 1)  # Apply forces on springs for timestep t-1
        advance(t)  # Integrate physical motion for timestep t
    compute_loss(sim_steps - 1)  # Compute loss at final timestep

    # Save simulation state data if requested
    if save_state and outdir is not None:
        compute_loss_sub(sim_steps - 1) # Compute per sub-robot loss at final timestep
        os.makedirs(os.path.join(outdir, "state"), exist_ok=True)
        # Save various quantities as NumPy arrays in the state folder
        np.save(os.path.join(outdir, "state", 'loss.npy'), loss.to_numpy())
        np.save(os.path.join(outdir, "state", 'loss_per_bot.npy'), sub_robot_loss.to_numpy())
        np.save(os.path.join(outdir, "state", "v.npy"), v.to_numpy())
        np.save(os.path.join(outdir, "state", "x.npy"), x.to_numpy())
        np.save(os.path.join(outdir, "state", "center.npy"), center.to_numpy())
        np.save(os.path.join(outdir, "state", "act.npy"), act.to_numpy())
        np.save(os.path.join(outdir, "state", "touch.npy"), touch_sensor.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_actuation.npy"), spring_actuation.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_anchor_a"), spring_anchor_a.to_numpy())
        np.save(os.path.join(outdir, "state", "spring_anchor_b"), spring_anchor_b.to_numpy())
        np.save(os.path.join(outdir, "state", "ground_x.npy"), ground_segs_x0.to_numpy())
        np.save(os.path.join(outdir, "state", "ground_y.npy"), ground_segs_y0.to_numpy())
        np.save(os.path.join(outdir, "state", "ground_len.npy"), ground_segs_len.to_numpy())
        np.save(os.path.join(outdir, "state", "ground_slope.npy"), ground_segs_slope.to_numpy())
        np.save(os.path.join(outdir, "state", "ground_shift.npy"), ground_segs_shift.to_numpy())

# =============================================================================
# PERFORM MANUAL BACKWARD PASS TO COMPUTE GRADIENTS OF LOSS WRT NN WEIGHTS
# =============================================================================
# Manually call backward gradient computations for each simulation kernel in reverse order,
# to compute gradients necessary for neural network weight updates using automatic differentiation.
def manual_backward():
    compute_loss.grad(sim_steps - 1)
    for t in range(sim_steps-1, 0, -1):
        advance.grad(t)
        apply_spring_force.grad(t - 1)
        nn2.grad(t - 1)
        nn1.grad(t - 1)
        if comm_enabled:
            comm_signals.grad(t - 1)
        compute_center.grad(t - 1)

# =============================================================================
# CLEAN LOSS VALUES IN HISTORY BY REMOVING NaNs and Infs; TRACK BEST PERFORMERS
# =============================================================================
# Updates loss history, flags invalid losses (NaN, Inf), and records lowest loss per robot.
# If use_mean is True, uses average loss across robots (for connector training).
def clean_loss(loss_hist, k, best_loss, use_mean=False):
    loss_numpy = loss.to_numpy()
    isnan = np.isnan(loss_numpy)
    isinf = np.isinf(loss_numpy)
    invalid_idx = np.unique(np.where(isnan | isinf)[0])
    if use_mean:
        loss_hist[:, k] = loss_numpy.mean() # Use mean loss for connectors
    else:
        loss_hist[:, k] = loss_numpy
    # Find robots with better loss this iteration and exclude invalid indices
    better_idx = np.where(loss_hist[:, k] < best_loss)[0]
    better_idx = np.setdiff1d(better_idx, invalid_idx)    
    best_loss[better_idx] = loss_hist[better_idx, k]
    return loss_hist, best_loss, better_idx

# =============================================================================
# UPDATE CONNECTOR WEIGHTS WITH GRADIENT DESCENT
# =============================================================================
# Similar to update_weights but applies to connector NN.
# Uses a single global learning rate due to shared connector usage.
@ti.kernel
def update_weights_conn():
    # Accumulate squared gradients over connector NN params
    for i, j in ti.ndrange(n_hidden, n_conn_inputs*max_springs):
        conn_update_scale[0] += (conn_weights1.grad[i, j]) ** 2
    for i in ti.ndrange(n_hidden):
        conn_update_scale[0] += (conn_bias1.grad[i]) ** 2
    for i, j, k in ti.ndrange(1, max_springs, n_hidden):
        conn_update_scale[0] += (conn_weights2.grad[i, j, k]) ** 2
    for i, j in ti.ndrange(1, max_springs):
        conn_update_scale[0] += (conn_bias2.grad[i, j]) ** 2
    # Compute global learning rate with gradient clipping
    conn_update_scale[0] += gradient_clip / (conn_update_scale[0] ** 0.5 + 1e-6) - conn_update_scale[0]

    # Apply gradient descent step on connector weights and biases
    for i, j in ti.ndrange( n_hidden, n_conn_inputs*max_springs):
        conn_weights1[i, j] -= conn_update_scale[0] * (conn_weights1.grad[i, j])
    for i in ti.ndrange(n_hidden):
        conn_bias1[i] -= conn_update_scale[0] * (conn_bias1.grad[i])
    for i, j, k in ti.ndrange(1, max_springs, n_hidden):
        conn_weights2[i, j, k] -= conn_update_scale[0] * (conn_weights2.grad[i, j, k])
    for i, j in ti.ndrange(1, max_springs):
        conn_bias2[i, j] -= conn_update_scale[0] * (conn_bias2.grad[i, j])

# =============================================================================
# UPDATE COMMUNICATION WEIGHTS WITH GRADIENT DESCENT
# =============================================================================
# Same normalized-gradient step as update_weights_conn; comm parameters are shared across robots.
@ti.kernel
def update_weights_comm():
    for s, i, k in ti.ndrange(n_sub_bots, n_hidden, 2):
        comm_update_scale[0] += comm_in.grad[s, i, k] ** 2
    for m, k in ti.ndrange(n_comm_mod, 3):
        comm_update_scale[0] += comm_w2.grad[m, k] ** 2
    for m in ti.ndrange(n_comm_mod):
        comm_update_scale[0] += comm_b2.grad[m] ** 2
    for m, k in ti.ndrange(n_comm_mod, 2):
        comm_update_scale[0] += comm_w3.grad[m, k] ** 2
    comm_update_scale[0] += comm_b3.grad[0] ** 2
    comm_update_scale[0] += gradient_clip / (comm_update_scale[0] ** 0.5 + 1e-6) - comm_update_scale[0]

    for s, i, k in ti.ndrange(n_sub_bots, n_hidden, 2):
        comm_in[s, i, k] -= comm_update_scale[0] * comm_in.grad[s, i, k]
    for m, k in ti.ndrange(n_comm_mod, 3):
        comm_w2[m, k] -= comm_update_scale[0] * comm_w2.grad[m, k]
    for m in ti.ndrange(n_comm_mod):
        comm_b2[m] -= comm_update_scale[0] * comm_b2.grad[m]
    for m, k in ti.ndrange(n_comm_mod, 2):
        comm_w3[m, k] -= comm_update_scale[0] * comm_w3.grad[m, k]
    comm_b3[0] -= comm_update_scale[0] * comm_b3.grad[0]

def init_comm(seed=0):
    # Incoming weights start at zero (behavior initially identical to no communication);
    # controller weights are small random so that gradients can reach comm_in.
    rng = np.random.default_rng(seed)
    comm_in.from_numpy(np.zeros((n_sub_bots, n_hidden, 2), dtype=np.float32))
    comm_w2.from_numpy(rng.normal(0, 0.5, (n_comm_mod, 3)).astype(np.float32))
    comm_b2.from_numpy(np.zeros(n_comm_mod, dtype=np.float32))
    comm_w3.from_numpy(rng.normal(0, 0.5, (n_comm_mod, 2)).astype(np.float32))
    comm_b3.from_numpy(np.zeros(1, dtype=np.float32))

# =============================================================================
# TRAIN ROBOTS TO WALK USING LEARNING LOOP
# =============================================================================
# Runs the main training loop, optimizing robot NN weights to minimize loss over multiple iterations.
# Saves weights if loss improves.
def optimize(outdir, idx0=None, idx1=None):
    global learning_iters
    ## Indices of robots to optimize
    idx0 = 0 if idx0 is None else idx0
    idx1 = n_robots if idx1 is None else idx1

    print(f"Optimizing {learning_iters} iterations for {sim_steps} steps...", flush=True)

    weights_dir = os.path.join(outdir, "weights")

    # Initialize loss history and best loss tracker
    loss_hist = np.zeros((n_robots, learning_iters+1))
    best_loss = np.ones(n_robots) * MAX_INT

    ## Save initial weights
    save_weights(weights_dir, np.arange(n_robots), idx0, "init")

    # Main learning loop
    for k in range(learning_iters):
        clear()  # Reset states, gradients, losses
        forward()  # Run forward simulation

        # Save weights if loss improved for any robots
        loss_hist, best_loss, save_idx = clean_loss(loss_hist, k, best_loss)
        if len(save_idx) > 0:
            save_weights(weights_dir, save_idx, idx0, f"best")

        # Set loss gradient for backward pass
        loss.grad.fill(1.0)
        manual_backward()  # Compute gradients by backward differentiation
        update_weights()  # Update NN weights based on gradients

    clear()
    forward()
    loss_hist, _, _ = clean_loss(loss_hist, k+1, best_loss)

    # Save full loss history to disk
    loss_save_path = os.path.join(outdir, f"loss_{idx0}-{idx1}.npy")
    np.save(loss_save_path, loss_hist)

# =============================================================================
# TRAIN CONNECTOR WEIGHTS TO HELP ROBOT COLLECTIVE MOVE COOPERATIVELY
# =============================================================================
# Similar learning loop but focuses on connector NN weights.
# Saves best connector weights during training.
def optimize_conn(weights_dir, idx):
    global learning_iters
    print(f"Optimizing {learning_iters} iterations for {sim_steps} steps...", flush=True)
    loss_hist = np.zeros((n_robots, learning_iters+1))
    best_loss = np.ones(n_robots) * MAX_INT
    # Save initial connector weights
    save_conn_weights(weights_dir, idx, "init")
    ## Learning loop
    for k in range(learning_iters):
        clear()
        forward()
        # Use mean loss across robots for connector training
        loss_hist, best_loss, save_idx = clean_loss(loss_hist, k, best_loss, use_mean=True)
        if len(save_idx) > 0:
            print("saving weights")
            save_conn_weights(weights_dir, idx, "best")
        loss.grad.fill(1.0)
        manual_backward()
        update_weights_conn()
        #print(loss.to_numpy(), best_loss[0], k)
    clear()
    forward()
    loss_hist, _, _ = clean_loss(loss_hist, k + 1, best_loss, use_mean=True)

# =============================================================================
# ENVIRONMENT SETUP FOR SIMULATION CONDITIONS
# =============================================================================
# Configures physics parameters such as friction, gravity, damping, treadmill velocity,
# and other environment-specific modifiers based on condition string.
def env_setup(cond, n_steps, coll_size):
    global friction, wind_factor, damping_y, absorb_factor, rest_factor, \
        gravity, treadmill_vel, base_ground_height, sim_steps, n_sub_bots
    print(f"Setting env {cond}...")

    # Set default environment physics parameters
    gravity = -4.8
    damping_y = 15
    friction = 1.0
    wind_factor = 0.0
    absorb_factor = 1
    rest_factor = 1.0
    treadmill_vel = 0.0

    # Override defaults based on specified environment condition
    if cond == "ice":
        friction = 1.5
    elif cond == "sticky":
        friction = 0.85
    elif cond == "wind":
        wind_factor = 0.04
    elif cond == "sand":
        damping_y = 20
        absorb_factor = 1.25
        rest_factor = 0.9
    elif cond == "low_gravity":
        gravity = -0.25
        damping_y = 5.0
    elif cond == "treadmill":
        treadmill_vel = 0.05
    if n_steps is not None:
        sim_steps = n_steps
    n_sub_bots = coll_size

# =============================================================================
# FULL SIMULATION PIPELINE: LOAD, RUN TRAINING, SAVE RESULTS
# =============================================================================
def simulate(robots_file, outdir, device_id, ground_file, logfile=None, idx0=None, idx1=None, seed=0, debug=False):
    # Redirect stdout and stderr to a log file if specified (for monitoring)
    if logfile is not None:
        if os.path.exists(logfile):
            logfile = logfile.split(".")[0] + "_offspring.log"
        sys.stdout = open(logfile, 'w')
        sys.stderr = sys.stdout
        print(f"Writing stdout,stderr to: {logfile}", flush=True)

    # Set GPU device if specified, else use CPU
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        print(f"Using Cuda device ID: {device_id}", flush=True)
        arch = ti.gpu
    else:
        print("Using CPU", flush=True)
        arch = ti.cpu

    # Initialize Taichi runtime with specified arch, seed, and memory fraction
    ti.init(default_fp=ti.f32, arch=arch, debug=debug, random_seed=seed, device_memory_fraction=0.99)

    # Setup global Taichi fields
    set_ti_globals()

    print(f"Writing losses to: {outdir}", flush=True)
    print(f"Random seed: {seed}", flush=True)

    # Load all robot and environment data into Taichi fields
    setup(robots_file, ground_file, None, idx0, idx1)

    # Run the main robot training optimization loop
    optimize(outdir, idx0, idx1)

# =============================================================================
# RUN SIMULATION FOR VISUALIZATION WITHOUT TRAINING
# =============================================================================
def forward_visualization(robots_file, outdir, ground_file, cond=None, n_steps=None, coll_size=1, conn_file=None):
    # Initialize Taichi with GPU
    ti.init(default_fp=ti.f32, arch=ti.gpu, device_memory_fraction=0.99)

    # Setup Taichi global memory and environment parameters
    set_ti_globals()
    env_setup(cond, n_steps, coll_size)
    # Load robots and optionally connectors into Taichi
    setup(robots_file, ground_file, conn_file)

    print(f"Running forward simulation for visualization...", flush=True)

    # Run simulation forward pass and save state for visualization playback
    forward(save_state=True, outdir=outdir)

    print(f"Simulation state saved to: {outdir}", flush=True)

# =============================================================================
# TRAIN CONNECTORS FOR ROBOT COLLECTIVE COOPERATION
# =============================================================================
def simulate_conn(coll_file, conn_dir, conn_id, n_steps, coll_size, iters, seed=0, debug=False):
    global learning_iters
    # Initialize Taichi runtime with GPU, seed, memory fraction
    ti.init(default_fp=ti.f32, arch=ti.gpu, debug=debug, random_seed=seed, device_memory_fraction=0.99)
    # Setup global fields and environment parameters
    set_ti_globals()
    env_setup(None, n_steps, coll_size)
    learning_iters = iters

    print(f"Random seed: {seed}", flush=True)

    # Load robot collective into Taichi
    setup(coll_file, None)

    # Run connector optimization
    optimize_conn(conn_dir, conn_id)