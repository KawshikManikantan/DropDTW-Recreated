import numpy as np

def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def dtw(dist):
    "Classical DTW algorithm"

    nrows, ncols = dist.shape
    dtw = np.zeros((nrows + 1,ncols + 1), dtype=np.float32)
    # get dtw table
    for i in range(0, nrows + 1):
        for j in range(0, ncols + 1):
            if (i == 0) and (j == 0):
                new_val = 0.0
                dtw[i, j] = new_val
            elif (i == 0) and (j != 0):
                new_val = np.inf
                dtw[i, j] = new_val
            elif (i != 0) and (j == 0):
                new_val = np.inf
                dtw[i, j] = new_val
            else:
                neighbors = [dtw[i, j - 1], dtw[i - 1, j - 1], dtw[i - 1, j]]
                new_val = dist[i - 1, j - 1] + min(neighbors)
                dtw[i, j] = new_val
    # get alignment path
    path = traceback(dtw)
    return dtw, path

def drop_dtw(zx_costs, drop_costs, exclusive=True, contiguous=True, return_labels=False):
    """Drop-DTW algorithm that allows drop only from one (video) side. See Algorithm 1 in the paper.

    Parameters
    ----------
    zx_costs: np.ndarray [K, N] 
        pairwise match costs between K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    exclusive: bool
        If True any clip can be matched with only one step, not many.
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    return_label: bool
        if True, returns output directly useful for segmentation computation (made for convenience)
    """
    
    K, N = zx_costs.shape
    
    # initialize solution matrices
    D = np.zeros([K + 1, N + 1, 2]) # the 2 last dimensions correspond to different states.
                                    # State (dim) 0 - x is matched; State 1 - x is dropped
    D[1:, 0, :] = np.inf  # no drops in z in any state
    D[0, 1:, 0] = np.inf  # no drops in x in state 0, i.e. state where x is matched
    D[0, 1:, 1] = np.cumsum(drop_costs)  # drop costs initizlization in state 1

    # initialize path tracking info for each state
    P = np.zeros([K + 1, N + 1, 2, 3], dtype=int) 
    for xi in range(1, N + 1):
        P[0, xi, 1] = 0, xi - 1, 1
    
    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1] 
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states] ## [((z-1),(x-1)),((z-1),(x-1))] -- indices
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states] ## [D[((z-1),(x-1)),0],D[((z-1),(x-1)),0]]

            left_neigh_states = [0, 1]
            left_neigh_coords = [(zi, xi - 1) for _ in left_neigh_states]
            left_neigh_costs = [D[zi, xi - 1, s] for s in left_neigh_states]

            left_pos_neigh_states = [0] if contiguous else left_neigh_states
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]

            top_pos_neigh_states = [0]
            top_pos_neigh_coords = [(zi - 1, xi) for _ in left_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in left_pos_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state 0: matching x to z
            if exclusive:
                neigh_states_pos = diag_neigh_states + left_pos_neigh_states
                neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords
                neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs
            else:
                neigh_states_pos = diag_neigh_states + left_pos_neigh_states + top_pos_neigh_states
                neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords + top_pos_neigh_coords
                neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs + top_pos_neigh_costs
            costs_pos = np.array(neigh_costs_pos) + zx_costs[z_cost_ind, x_cost_ind] 
            opt_ind_pos = np.argmin(costs_pos)
            P[zi, xi, 0] = *neigh_coords_pos[opt_ind_pos], neigh_states_pos[opt_ind_pos]
            D[zi, xi, 0] = costs_pos[opt_ind_pos]

            # state 1: x is dropped
            costs_neg = np.array(left_neigh_costs) + drop_costs[x_cost_ind] 
            opt_ind_neg = np.argmin(costs_neg)
            P[zi, xi, 1] = *left_neigh_coords[opt_ind_neg], left_neigh_states[opt_ind_neg]
            D[zi, xi, 1] = costs_neg[opt_ind_neg]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]
            
    # backtracking the solution
    zi, xi = K, N
    path, labels = [], np.zeros(N)
    x_dropped = [] if cur_state == 1 else [N]
    while not (zi == 0 and xi == 0):
        path.append((zi, xi,cur_state))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if xi > 0:
            labels[xi - 1] = zi * (cur_state == 0)  # either zi or 0
        if prev_state == 1:
            x_dropped.append(xi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state
    
    if not return_labels:
        return min_cost, path, x_dropped
    else:
        return labels

def double_drop_dtw(
    pairwise_zx_costs,
    x_drop_costs,
    z_drop_costs,
    contiguous=True,
    one_to_many=True,
    many_to_one=True,
    return_labels=False,
):
    """Drop-DTW algorithm that allows drops from both sequences. See Algorithm 1 in Appendix.

    Parameters
    ----------
    pairwise_zx_costs: np.ndarray [K, N]
        pairwise match costs between K steps and N video clips
    x_drop_costs: np.ndarray [N]
        drop costs for each clip
    z_drop_costs: np.ndarray [N]
        drop costs for each step
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    """
    K, N = pairwise_zx_costs.shape

    # initialize solution matrices
    D = np.zeros([K + 1, N + 1, 4])  # the 4 dimensions are the following states: zx, z-, -x, --
    # no drops allowed in zx DP. Setting the same for all DPs to change later here.
    D[1:, 0, :] = 99999999
    D[0, 1:, :] = 99999999
    D[0, 0, 1:] = 99999999
    # Allow to drop x in z- and --
    D[0, 1:, 1], D[0, 1:, 3] = np.cumsum(x_drop_costs), np.cumsum(x_drop_costs)
    # Allow to drop z in -x and --
    D[1:, 0, 2], D[1:, 0, 3] = np.cumsum(z_drop_costs), np.cumsum(z_drop_costs)

    # initialize path tracking info for each of the 4 DP tables:
    P = np.zeros([K + 1, N + 1, 4, 3], dtype=int)  # (zi, xi, prev_state)
    for zi in range(1, K + 1):
        P[zi, 0, 2], P[zi, 0, 3] = (zi - 1, 0, 2), (zi - 1, 0, 3)
    for xi in range(1, N + 1):
        P[0, xi, 1], P[0, xi, 3] = (0, xi - 1, 1), (0, xi - 1, 3)

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1, 2, 3]  # zx, z-, -x, --
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]

            left_pos_neigh_states = [0, 1]  # zx and z-
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]

            top_pos_neigh_states = [0, 2]  # zx and -x
            top_pos_neigh_coords = [(zi - 1, xi) for _ in top_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in top_pos_neigh_states]

            left_neg_neigh_states = [2, 3]  # -x and --
            left_neg_neigh_coords = [(zi, xi - 1) for _ in left_neg_neigh_states]
            left_neg_neigh_costs = [D[zi, xi - 1, s] for s in left_neg_neigh_states]

            top_neg_neigh_states = [1, 3]  # z- and --
            top_neg_neigh_coords = [(zi - 1, xi) for _ in top_neg_neigh_states]
            top_neg_neigh_costs = [D[zi - 1, xi, s] for s in top_neg_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # DP 0: coming to zx
            neigh_states_zx = diag_neigh_states
            neigh_coords_zx = diag_neigh_coords
            neigh_costs_zx = diag_neigh_costs
            if one_to_many:
                if contiguous:
                    neigh_states_zx.extend(left_pos_neigh_states[0:1])
                    neigh_coords_zx.extend(left_pos_neigh_coords[0:1])
                    neigh_costs_zx.extend(left_pos_neigh_costs[0:1])
                else:
                    neigh_states_zx.extend(left_pos_neigh_states)
                    neigh_coords_zx.extend(left_pos_neigh_coords)
                    neigh_costs_zx.extend(left_pos_neigh_costs)
            if many_to_one:
                neigh_states_zx.extend(top_pos_neigh_states)
                neigh_coords_zx.extend(top_pos_neigh_coords)
                neigh_costs_zx.extend(top_pos_neigh_costs)

            costs_zx = np.array(neigh_costs_zx) + pairwise_zx_costs[z_cost_ind, x_cost_ind]
            opt_ind_zx = np.argmin(costs_zx)
            P[zi, xi, 0] = *neigh_coords_zx[opt_ind_zx], neigh_states_zx[opt_ind_zx]
            D[zi, xi, 0] = costs_zx[opt_ind_zx]

            # DP 1: coming to z-
            neigh_states_z_ = left_pos_neigh_states
            neigh_coords_z_ = left_pos_neigh_coords
            neigh_costs_z_ = left_pos_neigh_costs
            costs_z_ = np.array(neigh_costs_z_) + x_drop_costs[x_cost_ind]
            opt_ind_z_ = np.argmin(costs_z_)
            P[zi, xi, 1] = *neigh_coords_z_[opt_ind_z_], neigh_states_z_[opt_ind_z_]
            D[zi, xi, 1] = costs_z_[opt_ind_z_]

            # DP 2: coming to -x
            neigh_states__x = top_pos_neigh_states
            neigh_coords__x = top_pos_neigh_coords
            neigh_costs__x = top_pos_neigh_costs
            costs__x = np.array(neigh_costs__x) + z_drop_costs[z_cost_ind]
            opt_ind__x = np.argmin(costs__x)
            P[zi, xi, 2] = *neigh_coords__x[opt_ind__x], neigh_states__x[opt_ind__x]
            D[zi, xi, 2] = costs__x[opt_ind__x]

            # DP 3: coming to --
            neigh_states___ = np.array(left_neg_neigh_states + top_neg_neigh_states)
            # adding negative left and top neighbors
            neigh_coords___ = np.array(left_neg_neigh_coords + top_neg_neigh_coords)
            costs___ = np.concatenate(
                [
                    left_neg_neigh_costs + x_drop_costs[x_cost_ind],
                    top_neg_neigh_costs + z_drop_costs[z_cost_ind],
                ],
                0,
            )

            opt_ind___ = costs___.argmin()
            P[zi, xi, 3] = *neigh_coords___[opt_ind___], neigh_states___[opt_ind___]
            D[zi, xi, 3] = costs___[opt_ind___]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]

    # unroll path
    path = []
    zi, xi = K, N
    x_dropped = [N] if cur_state in [1, 3] else []
    z_dropped = [K] if cur_state in [2, 3] else []
    while not (zi == 0 and xi == 0):
        path.append((zi, xi))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if prev_state in [1, 3]:
            x_dropped.append(xi_prev)
        if prev_state in [2, 3]:
            z_dropped.append(zi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state

    if return_labels:
        labels = np.zeros(N)
        for zi, xi in path:
            if zi not in z_dropped and xi not in x_dropped:
                labels[xi - 1] = zi
        return labels
    else:
        return min_cost, path, x_dropped, z_dropped
