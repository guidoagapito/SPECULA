from scipy.interpolate import Rbf
from specula.lib.make_mask import make_mask

def compute_zonal_ifunc(dim, n_act, xp, dtype, circ_geom=False, angle_offset=0,
                        do_mech_coupling=False, coupling_coeffs=[0.31, 0.05],
                        obsratio=0.0, diaratio=1.0, mask=None):
    """ Computes the ifs_cube matrix with Influence Functions using Thin Plate Splines """

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True, xp=xp)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)[0]

    step = float(dim) / float(n_act)

    # ----------------------------------------------------------
    # Actuator Coordinates
    if circ_geom:
        if n_act % 2 == 0:
            step *= float(n_act) / float(n_act - 1)
            na = xp.arange(round((n_act + 1) / 2)) * 6
        else:
            na = xp.arange(round(n_act / 2)) * 6
        na[0] = 1  # The first value is always 1

        n_act_tot = xp.sum(na)
        pol_coords = xp.zeros((2, n_act_tot))
        ka = 0
        for ia in range(len(na)):
            for ja in range(na[ia]):
                pol_coords[0, ka] = 360. / na[ia] * ja + angle_offset  # Angle in degrees
                pol_coords[1, ka] = ia * step  # Radial distance
                ka += 1

        # System center
        x_c, y_c = dim / 2, dim / 2  

        # Convert from polar to Cartesian coordinates
        x, y = pol_coords[1] * xp.cos(xp.radians(pol_coords[0])), pol_coords[1] * xp.sin(xp.radians(pol_coords[0]))
        x += x_c  # Shift to the center of the grid
        y += y_c  

        # Maximum radius (outer boundary)
        R = pol_coords[1].max()  # The maximum radial value is the outer boundary
    else:
        x, y = xp.meshgrid(xp.linspace(0, dim, n_act), xp.linspace(0, dim, n_act))
        x, y = x.ravel(), y.ravel()
        n_act_tot = n_act ** 2

    coordinates = xp.vstack((x, y))

    # ----------------------------------------------------------
    # Influence Function (ifs_cube) Computation
    ifs_cube = xp.zeros((n_act_tot, dim, dim))

    # Minimum distance between points
    min_distance_norm = max([4/n_act, 0.15])

    for i in range(n_act_tot):
        z = xp.zeros(n_act_tot)

        z[i] = 1.0  # Set the central actuator
        
        if n_act <= 20:
            x_close, y_close, z_close = x, y, z
        else:
            # reduce the set of points to improve performance
            distance = xp.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            
            # Always keep very close points
            min_distance = min_distance_norm * distance.max()
            idx_close_near = xp.where(distance <= min_distance)[0]

            # Randomly select additional points with weighted probability based on distance
            mask_far = distance > min_distance  # Consider only far points
            prob = xp.exp(-distance[mask_far] / (0.3 * distance.max()))  # Exponential decay
            prob /= prob.sum()  # Normalize probability

            if n_act <= 10:
                num_points = (len(distance) // 3)  # Maintain a balanced number of total points
            elif n_act <= 20:
                num_points = (len(distance) // 4)
            else:
                num_points = (len(distance) // 6)
            num_points = max(num_points, 0)  # Avoid negative values

            idx_far = xp.where(mask_far)[0]  # Indices of far points
            idx_close_far = xp.random.choice(idx_far, size=num_points, replace=False, p=prob) if num_points > 0 else xp.array([])

            # Merge selected points
            idx_close = xp.concatenate((idx_close_near, idx_close_far))

            if circGeom:
                idx_border = np.where(np.abs(np.sqrt((x - x_c)**2 + (y - y_c)**2) - R) < step / 2)[0]
                idx_close = np.unique(np.concatenate((idx_close, idx_border)))
            else:
                # Define corner points
                corner_points = np.array([[0, 0], [0, dim-1], [dim-1, 0], [dim-1, dim-1]])

                # Find indices of x and y that correspond to corner points
                corner_indices = []
                for cx, cy in corner_points:
                    indices = np.where((x == cx) & (y == cy))[0]
                    corner_indices.extend(indices)

                corner_indices = np.array(corner_indices)
                idx_close = np.unique(np.concatenate((idx_close, corner_indices)))

                x_close, y_close, z_close = x[idx_close], y[idx_close], z[idx_close]

            x_close, y_close, z_close = x[idx_close], y[idx_close], z[idx_close]

        # Interpolation using Thin Plate Splines
        rbf = Rbf(x_close, y_close, z_close, function='thin_plate')
        grid_x, grid_y = xp.meshgrid(xp.arange(dim), xp.arange(dim))
        z_interp = rbf(grid_x, grid_y)

        ifs_cube[i, :, :] = z_interp

        # Mechanical Coupling
        if do_mech_coupling:
            ifs_cube_orig = ifs_cube.copy()
            for j in range(n_act_tot):
                distance = xp.sqrt((x - x[j])**2 + (y - y[j])**2)

                close1_set = xp.where(distance <= step)[0]
                close2_set = xp.where((distance > step) & (distance <= 2 * step))[0]

                ifs_cube[j, :, :] = ifs_cube_orig[j, :, :]

                if len(close1_set) > 0:
                    for k in close1_set:
                        ifs_cube[j, :, :] += coupling_coeffs[0] * ifs_cube_orig[k, :, :]

                if len(close2_set) > 0:
                    for k in close2_set:
                        ifs_cube[j, :, :] += coupling_coeffs[1] * ifs_cube_orig[k, :, :]

        print(f"\rCompute IFs: {int((i / n_act_tot) * 100)}% done", end="")

    ifs_2d = xp.array([ifs_cube[i][idx] for i in range(n_act_tot)], dtype=dtype)

    print("\nComputation completed.")

    return ifs_2d, mask