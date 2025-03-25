import specula

specula.init(device_idx=-1, precision=0)

from specula import xp

print(f"Using {xp.__name__} as backend")

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft, compute_ifs_covmat

# parameters
telescope_diameter = 8.0
obsratio = 0.4
diaratio = 1.0
circGeom = True
angleOffset = 0
doMechCoupling = False
couplingCoeffs = [0.31, 0.05]
doSlaving = True
slavingThr = 0.1
oversampling = 2
n_actuators = 10
r0 = 0.2
L0 = 25.0
zern_modes = 5
pupil_pixels = 256
if_max_condition_number = None
dtype = np.float32

# Generate zonal influence functions
influence_functions, pupil_mask = compute_zonal_ifunc(
    pupil_pixels, n_actuators, circ_geom=circGeom, angle_offset=angleOffset, do_mech_coupling=doMechCoupling,
    coupling_coeffs=couplingCoeffs, do_slaving=doSlaving, slaving_thr=slavingThr,
    obsratio=obsratio, diaratio=diaratio, mask=None, xp=np, dtype=dtype,
    return_coordinates=False
)

# Print shapes for debugging
print(f"Pupil mask shape: {pupil_mask.shape}")
print(f"Pupil mask sum: {np.sum(pupil_mask)}")
print(f"Influence functions shape: {influence_functions.shape}")

# Generate the modal base using the generated inputs
kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
    pupil_mask=pupil_mask,
    diameter=telescope_diameter,
    influence_functions=influence_functions,  # Already in correct shape (Nact, Nmask)
    r0=r0,
    L0=L0,
    zern_modes=zern_modes,
    oversampling=oversampling,
    if_max_condition_number=if_max_condition_number
)

cov_mat = compute_ifs_covmat(pupil_mask, telescope_diameter, kl_basis, r0, L0, oversampling)

# Output the results
print("KL Basis Shape:", kl_basis.shape)
print("Modes-to-Command Matrix Shape:", m2c.shape)

# Plot singular values
plt.figure(figsize=(10, 6))
plt.semilogy(singular_values['S1'], 'o-', label='IF Covariance')
plt.semilogy(singular_values['S2'], 'o-', label='Turbulence Covariance')
plt.xlabel('Mode number')
plt.ylabel('Singular value')
plt.title('Singular values of covariance matrices')
plt.legend()
plt.grid(True)

# Plot some modes
max_modes = kl_basis.shape[0]

# Create a mask array for display
mode_display = np.zeros((max_modes, pupil_mask.shape[0], pupil_mask.shape[1]))

# Place each mode vector into the 2D pupil shape
idx_mask = np.where(pupil_mask)
for i in range(max_modes):
    mode_img = np.zeros(pupil_mask.shape)
    mode_img[idx_mask] = kl_basis[i]
    mode_display[i] = mode_img

# Plot the reshaped modes
n_rows = int(np.round(np.sqrt(max_modes)))
n_cols = int(np.ceil(max_modes / n_rows))
plt.figure(figsize=(18, 12))
for i in range(max_modes):
    plt.subplot(n_rows, n_cols, i+1)
    plt.imshow(mode_display[i], cmap='viridis')
    plt.title(f'Mode {i+1}')
    plt.axis('off')
plt.tight_layout()

# Plot the covariance matrix
plt.figure(figsize=(10, 6))
vmin = np.max(cov_mat) / 1e6  # Set the minimum value to 1e6 times the maximum value of the array
plt.imshow(cov_mat, cmap='viridis', norm=LogNorm(vmin=vmin))
plt.colorbar()


plt.show()