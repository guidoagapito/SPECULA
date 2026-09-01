.. _elt_segmented_dm_tutorial:

Building an ELT-Class Segmented Pupil and Modal Basis
========================================================

This tutorial shows how to build, entirely from public parameters and the
calibration tools built into SPECULA, the pupil mask and modal basis for an
Extremely Large Telescope (ELT)-class system whose primary mirror is
supported by several independent structures ("petals"). It is a
self-contained, reusable prerequisite: any tutorial or simulation that
needs an ELT-class segmented pupil and DM (for example the
:ref:`segmented_pupil_soft_limiter_tutorial`) can build on the products
generated here.

**What you'll learn:**

* Why a segmented pupil needs its own influence-function/modal-basis
  generation step, and how to build one from scratch
* Computing a shared pupil mask once and reusing it consistently across
  every influence-function/modal-basis product (and why that matters)
* Generating petal influence functions with :func:`compute_petal_ifunc`
* Generating a zonal DM influence-function set and a KL modal basis with
  :func:`compute_zonal_ifunc` and :func:`make_modal_base_from_ifs_fft`
* Saving everything with the :class:`CalibManager`, ready to be referenced
  by tag from a simulation YAML file

**Prerequisites:**

* SPECULA installed and working (see :doc:`../installation`)
* Basic understanding of adaptive optics concepts (influence functions,
  modal bases, wavefront sensing)
* Python familiarity
* Time and a real machine -- see the warning immediately below before you
  start

.. warning::

   **This is not a quick tutorial to run.** "ELT-class" means a large
   pupil and thousands of actuators, and generating that is genuinely
   expensive -- there is no small/fast version of this that still means
   anything (see the note on modal bandwidth in
   :ref:`elt_petal_mmse_reconstructor_tutorial` for why a shrunk-down
   version actively misleads rather than simplifies). On a single CPU
   core, for the configuration this tutorial uses (400x400 pixels,
   90-actuator grid, 4000 modes), expect the whole pipeline to take on the
   order of **an hour**, with :func:`compute_zonal_ifunc` (Step 3)
   responsible for most of it: it evaluates a thin-plate spline for every
   actuator over every pupil pixel, and both counts are large at this
   scale. This is almost certainly why real ELT-class projects treat this
   as a pre-computed calibration product rather than something regenerated
   on demand. Run it as a background job, and save the result (as this
   tutorial does) so you only pay this cost once.

   If you have a GPU available, passing a GPU index to ``specula.init()``
   (instead of ``-1``) and letting ``specula.xp`` select ``cupy`` cuts
   this down substantially -- every function used in this tutorial accepts
   an ``xp`` module. The exact speedup depends on your hardware, so treat
   "an hour" above as the pessimistic, CPU-only baseline.

   If you just want to confirm the code below runs correctly on your
   machine before committing to this, see
   :ref:`elt_dm_smoke_test` -- a deliberately tiny, explicitly
   non-representative configuration for that purpose only.

Everything used here (pupil mask, influence functions, modal basis) is
generated in Python from public, generic parameters -- nothing is loaded
from an external, project-specific calibration archive.

Why not just use an existing pupil?
------------------------------------

This is a simplified case: generic influence-function generators stand in
for the true, as-designed M1 segment layout and M4 influence functions a
real ELT-class project (e.g. ANDES) would ship. If you have the real pupil
and DM influence functions for your own project, adapting this script to
use them instead is straightforward.

.. note::

   "Segmented" here means six large **petal** sectors, one per independent
   support structure -- not the individual hexagonal M1 segments and their
   gaps, which are irrelevant to the petal-piston control problem this
   tutorial addresses.

Design choices
---------------

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Parameter
     - Value
     - Notes
   * - Telescope diameter
     - 39 m
     - ELT-class
   * - Pupil sampling
     - 400 x 400 pixels
     - matches the resolution used in ELT-class SCAO studies
   * - Central obstruction
     - 28%
     - representative of an ELT-class M2/M5 obstruction
   * - Number of petals
     - 6
     - one per independent primary-mirror support structure
   * - Spider
     - 3 pixels wide
     - matches the real ELT spider thickness at this sampling, see note below
   * - Actuator-to-actuator coupling
     - none (``do_mech_coupling=False``)
     - see note below -- M4 is not a stacked-actuator DM
   * - Edge actuator handling
     - linear (piston+tip+tilt) slaving
     - smoother edge extrapolation than plain weighted-average slaving
   * - Modal basis size
     - generate several thousand, keep 4000
     - a choice made for this tutorial, see note below

.. note::

   **Why no mechanical coupling.** The ``do_mech_coupling`` option in
   :func:`compute_zonal_ifunc` models the nearest/next-nearest-neighbor
   print-through typical of a stacked-piezo DM. An ELT-class M4-type mirror
   has internal metrology that actively imposes the commanded displacement
   on each actuator, eliminating that coupling, so we leave
   ``do_mech_coupling=False`` here.

.. note::

   **Why a 3-pixel spider.** The real ELT primary is split by spiders
   roughly 310 mm thick. At the sampling used here (39 m across 400
   pixels, i.e. ~0.0975 m/pixel), that is close to 3 pixels, so
   ``spider_width=3`` is what a like-for-like ELT model calls for.

.. note::

   **Why exactly 4000 modes.** No fixed rule sets this number -- you can
   use anywhere up to the full count of generated modes. 4000 is simply
   the choice made here (and in the companion
   :ref:`segmented_pupil_soft_limiter_tutorial`); change
   ``n_modes_to_use`` freely if you are adapting this to a different case.

Step 1: A shared pupil mask
-----------------------------

The single most important design decision in this tutorial is to compute
the pupil mask **once**, and pass that exact array to every subsequent
influence-function generator. If instead each generator were left to build
its own mask independently, even a tiny difference in how each one
rasterizes the aperture edge would leave the petal influence functions and
the DM (zonal/KL) influence functions defined over subtly different pixel
sets. Any later step that mixes information from both bases (e.g. a
petal-to-mode reconstruction matrix) would then silently be misaligned --
a very difficult bug to track down after the fact, because every individual
piece still looks reasonable in isolation.

We get the shared mask directly from :func:`compute_petal_ifunc`, since
that call also gives us the petal influence functions we need:

.. code-block:: python

    import specula
    specula.init(-1)  # -1 selects the CPU; use a GPU index if you have one

    import numpy as np
    import matplotlib.pyplot as plt

    from specula.lib.compute_petal_ifunc import compute_petal_ifunc
    from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
    from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
    from specula.data_objects.ifunc import IFunc
    from specula.data_objects.m2c import M2C
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.calib_manager import CalibManager
    from specula import cpuArray

    # --- Physical configuration ---
    telescope_diameter = 39.0    # meters, ELT-class
    obsratio = 0.28              # central obstruction, ELT-class
    n_petals = 6                 # one per M4-like support structure
    angle_offset = 0.0           # degrees

    # --- Resolution / actuator grid -- see the warning above for the cost ---
    pixel_pupil = 400
    n_act = 90

    dtype = specula.xp.float32

    petal_ifunc, pupil_mask, _ = compute_petal_ifunc(
        pixel_pupil, n_petals, xp=specula.xp, dtype=dtype,
        angle_offset=angle_offset, obsratio=obsratio, diaratio=1.0,
        mask=None, spider=True, spider_width=3,
        add_tilts=False, special_last_petal=False)

    print(f'Petal influence functions: {petal_ifunc.shape} '
          f'(6 petals x {petal_ifunc.shape[1]} valid pixels)')
    print(f'Valid pixels in shared mask: {int(specula.xp.sum(pupil_mask))} '
          f'/ {pixel_pupil**2}')

    plt.figure(figsize=(5, 5))
    plt.imshow(cpuArray(pupil_mask), cmap='gray')
    plt.title(f'Shared pupil mask ({n_petals} petals, {pixel_pupil}px)')
    plt.colorbar(label='mask value')
    plt.show()

This step is essentially free (well under a second). With a 28% central
obstruction, expect somewhat more than two thirds of the 400x400 grid to
come out as valid pixels.

``pupil_mask`` is the array we will now pass, unchanged, to every other step.

Step 2: Saving the Pupilstop
-------------------------------

Before moving on, save the pupil as a :class:`Pupilstop` object -- this is
the object a SPECULA simulation YAML file actually references (via
``pupilstop_object``) to define the telescope aperture.

.. code-block:: python

    calib = CalibManager('./calib_elt_segmented_dm_tutorial')

    simul_params = SimulParams(pixel_pupil=pixel_pupil,
                                pixel_pitch=telescope_diameter / pixel_pupil)

    pupilstop_obj = Pupilstop(simul_params, input_mask=cpuArray(pupil_mask))
    pupilstop_filename = calib.filename('pupilstop', 'ELT39_6petals')
    pupilstop_obj.save(pupilstop_filename, overwrite=True)
    print(f'Saved: {pupilstop_filename}')

    petal_ifunc_obj = IFunc(ifunc=petal_ifunc, mask=pupil_mask)
    petal_ifunc_filename = calib.filename('ifunc', 'ELT39_6petals')
    petal_ifunc_obj.save(petal_ifunc_filename, overwrite=True)
    print(f'Saved: {petal_ifunc_filename}')

Using :class:`CalibManager` (rather than hand-built paths) means these
products can be referenced later purely by tag, exactly the way a
simulation YAML file resolves ``ifunc_object`` or ``pupilstop_object`` tags
under a shared ``root_dir`` -- see :ref:`calibration_manager` if this is new
to you.

A differential petal inverse, for ground-truth diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A closed-loop simulation typically wants a ground-truth readout of the
current petal-piston state directly from the electric field (e.g. via
:class:`ModalAnalysis`), independent of the MMSE reconstructor built in
:ref:`elt_petal_mmse_reconstructor_tutorial`. That needs an inverse of the
petal basis -- and it is easy to get wrong in a way that still looks
plausible at first glance:

.. code-block:: python

    # 5 rows only (petal 6 stays the reference), NOT all 6:
    petal_ifunc_5_obj = IFunc(ifunc=petal_ifunc[:n_petals - 1, :], mask=pupil_mask)
    petal_inv_obj = petal_ifunc_5_obj.inverse()   # remove_piston=True by default
    petal_inv_filename = calib.filename('ifunc', 'ELT39_6petals_inv')
    petal_inv_obj.save(petal_inv_filename, overwrite=True)
    print(f'Saved: {petal_inv_filename}')

.. warning::

   **The obvious-looking alternative is wrong.** It is tempting to invert
   all *6* raw petal rows at once and keep only the first 5 columns of the
   result (e.g. via the ``nmodes=5`` parameter of :class:`ModalAnalysis`,
   which just slices columns). That gives, for each petal, the *absolute* mean phase over its
   own pixels -- not the *differential* piston relative to petal 6 -- so
   it tracks any true global piston in the wavefront almost one-for-one
   (measured: an injected 2000 nm global offset leaked out at slope 0.99,
   with a resulting RMS error of ~2000 nm on an 800 nm-scale signal). It
   looks fine in isolation because a global piston is usually small or
   absent in a synthetic test -- it only shows up once a real closed loop
   with genuine low-order residual is running.

   Dropping the sixth row *before* inverting, as done above, is not just a
   truncation of the same computation: the default ``remove_piston=True``
   of :func:`IFunc.inverse` centers each of the 5 remaining rows (subtracting
   its own mean), which makes each of them exactly orthogonal to the
   uniform/global-piston pattern -- a standard regression identity
   (centering the regressors makes the fitted coefficients invariant to a
   constant offset in the response). Verified numerically: with the fix,
   the same 2000 nm injected global offset leaks out at slope ~0.01 (i.e.
   essentially not at all), with sub-nanometer RMS error on the same
   800 nm-scale signal.

Step 3: Zonal DM influence functions, on the *same* mask
------------------------------------------------------------

Now we generate the deformable-mirror influence functions -- a much larger
set, one per actuator, used as the basis for the KL modal decomposition in
Step 4. Note that ``mask=pupil_mask`` reuses the exact array from Step 1,
rather than letting :func:`compute_zonal_ifunc` build its own. **This is
the expensive step -- see the warning at the top of this page.**

.. code-block:: python

    zonal_ifunc, pupil_mask_check, _, _ = compute_zonal_ifunc(
        pixel_pupil, n_act, xp=specula.xp, dtype=dtype,
        circ_geom=True, angle_offset=angle_offset,
        do_mech_coupling=False,
        do_slaving=True, slaving_thr=0.1, linear_slaving=True,
        obsratio=obsratio, diaratio=1.0, mask=pupil_mask)

    assert np.array_equal(cpuArray(pupil_mask_check), cpuArray(pupil_mask)), \
        "mask mismatch -- the zonal ifunc is not on the same pixel grid!"

    print(f'Zonal influence functions: {zonal_ifunc.shape[0]} valid actuators '
          f'(after slaving) x {zonal_ifunc.shape[1]} pixels')

We leave mechanical coupling disabled (see the design-choices note above)
and enable ``do_slaving`` with ``linear_slaving=True``: rather than simply
averaging neighboring master actuators, each weakly-coupled edge actuator
is extrapolated from a local piston+tip+tilt plane fit to nearby masters --
a smoother, more physically reasonable edge behavior than a flat weighted
average. The explicit ``assert`` is not decorative: it is exactly the
check that would have caught the shared-mask problem described above, had
one crept in.

This is the slow step (most of the "about an hour" from the warning at the
top). With ``n_act=90``, expect several thousand valid actuators after
slaving -- a small fraction of them slaved at the edge, the rest masters.

Step 4: A KL modal basis, generating more modes than we need
------------------------------------------------------------------

:func:`make_modal_base_from_ifs_fft` turns the zonal influence functions
into a Karhunen-Loeve-like modal basis, ranked by the turbulence power they
capture:

.. code-block:: python

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=pupil_mask, diameter=telescope_diameter,
        influence_functions=zonal_ifunc,
        r0=0.15, L0=25.0,          # only used to *weight* the basis
        zern_modes=3, oversampling=2,
        if_max_condition_number=None,
        xp=specula.xp, dtype=dtype)

    print(f'KL basis: {kl_basis.shape[0]} modes x {kl_basis.shape[1]} pixels')

``r0``/``L0`` here only shape which spatial frequencies the basis
prioritizes; they are independent of whatever atmosphere you eventually
simulate in closed loop.

This is comparatively cheap next to Step 3 -- a few minutes rather than
most of an hour -- and produces one mode fewer than the actuator count
(the global piston is not a controllable DM mode).

You can use anywhere up to the full count of generated modes -- keep as
many as you need without re-running Step 3:

.. code-block:: python

    n_modes_to_use = min(4000, kl_basis.shape[0])  # see note on this choice above
    kl_basis = kl_basis[:n_modes_to_use]
    m2c = m2c[:, :n_modes_to_use]
    print(f'Using the first {n_modes_to_use} of {singular_values["S1"].shape[0]} '
          f'generated modes')

With ``n_act=90`` there are more than 4000 modes available, so this line
does real work, keeping the first 4000 and dropping the rest.

Saving the modal basis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m2c_obj = M2C(m2c=m2c)
    m2c_filename = calib.filename('m2c', 'ELT39_KL4000')
    m2c_obj.save(m2c_filename, overwrite=True)
    print(f'Saved: {m2c_filename}')

    zonal_ifunc_obj = IFunc(ifunc=zonal_ifunc, mask=pupil_mask)
    zonal_ifunc_filename = calib.filename('ifunc', 'ELT39_zonal')
    zonal_ifunc_obj.save(zonal_ifunc_filename, overwrite=True)
    print(f'Saved: {zonal_ifunc_filename}')

    # Also save the forward KL basis itself as an IFunc (not just its
    # inverse): building a petal<->KL reconstructor in the next tutorial
    # needs the forward basis to compute a turbulence covariance matrix in
    # mode space.
    kl_ifunc_obj = IFunc(ifunc=cpuArray(kl_basis), mask=cpuArray(pupil_mask))
    kl_ifunc_filename = calib.filename('ifunc', 'ELT39_KL4000')
    kl_ifunc_obj.save(kl_ifunc_filename, overwrite=True)
    print(f'Saved: {kl_ifunc_filename}')

    # IFunc.inverse() computes the pseudo-inverse and hands back a
    # ready-made IFuncInv.
    ifunc_inv_obj = kl_ifunc_obj.inverse()
    ifunc_inv_filename = calib.filename('ifunc', 'ELT39_KL4000_inv')
    ifunc_inv_obj.save(ifunc_inv_filename, overwrite=True)
    print(f'Saved: {ifunc_inv_filename}')

This last step -- a pseudo-inverse of a (4000 modes, many pixels) matrix --
adds a further few minutes. ``IFunc.inverse()`` computes it via the smaller
of the two Gram matrices (:func:`specula.lib.fast_pinv.fast_pinv`) rather
than an SVD of the full rectangular matrix, which is what makes this
tractable at all at this scale; a direct ``xp.linalg.pinv`` on the full
matrix would take substantially longer.

Sanity checks and visualization
----------------------------------

A couple of cheap checks are worth running every time you regenerate a
basis, since a silent mistake here (e.g. the shared-mask issue discussed
above) would otherwise only surface much later, in a confusing way, deep
inside a closed-loop simulation:

.. code-block:: python

    # Mode RMS should be well-behaved (non-zero, no NaNs) for every mode
    rms = np.sqrt(np.mean(cpuArray(kl_basis)**2, axis=1))
    print(f'Mode RMS: min={rms.min():.3g}, max={rms.max():.3g}, '
          f'any NaN: {np.any(np.isnan(rms))}')

    # Singular value spectrum: should decay smoothly, no discontinuities
    plt.figure(figsize=(8, 5))
    plt.semilogy(cpuArray(singular_values['S1']), 'o-', label='IF covariance')
    plt.semilogy(cpuArray(singular_values['S2']), 'o-', label='Turbulence covariance')
    plt.xlabel('Mode number')
    plt.ylabel('Singular value')
    plt.legend()
    plt.grid(True)
    plt.title('Singular value spectrum')
    plt.show()

    # A handful of KL modes, reshaped onto the pupil, for a visual check
    kl_np = cpuArray(kl_basis)
    mask_np = cpuArray(pupil_mask)
    idx_mask = np.where(mask_np)
    n_show = min(9, kl_np.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat[:n_show]):
        mode_img = np.zeros(mask_np.shape)
        mode_img[idx_mask] = kl_np[i]
        ax.imshow(mode_img, cmap='viridis')
        ax.set_title(f'Mode {i + 1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

.. _elt_dm_smoke_test:

Smoke-testing the pipeline before committing to the full run
------------------------------------------------------------------

If you only want to check that the code above runs on your machine --
no typos, no missing dependencies, no shape mismatches -- before spending
close to an hour on it, shrink the two size parameters drastically:

.. code-block:: python

    pixel_pupil = 40
    n_act = 9

This finishes in a few seconds. **Do not draw any conclusion from the
numbers it produces** -- valid-pixel counts, actuator counts, mode counts,
and (if you carry it into :ref:`elt_petal_mmse_reconstructor_tutorial`)
any reconstruction accuracy are all specific to this toy size and do not
scale down meaningfully from the full configuration. Its only job is to
confirm the pipeline executes; once it does, switch back to
``pixel_pupil = 400`` and ``n_act = 90`` and let the full run complete.

Summary and what's next
--------------------------

At this point you have, generated entirely from public parameters and
saved through the :class:`CalibManager`:

* a shared pupil mask (400x400 pixels, 39 m, 28% obstruction, 6 petals),
  saved as a :class:`Pupilstop`
* petal influence functions (:class:`IFunc`, 6 petals) on that mask, and a
  properly differential 5-petal :class:`IFuncInv` for ground-truth
  diagnostics
* a zonal DM influence-function set (:class:`IFunc`, several thousand
  actuators) on the *same* mask
* a 4000-mode KL modal basis, saved both as an :class:`M2C` and directly as
  an :class:`IFunc` (the forward basis), plus the corresponding
  :class:`IFuncInv` (its pseudo-inverse) -- all on the same shared mask

These products are the starting point for
:ref:`elt_petal_mmse_reconstructor_tutorial`, which uses the shared mask to
build a self-consistent reconstructor from KL-mode commands to petal-piston
estimates -- and, further down the line, for the closed-loop simulation in
:ref:`segmented_pupil_soft_limiter_tutorial`.
