"""
MODERATION EFFECT IN A PWR FUEL ASSEMBLY  —  OpenMC sweep + 4-panel animation
=============================================================================PHYSICS
-------
The fuel volume is held CONSTANT while the moderator (water) is varied in either
  * volume  (MODE = "pitch"   : pin pitch changes, fuel radius fixed)   or
  * density (MODE = "density" : pitch fixed, water density changes)
so that under-moderated / optimally moderated / over-moderated conditions can be
compared.

Quantities computed and stored at every sweep point:
  - k_inf (with statistical uncertainty)
  - 2-D mesh neutron flux maps in 3 groups (thermal / epithermal / fast)
  - 2-D fission reaction-rate map
  - assembly-averaged neutron spectrum (flux per unit lethargy)
  - thermal utilization f, thermal fission fraction, epithermal capture fraction
  - N_H/N_U (hydrogen-to-uranium atom ratio) and V_m/V_f
REQUIREMENTS
------------
numpy + matplotlib always; openmc only for a real sweep (not needed when
DEMO = True); ffmpeg for MP4 output (it falls back to GIF automatically).
"""

import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np


# ###########################################################################
class CFG:
    # ---- what to do -------------------------------------------------------
    demo = False           # True -> OpenMC is never called; the layout is
    #                        previewed with synthetic data
    run_sweep = True       # False -> skip the sweep and read the existing npz
    make_video = True      # False -> sweep only, no animation

    # ---- sweep (OpenMC) ---------------------------------------------------
    mode = "pitch"         # "pitch" (water VOLUME varies) | "density" (water DENSITY)
    points = 17            # number of sweep points
    pitch_min = 1.00       # cm    (mode = "pitch")
    pitch_max = 2.60       # cm    (mode = "pitch")
    dens_min = 0.05        # g/cm3 (mode = "density")
    dens_max = 1.00        # g/cm3 (mode = "density")
    water_density = 0.74   # g/cm3, fixed water density in "pitch" mode
    pitch = 1.26           # cm,    fixed pin pitch in "density" mode

    enrichment = 3.1       # % U-235
    boron_ppm = 0.0        # soluble boron [weight ppm]
    lattice = 17           # NxN lattice
    holes = None           # what sits at the 25 guide-tube positions:
    mesh_per_pin = 4       # mesh cells per pin, per direction (>=4 recommended)
    symmetrize = True      # average the maps over the 8 square symmetries

    particles = 12000
    batches = 70
    inactive = 20
    threads = None         # None -> OpenMC default (all cores)
    hot = False            # True -> hot conditions (fuel 900K, clad 600K, water 580K)

    quick = False          # fast smoke test (fewer particles/points)
    hq = False             # high quality maps (roughly 4x the runtime)

    keep_runs = False      # keep the OpenMC run folders
    verbose = False        # show the OpenMC output
    workdir = "openmc_runs"

    # Cross-section library. The OPENMC_CROSS_SECTIONS environment variable wins
    # if it is set; otherwise the path below is used.
    cross_sections = os.environ.get(
        "OPENMC_CROSS_SECTIONS",
        "/home/emil/openmc/Cross_Section_Libraries/"
        "endfb-vii.1-hdf5/cross_sections.xml",
    )

    # ---- output files -----------------------------------------------------
    npz_out = None         # None -> "results_<mode>.npz"
    video_out = None       # None -> "moderation_<mode>.mp4"

    # ---- animation --------------------------------------------------------
    fps = 15
    interp = 5             # intermediate frames between sweep points (1 = none)
    hold = 10              # hold frames at the start / optimum / end points
    pingpong = True        # loop forward and back
    scale = "global"       # colour scale: "global" (shared) | "frame" (per frame)
    gif = False            # also write a GIF next to the MP4
    dpi = 100              # 100 -> 1920x1080


# ===========================================================================
#  SHARED CONSTANTS  (17x17 Westinghouse-type PWR fuel assembly)
# ===========================================================================
R_FUEL = 0.4096      # cm, fuel pellet radius              (HELD CONSTANT)
R_GAP = 0.4180       # cm, outer radius of pellet-clad gap
R_CLAD = 0.4750      # cm, clad outer radius
R_GT_IN = 0.5610     # cm, guide tube inner radius
R_GT_OUT = 0.6020    # cm, guide tube outer radius
PITCH_NOM = 1.26     # cm, nominal PWR pin pitch
E_THERMAL = 0.625    # eV, thermal group upper bound
E_FAST = 1.0e5       # eV, epithermal/fast group boundary
E_MAX = 2.0e7        # eV
E_MIN = 1.0e-5       # eV

# Molar masses / densities
RHO_UO2 = 10.3070    # g/cm3
M_UO2 = 270.03       # g/mol  (approximate: U235/U238 mix + 2*O16)
M_H2O = 18.0153      # g/mol
N_AVO = 6.02214076e23

# Guide-tube and instrument-tube locations in a 17x17 assembly (0-based row, col)
GUIDE_TUBE_POSITIONS = [
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13),
    (14, 5), (14, 8), (14, 11),
]


def hole_positions(n_lat, holes):
    """Lattice positions that carry a guide tube / water hole instead of a pin."""
    if holes == "none":
        return []
    return [(i, j) for (i, j) in GUIDE_TUBE_POSITIONS if i < n_lat and j < n_lat]


# ===========================================================================
#                          PART 1 — OpenMC PARAMETER SWEEP
# ===========================================================================
def _import_openmc(xs_path):
    
    os.environ["OPENMC_CROSS_SECTIONS"] = xs_path
    if not os.path.exists(xs_path):
        sys.exit("ERROR: cross-section file not found:\n  %s\n"
                 "Fix CFG.cross_sections in the SETTINGS block, or set the "
                 "OPENMC_CROSS_SECTIONS environment variable." % xs_path)
    try:
        import openmc                                    # noqa: E401
    except ImportError as exc:
        sys.exit("ERROR: OpenMC is not importable (%s).\n"
                 "It is needed for a real sweep; set CFG.demo = True in the "
                 "SETTINGS block to preview the animation layout only." % exc)
    return openmc


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def build_materials(openmc, enrichment, water_density, boron_ppm, hot):
    """Create the UO2 / He / zirconium / water materials."""
    t_fuel, t_clad, t_water = (900.0, 600.0, 580.0) if hot else (293.6, 293.6, 293.6)

    uo2 = openmc.Material(name="UO2 fuel")
    uo2.set_density("g/cm3", RHO_UO2)
    uo2.add_element("U", 1.0, enrichment=enrichment)
    uo2.add_nuclide("O16", 2.0)
    uo2.temperature = t_fuel

    helium = openmc.Material(name="Helium gap")
    helium.set_density("g/cm3", 0.001598)
    helium.add_nuclide("He4", 1.0)
    helium.temperature = t_clad

    # Pure natural Zr instead of Zircaloy: Zr90-96 are present in ENDF/B-VII.1,
    # while adding Sn/Cr/Fe increases library dependence for negligible
    # neutronic benefit in this study.
    clad = openmc.Material(name="Zirconium clad")
    clad.set_density("g/cm3", 6.55)
    clad.add_element("Zr", 1.0)
    clad.temperature = t_clad

    water = openmc.Material(name="Water (moderator)")
    water.set_density("g/cm3", water_density)
    water.add_nuclide("H1", 2.0)
    water.add_nuclide("O16", 1.0)
    if boron_ppm > 0.0:
        # soluble boron in weight ppm -> B atoms per H2O molecule
        wt = boron_ppm * 1.0e-6
        water.add_element("B", wt * M_H2O / 10.811 / (1.0 - wt), "ao")
    water.add_s_alpha_beta("c_H_in_H2O")
    water.temperature = t_water

    return uo2, helium, clad, water


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def build_geometry(openmc, pitch, n_lat, uo2, helium, clad, water, holes):
    r_f = openmc.ZCylinder(r=R_FUEL)
    r_g = openmc.ZCylinder(r=R_GAP)
    r_c = openmc.ZCylinder(r=R_CLAD)

    # --- fuel pin universe ---
    c_fuel = openmc.Cell(name="fuel", fill=uo2, region=-r_f)
    c_gap = openmc.Cell(name="gap", fill=helium, region=+r_f & -r_g)
    c_clad = openmc.Cell(name="clad", fill=clad, region=+r_g & -r_c)
    c_mod = openmc.Cell(name="moderator", fill=water, region=+r_c)
    pin = openmc.Universe(name="fuel pin", cells=[c_fuel, c_gap, c_clad, c_mod])

    # --- pure water universe (lattice outer) ---
    water_univ = openmc.Universe(
        name="water", cells=[openmc.Cell(name="water cell", fill=water)]
    )

    # --- guide tube / water hole universe ---
    if holes == "tube":
        gi = openmc.ZCylinder(r=R_GT_IN)
        go = openmc.ZCylinder(r=R_GT_OUT)
        gt = openmc.Universe(
            name="guide tube",
            cells=[
                openmc.Cell(fill=water, region=-gi),
                openmc.Cell(fill=clad, region=+gi & -go),
                openmc.Cell(fill=water, region=+go),
            ],
        )
    elif holes == "water":
        gt = water_univ
    else:
        gt = None

    # --- lattice ---
    lattice = openmc.RectLattice(name="assembly lattice")
    lattice.pitch = (pitch, pitch)
    lattice.lower_left = (-n_lat * pitch / 2.0, -n_lat * pitch / 2.0)
    univ_map = np.full((n_lat, n_lat), pin, dtype=object)
    for (i, j) in hole_positions(n_lat, holes):
        univ_map[i, j] = gt
    lattice.universes = univ_map
    lattice.outer = water_univ

    # --- root cell: 4 reflective planes (infinite in z = 2-D model) ---
    half = n_lat * pitch / 2.0
    xmin = openmc.XPlane(x0=-half, boundary_type="reflective")
    xmax = openmc.XPlane(x0=+half, boundary_type="reflective")
    ymin = openmc.YPlane(y0=-half, boundary_type="reflective")
    ymax = openmc.YPlane(y0=+half, boundary_type="reflective")
    root_cell = openmc.Cell(
        name="assembly", fill=lattice, region=+xmin & -xmax & +ymin & -ymax
    )
    root = openmc.Universe(name="root", cells=[root_cell])
    return openmc.Geometry(root), half


# ---------------------------------------------------------------------------
# Tallies energy filter fission meshing
# ---------------------------------------------------------------------------
def build_tallies(openmc, half, n_mesh, uo2, mesh_h=1.0, n_spec=500):
    tallies = openmc.Tallies()

    mesh = openmc.RegularMesh()
    mesh.dimension = (n_mesh, n_mesh, 1)
    mesh.lower_left = (-half, -half, -mesh_h / 2.0)
    mesh.upper_right = (half, half, +mesh_h / 2.0)
    mesh_filter = openmc.MeshFilter(mesh)

    e3 = openmc.EnergyFilter([E_MIN, E_THERMAL, E_FAST, E_MAX])
    e2 = openmc.EnergyFilter([E_MIN, E_THERMAL, E_MAX])

    t = openmc.Tally(name="flux_mesh")
    t.filters = [mesh_filter, e3]       # filter order: mesh slowest, energy fastest
    t.scores = ["flux"]
    tallies.append(t)

    t = openmc.Tally(name="fission_mesh")
    t.filters = [mesh_filter]
    t.scores = ["fission"]
    tallies.append(t)

    e_grid = np.logspace(np.log10(E_MIN), np.log10(E_MAX), n_spec + 1)
    t = openmc.Tally(name="spectrum")
    t.filters = [openmc.EnergyFilter(e_grid)]
    t.scores = ["flux"]
    tallies.append(t)

    t = openmc.Tally(name="fuel_rates")
    t.filters = [openmc.MaterialFilter([uo2]), e2]
    t.scores = ["absorption", "fission", "nu-fission"]
    tallies.append(t)

    t = openmc.Tally(name="global_rates")
    t.filters = [e2]
    t.scores = ["absorption", "fission", "nu-fission"]
    tallies.append(t)

    return tallies, e_grid, mesh_h


# ---------------------------------------------------------------------------
# Single sweep point
# ---------------------------------------------------------------------------
def make_source(openmc, half):
    lo = (-half, -half, -1.0)
    hi = (+half, +half, +1.0)
    src_cls = getattr(openmc, "IndependentSource", None) or openmc.Source
    try:
        return src_cls(space=openmc.stats.Box(lo, hi),
                       constraints={"fissionable": True})
    except TypeError:
        return src_cls(space=openmc.stats.Box(lo, hi, only_fissionable=True))


def resolve_statepoint(returned, rundir):
    if returned is not None and os.path.exists(str(returned)):
        return str(returned)

    candidates = glob.glob(os.path.join(rundir, "statepoint.*.h5"))
    if not candidates:
        candidates = glob.glob(os.path.join(os.getcwd(), "statepoint.*.h5"))
    if not candidates:
        raise RuntimeError(
            "No statepoint file was produced.\n"
            "  searched: %s\n"
            "            %s\n"
            "  directory contents: %s\n"
            "Set CFG.verbose = True to see the OpenMC output."
            % (os.path.abspath(rundir), os.getcwd(),
               sorted(os.listdir(rundir)) if os.path.isdir(rundir) else "(missing)")
        )

    def batch_no(path):
        m = re.search(r"statepoint\.(\d+)\.h5$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    candidates.sort(key=lambda p: (batch_no(p), os.path.getmtime(p)))
    return candidates[-1]


def symmetrize(a):
    b = (a + a[::-1, :] + a[:, ::-1] + a[::-1, ::-1]) / 4.0
    return (b + b.T) / 2.0


def run_point(openmc, idx, pitch, water_density, cfg, workdir):
    """Run one moderation point and return a dictionary of results."""
    uo2, helium, clad, water = build_materials(
        openmc, cfg.enrichment, water_density, cfg.boron_ppm, cfg.hot
    )
    materials = openmc.Materials([uo2, helium, clad, water])

    geometry, half = build_geometry(
        openmc, pitch, cfg.lattice, uo2, helium, clad, water, cfg.holes
    )
    n_mesh = cfg.lattice * cfg.mesh_per_pin
    tallies, e_grid, mesh_h = build_tallies(openmc, half, n_mesh, uo2)

    settings = openmc.Settings()
    settings.particles = cfg.particles
    settings.batches = cfg.batches
    settings.inactive = cfg.inactive
    settings.source = make_source(openmc, half)
    settings.output = {"tallies": False, "summary": False}
    settings.verbosity = 4 if cfg.verbose else 1
    if cfg.hot:
        settings.temperature = {"method": "interpolation", "tolerance": 1000.0}

    model = openmc.Model(
        geometry=geometry, materials=materials, settings=settings, tallies=tallies
    )

    rundir = os.path.join(workdir, "run_%02d" % idx)
    os.makedirs(rundir, exist_ok=True)
    for stale in glob.glob(os.path.join(rundir, "statepoint.*.h5")):
        os.remove(stale)          # never pick up a statepoint from an earlier run

    t0 = time.time()
    returned = model.run(cwd=rundir, threads=cfg.threads, output=cfg.verbose)
    wall = time.time() - t0
    sp_path = resolve_statepoint(returned, rundir)

    with openmc.StatePoint(sp_path) as sp:
        k = sp.keff.nominal_value
        k_std = sp.keff.std_dev

        # --- 3-group flux maps ------------------------------------------
        t = sp.get_tally(name="flux_mesh")
        d = t.mean.ravel().reshape(n_mesh * n_mesh, 3)    # [mesh, energy]
        cell_vol = (2.0 * half / n_mesh) ** 2 * mesh_h
        d = d / cell_vol                                   # cm^-2 per source neutron
        # mesh ordering: x fastest, then y, then z
        maps = d.reshape(1, n_mesh, n_mesh, 3)[0]          # (ny, nx, 3)
        flux_th = maps[:, :, 0]
        flux_epi = maps[:, :, 1]
        flux_fast = maps[:, :, 2]
        if cfg.symmetrize:
            flux_th = symmetrize(flux_th)
            flux_epi = symmetrize(flux_epi)
            flux_fast = symmetrize(flux_fast)

        # --- fission rate map -------------------------------------------
        t = sp.get_tally(name="fission_mesh")
        fis = t.mean.ravel().reshape(1, n_mesh, n_mesh)[0] / cell_vol
        if cfg.symmetrize:
            fis = symmetrize(fis)

        # --- spectrum -----------------------------------------------------
        t = sp.get_tally(name="spectrum")
        phi = t.mean.ravel()
        dlethargy = np.log(e_grid[1:] / e_grid[:-1])
        phi_leth = phi / dlethargy
        e_mid = np.sqrt(e_grid[1:] * e_grid[:-1])
        mean_E = float(np.sum(phi * e_mid) / np.sum(phi))

        # --- reaction rates (thermal / epithermal+fast) -------------------
        t = sp.get_tally(name="fuel_rates")
        fr = t.mean.reshape(2, 3)        # [energy group, score]
        t = sp.get_tally(name="global_rates")
        gr = t.mean.reshape(2, 3)

        abs_f_th, fis_f_th, nuf_f_th = fr[0]
        abs_f_ep, fis_f_ep, nuf_f_ep = fr[1]
        abs_t_th, fis_t_th, nuf_t_th = gr[0]
        abs_t_ep, fis_t_ep, nuf_t_ep = gr[1]

        f_util = float(abs_f_th / abs_t_th) if abs_t_th > 0 else np.nan
        fis_tot = fis_t_th + fis_t_ep
        th_fis_frac = float(fis_t_th / fis_tot) if fis_tot > 0 else np.nan
        abs_tot = abs_t_th + abs_t_ep
        epi_abs_frac = float(abs_f_ep / abs_tot) if abs_tot > 0 else np.nan
        eta = float((nuf_f_th + nuf_f_ep) / (abs_f_th + abs_f_ep))

        phi_th = float(flux_th.mean())
        phi_epi = float(flux_epi.mean())
        phi_fast = float(flux_fast.mean())
        spec_index = phi_th / phi_fast if phi_fast > 0 else np.nan
        th_flux_frac = phi_th / (phi_th + phi_epi + phi_fast)

    if not cfg.keep_runs:
        shutil.rmtree(rundir, ignore_errors=True)

    # --- volume / atom-ratio metrics (whole assembly, per unit height) ---
    n_cells = cfg.lattice ** 2
    n_holes = len(hole_positions(cfg.lattice, cfg.holes))
    n_pins = n_cells - n_holes
    a_fuel = n_pins * np.pi * R_FUEL ** 2
    a_mod = n_cells * pitch ** 2 - n_pins * np.pi * R_CLAD ** 2
    if cfg.holes == "tube":
        a_mod -= n_holes * np.pi * (R_GT_OUT ** 2 - R_GT_IN ** 2)
    vm_vf = a_mod / a_fuel
    n_h = 2.0 * water_density / M_H2O * N_AVO * a_mod
    n_u = RHO_UO2 / M_UO2 * N_AVO * a_fuel
    h_u = n_h / n_u

    return dict(
        pitch=pitch, water_density=water_density, k=k, k_std=k_std,
        flux_th=flux_th, flux_epi=flux_epi, flux_fast=flux_fast, fission=fis,
        extent=np.array([-half, half, -half, half]),
        spec_phi=phi_leth, mean_E=mean_E,
        f_util=f_util, th_fis_frac=th_fis_frac, epi_abs_frac=epi_abs_frac,
        eta=eta, spec_index=spec_index, th_flux_frac=th_flux_frac,
        phi_th=phi_th, phi_epi=phi_epi, phi_fast=phi_fast,
        vm_vf=vm_vf, h_u=h_u, wall=wall, e_grid=e_grid,
    )


# ---------------------------------------------------------------------------
# Whole sweep
# ---------------------------------------------------------------------------
def normalize_cfg(cfg):
    """Apply the quick / hq presets and validate the settings."""
    if cfg.quick:
        cfg.points = min(cfg.points, 7)
        cfg.particles = 1500
        cfg.batches = 25
        cfg.inactive = 8
        cfg.mesh_per_pin = 2
    if cfg.hq:
        cfg.particles = max(cfg.particles, 40000)
        cfg.batches = max(cfg.batches, 110)
        cfg.inactive = max(cfg.inactive, 25)
        cfg.mesh_per_pin = max(cfg.mesh_per_pin, 6)

    if cfg.mode not in ("pitch", "density"):
        sys.exit("ERROR: CFG.mode must be 'pitch' or 'density'.")
    if cfg.scale not in ("global", "frame"):
        sys.exit("ERROR: CFG.scale must be 'global' or 'frame'.")

    if cfg.holes is None:
        # A real guide tube does not fit below pitch = 1.204 cm, so pitch mode
        # uses plain water holes: they are valid at any pitch and they give the
        # assembly the structure a uniform lattice simply does not have.
        cfg.holes = "tube" if cfg.mode == "density" else "water"
    if cfg.holes not in ("tube", "water", "none"):
        sys.exit("ERROR: CFG.holes must be 'tube', 'water', 'none' or None.")
    if cfg.mode == "pitch" and cfg.holes == "tube" and cfg.pitch_min < 2 * R_GT_OUT:
        sys.exit("ERROR: pitch_min=%.3f cm is smaller than the guide tube diameter "
                 "2*%.3f=%.3f cm. Set CFG.holes = 'water' or raise pitch_min."
                 % (cfg.pitch_min, R_GT_OUT, 2 * R_GT_OUT))
    if cfg.mode == "density" and cfg.holes == "tube" and cfg.pitch < 2 * R_GT_OUT:
        sys.exit("ERROR: pitch=%.3f cm is smaller than the guide tube diameter "
                 "%.3f cm. Set CFG.holes = 'water'." % (cfg.pitch, 2 * R_GT_OUT))
    return cfg


def run_sweep(cfg):
    
    openmc = _import_openmc(cfg.cross_sections)

    if cfg.mode == "pitch":
        pitches = np.linspace(cfg.pitch_min, cfg.pitch_max, cfg.points)
        densities = np.full(cfg.points, cfg.water_density)
        param = pitches
        param_name = "pitch"
        param_label = "Pin pitch [cm]"
    else:
        densities = np.linspace(cfg.dens_min, cfg.dens_max, cfg.points)
        pitches = np.full(cfg.points, cfg.pitch)
        param = densities
        param_name = "water_density"
        param_label = "Water density [g/cm3]"

    os.makedirs(cfg.workdir, exist_ok=True)
    out = cfg.npz_out or "results_%s.npz" % cfg.mode

    print("=" * 78)
    print(" PWR ASSEMBLY MODERATION SWEEP  (OpenMC %s)" % openmc.__version__)
    n_holes = len(hole_positions(cfg.lattice, cfg.holes))
    print(" mode: %-8s | points: %d | lattice: %dx%d | holes: %s (%d)"
          % (cfg.mode, cfg.points, cfg.lattice, cfg.lattice, cfg.holes, n_holes))
    print(" fuel: UO2 %.2f%% enriched, r=%.4f cm (CONSTANT), %d pins"
          % (cfg.enrichment, R_FUEL, cfg.lattice ** 2 - n_holes))
    print(" stats: %d particles x %d batches (%d inactive) | map: %dx%d cells%s"
          % (cfg.particles, cfg.batches, cfg.inactive,
             cfg.lattice * cfg.mesh_per_pin, cfg.lattice * cfg.mesh_per_pin,
             " | 8-fold symmetrised" if cfg.symmetrize else ""))
    print("=" * 78)
    print("%3s %8s %8s %8s %8s %10s %9s %8s %8s"
          % ("#", "pitch", "rho_H2O", "Vm/Vf", "N_H/N_U", "k_inf", "+/-", "f", "time"))
    print("-" * 78)

    results = []
    for i, (pi, di) in enumerate(zip(pitches, densities)):
        r = run_point(openmc, i, float(pi), float(di), cfg, cfg.workdir)
        results.append(r)
        print("%3d %8.4f %8.4f %8.3f %8.2f %10.5f %9.5f %8.4f %7.1fs"
              % (i, r["pitch"], r["water_density"], r["vm_vf"], r["h_u"],
                 r["k"], r["k_std"], r["f_util"], r["wall"]))
        sys.stdout.flush()

    # ---------------- assemble ----------------
    k = np.array([r["k"] for r in results])
    i_peak = int(np.argmax(k))
    if cfg.mode == "pitch":
        i_nom = int(np.argmin(np.abs(pitches - PITCH_NOM)))
    else:
        i_nom = int(np.argmin(np.abs(densities - 0.74)))

    meta = dict(
        mode=cfg.mode, param_name=param_name, param_label=param_label,
        lattice=cfg.lattice, holes=cfg.holes, n_holes=n_holes,
        mesh_per_pin=cfg.mesh_per_pin, symmetrized=bool(cfg.symmetrize),
        enrichment=cfg.enrichment, boron_ppm=cfg.boron_ppm,
        r_fuel=R_FUEL, r_clad=R_CLAD, pitch_nom=PITCH_NOM,
        particles=cfg.particles, batches=cfg.batches, inactive=cfg.inactive,
        hot=bool(cfg.hot), openmc_version=str(openmc.__version__),
        e_thermal=E_THERMAL, e_fast=E_FAST,
        i_peak=i_peak, i_nom=i_nom,
        cross_sections=os.environ["OPENMC_CROSS_SECTIONS"],
    )

    def col(key):
        return np.array([r[key] for r in results])

    data = dict(
        param=param, pitch=pitches, water_density=densities,
        vm_vf=col("vm_vf"), h_u=col("h_u"), k=k, k_std=col("k_std"),
        flux_th=col("flux_th"), flux_epi=col("flux_epi"),
        flux_fast=col("flux_fast"), fission=col("fission"),
        extent=col("extent"), spec_phi=col("spec_phi"),
        e_grid=results[0]["e_grid"], mean_E=col("mean_E"),
        f_util=col("f_util"), th_fis_frac=col("th_fis_frac"),
        epi_abs_frac=col("epi_abs_frac"), eta=col("eta"),
        spec_index=col("spec_index"), th_flux_frac=col("th_flux_frac"),
        phi_th=col("phi_th"), phi_epi=col("phi_epi"), phi_fast=col("phi_fast"),
    )

    # ---------------- save ----------------
    np.savez_compressed(out, meta=json.dumps(meta), **data)

    csv_path = os.path.splitext(out)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("idx,pitch_cm,rho_h2o,Vm_Vf,N_H_N_U,k_inf,k_std,f_util,"
                 "th_fis_frac,epi_abs_frac,eta,spec_index,mean_E_eV,regime\n")
        for i, r in enumerate(results):
            reg = ("under" if i < i_peak else "optimum" if i == i_peak else "over")
            fh.write("%d,%.4f,%.4f,%.4f,%.3f,%.6f,%.6f,%.5f,%.5f,%.5f,%.5f,%.4f,%.4f,%s\n"
                     % (i, r["pitch"], r["water_density"], r["vm_vf"], r["h_u"],
                        r["k"], r["k_std"], r["f_util"], r["th_fis_frac"],
                        r["epi_abs_frac"], r["eta"], r["spec_index"], r["mean_E"], reg))

    print("-" * 78)
    print("k_inf maximum : point %d  (%s = %.4f, N_H/N_U = %.2f, k = %.5f)"
          % (i_peak, param_name, param[i_peak], results[i_peak]["h_u"], k[i_peak]))
    print("nominal PWR   : point %d  (%s = %.4f)" % (i_nom, param_name, param[i_nom]))
    print("  -> points < %d : UNDER-MODERATED" % i_peak)
    print("  -> points > %d : OVER-MODERATED" % i_peak)
    print("\nSaved: %s\n       %s" % (out, csv_path))

    data["meta"] = meta
    return data


# ===========================================================================
#                       PANEL ANIMATION
# ===========================================================================
# ---------------------------------------------------------------------------
# Colour system (dark theme; dark-mode categorical slots, CVD validated)
# ---------------------------------------------------------------------------
SURFACE = "#1a1a19"
PANEL = "#232321"
INK = "#f0efec"
INK2 = "#c3c2b7"
INK3 = "#8a8a80"
GRID = "#3a3a36"

C_BLUE = "#3987e5"     # slot 1 - thermal
C_ORANGE = "#d95926"   # slot 2 - fast
C_AQUA = "#199e70"     # slot 3 - fission / f
C_YELLOW = "#c98500"   # slot 4 - secondary series

REGIME_COLORS = {
    "UNDER-MODERATED": "#d95926",
    "OPTIMUM MODERATION": "#0ca30c",
    "OVER-MODERATED": "#3987e5",
}

CMAP_TH = "Blues_r"      # single hue, monotonic lightness (low=dark, high=light)
CMAP_FAST = "Oranges_r"
CMAP_FIS = "Greens_r"

# Layout (figure fractions) --------------------------------------------------
ASPECT = 1080.0 / 1920.0
QW, QH = 0.435, 0.295                 # quadrant width / height
QX = (0.040, 0.525)                   # left / right quadrant left edge
QY = (0.515, 0.105)                   # top / bottom quadrant bottom edge
MAP_W = QH * ASPECT                   # square map width
PLOT_X = 0.245                        # small plot left edge within the quadrant
PLOT_W = QW - PLOT_X - 0.005

# Filled in by _import_matplotlib()
plt = animation = LogNorm = None


def _import_matplotlib():
    """Import matplotlib with the Agg backend, on first use only."""
    global plt, animation, LogNorm
    if plt is not None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib import animation as _animation
        from matplotlib.colors import LogNorm as _LogNorm
    except ImportError as exc:
        sys.exit("ERROR: matplotlib is needed for the animation (%s)." % exc)
    plt, animation, LogNorm = _plt, _animation, _LogNorm


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK3,
        "ytick.color": INK3,
        "grid.color": GRID,
        "font.size": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def hole_note(meta):
    """What sits at the 25 guide-tube positions of the lattice."""
    nh = meta.get("n_holes", 0)
    return {"tube": "%d guide tubes" % nh,
            "water": "%d water holes" % nh,
            "none": "uniform lattice"}.get(meta.get("holes", "none"), "")


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def synth_demo(n=15, nm=34, ne=500):
    """Synthetic (DEMO) data so the layout can be previewed without OpenMC."""
    pitch = np.linspace(1.00, 2.60, n)
    rho = np.full(n, 0.74)
    a_f = np.pi * R_FUEL ** 2
    a_m = pitch ** 2 - np.pi * R_CLAD ** 2
    vm_vf = a_m / a_f
    h_u = 2.0 * rho / M_H2O * a_m / (RHO_UO2 / M_UO2 * a_f)
    k = 1.33 * np.exp(-(np.log(h_u / 6.0) ** 2) / 0.95)
    y, x = np.mgrid[0:nm, 0:nm]
    per = nm / 17.0
    pin = (0.5 * (1 + np.cos(2 * np.pi * x / per))
           * 0.5 * (1 + np.cos(2 * np.pi * y / per)))        # 1 in water, 0 in fuel
    hole = np.zeros_like(pin)
    for (gi, gj) in GUIDE_TUBE_POSITIONS:
        hole[int(gi * per):int((gi + 1) * per), int(gj * per):int((gj + 1) * per)] = 1.0
    pin = np.where(hole > 0, 1.0, pin)
    e_grid = np.logspace(np.log10(E_MIN), np.log10(E_MAX), ne + 1)
    e_mid = np.sqrt(e_grid[1:] * e_grid[:-1])
    flux_th, flux_fast, flux_epi, fis, spec = [], [], [], [], []
    for i in range(n):
        s = h_u[i] / 6.0
        flux_th.append((0.55 + 0.45 * pin + 0.5 * hole) * s ** 1.6 * 1e-2)
        flux_fast.append((1.0 - 0.45 * pin - 0.3 * hole) / s ** 0.8 * 1e-2)
        flux_epi.append((0.7 + 0.3 * pin) * 1e-2)
        fis.append(((1.0 - 0.8 * pin) * (1.0 - hole) + 1e-3) * s ** 1.3 * 1e-3)
        th = np.exp(-(np.log10(e_mid) + 1.4) ** 2 / 0.5) * s ** 2
        fa = np.exp(-(np.log10(e_mid) - 6.1) ** 2 / 0.6)
        ep = 0.3 * ((e_mid > 1.0) & (e_mid < 1e5))
        spec.append(th + fa + ep)
    flux_th, flux_fast = np.array(flux_th), np.array(flux_fast)
    flux_epi, fis = np.array(flux_epi), np.array(fis)
    pth, pep, pfa = flux_th.mean((1, 2)), flux_epi.mean((1, 2)), flux_fast.mean((1, 2))
    i_peak = int(np.argmax(k))
    meta = dict(mode="pitch", param_name="pitch", param_label="Pin pitch [cm]",
                lattice=17, holes="water", n_holes=len(GUIDE_TUBE_POSITIONS),
                mesh_per_pin=4, symmetrized=True, enrichment=3.1, boron_ppm=0.0,
                r_fuel=R_FUEL, r_clad=R_CLAD, pitch_nom=PITCH_NOM,
                particles=0, batches=0, inactive=0, hot=False,
                openmc_version="DEMO", e_thermal=E_THERMAL, e_fast=E_FAST,
                i_peak=i_peak, i_nom=int(np.argmin(np.abs(pitch - PITCH_NOM))),
                demo=True, cross_sections="DEMO")
    return dict(
        param=pitch, pitch=pitch, water_density=rho, vm_vf=vm_vf, h_u=h_u,
        k=k, k_std=np.full(n, 4e-4),
        flux_th=flux_th, flux_epi=flux_epi, flux_fast=flux_fast, fission=fis,
        extent=np.array([[-8.5 * p, 8.5 * p, -8.5 * p, 8.5 * p] for p in pitch]),
        spec_phi=np.array(spec), e_grid=e_grid,
        mean_E=2.0e5 / (h_u / 3.0) ** 1.5,
        f_util=0.97 - 0.05 * h_u / 3.0,
        th_fis_frac=0.55 + 0.1 * np.tanh(h_u / 6 - 1),
        epi_abs_frac=0.40 * np.exp(-h_u / 9.0),
        eta=np.full(n, 1.85), spec_index=pth / pfa,
        th_flux_frac=pth / (pth + pep + pfa),
        phi_th=pth, phi_epi=pep, phi_fast=pfa, meta=meta,
    )


def build_frames(d, interp, hold):
    n = len(d["param"])
    i_peak = d["meta"]["i_peak"]
    seq = []
    for i in range(n - 1):
        if i in (0, i_peak):
            seq += [(i, i, 0.0)] * hold
        for j in range(interp):
            seq.append((i, i + 1, j / float(interp)))
    seq += [(n - 1, n - 1, 0.0)] * (hold + 1)
    return seq


def lerp(a, b, w):
    return a * (1.0 - w) + b * w


def regime(i_frac, i_peak, n):
    tol = max(0.35, 0.03 * n)
    if i_frac < i_peak - tol:
        return "UNDER-MODERATED"
    if i_frac > i_peak + tol:
        return "OVER-MODERATED"
    return "OPTIMUM MODERATION"


def positive_range(arr, lo_pct=1.0, hi_pct=99.8):
    v = arr[np.isfinite(arr) & (arr > 0)]
    if v.size == 0:
        return 1e-12, 1.0
    lo, hi = np.percentile(v, lo_pct), np.percentile(v, hi_pct)
    if lo <= 0 or lo >= hi:
        lo = hi / 1e3
    return lo, hi


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------
def mini_axes(fig, rect, labelsize=8.5):
    a = fig.add_axes(rect)
    a.set_facecolor("#1e1e1c")
    for s in a.spines.values():
        s.set_color(GRID)
    a.tick_params(colors=INK3, labelsize=labelsize, length=3, pad=2)
    a.grid(True, color=GRID, lw=0.6, alpha=0.7)
    a.set_axisbelow(True)
    return a


def panel_title(fig, qx, qy, text, subtitle=""):
    fig.text(qx, qy + QH + 0.042, text, fontsize=15, fontweight="bold",
             color=INK, va="center", ha="left")
    if subtitle:
        fig.text(qx, qy + QH + 0.016, subtitle, fontsize=10, color=INK3,
                 va="center", ha="left")


class MapPanel:
    """Square flux map with a colour bar attached to its right edge."""

    def __init__(self, fig, qx, qy, data, cmap, scale="global"):
        self.ax = fig.add_axes([qx, qy, MAP_W, QH])
        self.scale = scale
        vmin, vmax = positive_range(data)
        self.im = self.ax.imshow(
            data[0], origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="bilinear", aspect="auto")
        self.ax.set_xlabel("x [cm]", fontsize=10, labelpad=2)
        self.ax.set_ylabel("y [cm]", fontsize=10, labelpad=2)
        self.ax.tick_params(labelsize=9)
        cax = fig.add_axes([qx + MAP_W + 0.006, qy, 0.0085, QH])
        cb = fig.colorbar(self.im, cax=cax)
        cb.ax.tick_params(colors=INK3, labelsize=8)
        cb.outline.set_edgecolor(GRID)
        self.cb = cb

    def update(self, arr, ext):
        if self.scale == "frame":
            vmin, vmax = positive_range(arr[None, ...], 1.0, 99.8)
            self.im.set_norm(LogNorm(vmin=vmin, vmax=vmax))
        self.im.set_data(arr)
        self.im.set_extent(ext)
        self.ax.set_xlim(ext[0], ext[1])
        self.ax.set_ylim(ext[2], ext[3])


def curve_plot(a, x, series, xlabel, ylabel, x_peak, x_nom, logy=False):
    """Sweep curve(s) plus a moving cursor. series: [(y, colour, label)]"""
    a.axvline(x_peak, color=INK3, lw=1.1, ls="--", alpha=0.85)
    if x_nom is not None:
        a.axvline(x_nom, color=C_YELLOW, lw=1.1, ls=":", alpha=0.95)
    markers = []
    for (y, c, lab) in series:
        a.plot(x, y, color=c, lw=2.0, label=lab, solid_capstyle="round")
        (m,) = a.plot([x[0]], [y[0]], "o", color=c, ms=9, mec=SURFACE, mew=1.8,
                      zorder=6)
        markers.append((m, y))
    v = a.axvline(x[0], color=INK, lw=1.2, alpha=0.6)
    a.set_xlabel(xlabel, fontsize=9.5, color=INK2, labelpad=2)
    a.set_ylabel(ylabel, fontsize=9.5, color=INK2, labelpad=3)
    if len(series) > 1:
        a.legend(fontsize=9, loc="best", labelcolor=INK2, handlelength=1.4,
                 borderpad=0.2, handletextpad=0.6)
    if logy:
        a.set_yscale("log")
    a.set_xlim(x.min(), x.max())
    return markers, v


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render(d, out, fps=15, interp=5, hold=10, pingpong=True, gif=False,
           dpi=100, scale="global"):
    _import_matplotlib()
    style()
    meta = d["meta"]
    n = len(d["param"])
    x = d["h_u"]                      # common moderation parameter: N_H / N_U
    i_peak = int(meta["i_peak"])
    i_nom = meta.get("i_nom")
    x_peak = x[i_peak]
    x_nom = x[i_nom] if i_nom is not None else None
    demo = bool(meta.get("demo", False))
    XL = r"$N_H/N_U$  (hydrogen / uranium atom ratio)"

    fig = plt.figure(figsize=(19.20, 10.80), dpi=dpi)

    # ---------------- Panel 1: thermal flux ----------------
    qx, qy = QX[0], QY[0]
    panel_title(fig, qx, qy,
                "1 — THERMAL NEUTRON FLUX   (E < %.3g eV)" % meta["e_thermal"],
                r"map: flux [cm$^{-2}$ per source neutron]   ·   "
                r"plot: infinite multiplication factor $k_\infty$")
    p1 = MapPanel(fig, qx, qy, d["flux_th"], CMAP_TH, scale)
    a1 = mini_axes(fig, [qx + PLOT_X, qy, PLOT_W, QH])
    m1, v1 = curve_plot(a1, x, [(d["k"], C_BLUE, r"$k_\infty$")], XL,
                        r"$k_\infty$", x_peak, x_nom)

    # ---------------- Panel 2: fast flux ----------------
    qx, qy = QX[1], QY[0]
    panel_title(fig, qx, qy,
                "2 — FAST NEUTRON FLUX   (E > %.3g MeV)" % (meta["e_fast"] / 1e6),
                r"map: flux [cm$^{-2}$ per source neutron]   ·   "
                r"plot: spectral index (thermal / fast flux ratio)")
    p2 = MapPanel(fig, qx, qy, d["flux_fast"], CMAP_FAST, scale)
    a2 = mini_axes(fig, [qx + PLOT_X, qy, PLOT_W, QH])
    m2, v2 = curve_plot(a2, x, [(d["spec_index"], C_ORANGE, None)], XL,
                        r"$\phi_{thermal}\,/\,\phi_{fast}$", x_peak, x_nom, logy=True)

    # ---------------- Panel 3: fission rate ----------------
    qx, qy = QX[0], QY[1]
    panel_title(fig, qx, qy, "3 — FISSION REACTION RATE DENSITY",
                r"map: fissions [cm$^{-3}$ per source neutron]   ·   "
                r"plot: thermal utilization and epithermal capture")
    p3 = MapPanel(fig, qx, qy, d["fission"], CMAP_FIS, scale)
    a3 = mini_axes(fig, [qx + PLOT_X, qy, PLOT_W, QH])
    m3, v3 = curve_plot(
        a3, x,
        [(d["f_util"], C_AQUA, "thermal utilization  $f$"),
         (d["epi_abs_frac"], C_YELLOW, "epithermal capture fraction")],
        XL, "fraction  [-]", x_peak, x_nom)

    # ---------------- Panel 4: spectrum ----------------
    qx, qy = QX[1], QY[1]
    panel_title(fig, qx, qy, "4 — NEUTRON SPECTRUM   (assembly average)",
                "curve: flux per unit lethargy   ·   plots: mean neutron energy "
                "and thermal flux fraction")
    e_grid = d["e_grid"]
    e_mid = np.sqrt(e_grid[1:] * e_grid[:-1])
    spec = d["spec_phi"].astype(float)
    spec = spec / np.maximum(spec.sum(axis=1, keepdims=True), 1e-30)
    ax4 = mini_axes(fig, [qx, qy, 0.228, QH], labelsize=9)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.axvspan(e_grid[0], meta["e_thermal"], color=C_BLUE, alpha=0.10, zorder=0)
    ax4.axvspan(meta["e_fast"], e_grid[-1], color=C_ORANGE, alpha=0.10, zorder=0)
    for i in range(n):
        ax4.plot(e_mid, spec[i], color=INK3, lw=0.7, alpha=0.30, zorder=1)
    (spec_line,) = ax4.plot(e_mid, spec[0], color=C_BLUE, lw=2.6, zorder=4,
                            solid_capstyle="round")
    ax4.text(0.03, 0.96, "thermal", transform=ax4.transAxes, color=C_BLUE,
             fontsize=9.5, va="top")
    ax4.text(0.97, 0.96, "fast", transform=ax4.transAxes, color=C_ORANGE,
             fontsize=9.5, va="top", ha="right")
    ax4.text(0.03, 0.06, "faint grey: all sweep points",
             transform=ax4.transAxes, color=INK3, fontsize=8.5, va="bottom")
    lo = max(spec[spec > 0].min(), spec.max() * 2e-4)
    ax4.set_ylim(lo, spec.max() * 2.2)
    ax4.set_xlim(e_grid[0], e_grid[-1])
    ax4.set_xlabel("Neutron energy [eV]", fontsize=9.5, color=INK2, labelpad=2)
    ax4.set_ylabel("flux per unit lethargy (normalised)", fontsize=9.5, color=INK2,
                   labelpad=3)

    mh = (QH - 0.062) / 2.0
    a4a = mini_axes(fig, [qx + 0.283, qy + QH - mh, QW - 0.288, mh])
    m4a, v4a = curve_plot(a4a, x, [(d["mean_E"], C_BLUE, None)], "",
                          r"mean energy $\bar{E}$ [eV]", x_peak, x_nom, logy=True)
    a4a.set_xticklabels([])
    a4b = mini_axes(fig, [qx + 0.283, qy, QW - 0.288, mh])
    m4b, v4b = curve_plot(a4b, x, [(d["th_flux_frac"], C_ORANGE, None)], XL,
                          "thermal flux fraction", x_peak, x_nom)

    # ---------------- header / status ----------------
    fig.text(0.040, 0.962, "MODERATION EFFECT IN A PWR FUEL ASSEMBLY",
             fontsize=28, fontweight="bold", color=INK, va="center")
    fig.text(0.040, 0.925,
             "Code %s   ·   %dx%d assembly, %s   ·   UO$_2$ %.1f%% enriched, "
             "r = %.4f cm (CONSTANT)   ·   varying: %s"
             % (meta["openmc_version"], meta["lattice"], meta["lattice"],
                hole_note(meta), meta["enrichment"], meta["r_fuel"],
                "water VOLUME (pin pitch)" if meta["mode"] == "pitch"
                else "water DENSITY"),
             fontsize=12.5, color=INK2, va="center")
    status = fig.text(0.040, 0.884, "", fontsize=14, color=INK, va="center",
                      family="monospace")
    badge = fig.text(0.960, 0.945, "", fontsize=20, fontweight="bold", color=INK,
                     va="center", ha="right",
                     bbox=dict(boxstyle="round,pad=0.5", fc="#2a2a26", ec=GRID,
                               lw=1.4))

    foot = ("%d sweep points  ·  %d map cells per pin  ·  "
            "infinite lattice, reflective x–y (2-D)  ·  "
            "dashed grey: $k_\\infty$ maximum = optimum moderation  ·  "
            "dotted yellow: nominal PWR point  ·  frames between points interpolated"
            % (n, meta.get("mesh_per_pin", 0))
            + ("  ·  maps 8-fold symmetrised" if meta.get("symmetrized") else ""))
    if demo:
        foot = "!!! DEMO — SYNTHETIC DATA  ·  " + foot
    fig.text(0.040, 0.018, foot, fontsize=10, color=INK3, va="center")

    bar_bg = fig.add_axes([0.040, 0.042, 0.920, 0.007])
    bar_bg.set_facecolor("#2f2f2b")
    bar_bg.set_xticks([]); bar_bg.set_yticks([])
    for s in bar_bg.spines.values():
        s.set_visible(False)
    bar = bar_bg.barh([0], [0], height=1.0, color=C_BLUE, align="center")[0]
    bar_bg.set_xlim(0, 1); bar_bg.set_ylim(-0.5, 0.5)

    if demo:
        fig.text(0.5, 0.5, "DEMO", fontsize=190, color="#ffffff", alpha=0.05,
                 ha="center", va="center", rotation=20, zorder=50, fontweight="bold")

    # ---------------- frame sequence ----------------
    seq = build_frames(d, interp, hold)
    if pingpong:
        seq = seq + [(b, a, 1.0 - w) for (a, b, w) in reversed(seq[1:-1])]

    pitches, rho, vm_vf = d["pitch"], d["water_density"], d["vm_vf"]
    k, k_std = d["k"], d["k_std"]
    all_markers = m1 + m2 + m3 + m4a + m4b
    all_vlines = (v1, v2, v3, v4a, v4b)

    def frame(fi):
        i0, i1, w = seq[fi]
        i_frac = i0 * (1 - w) + i1 * w
        ext = lerp(d["extent"][i0], d["extent"][i1], w)
        p1.update(lerp(d["flux_th"][i0], d["flux_th"][i1], w), ext)
        p2.update(lerp(d["flux_fast"][i0], d["flux_fast"][i1], w), ext)
        p3.update(lerp(d["fission"][i0], d["fission"][i1], w), ext)
        spec_line.set_ydata(lerp(spec[i0], spec[i1], w))

        xc = lerp(x[i0], x[i1], w)
        for (m, y) in all_markers:
            m.set_data([xc], [lerp(y[i0], y[i1], w)])
        for v in all_vlines:
            v.set_xdata([xc, xc])

        reg = regime(i_frac, i_peak, n)
        badge.set_text(reg)
        badge.set_color(REGIME_COLORS[reg])
        badge.get_bbox_patch().set_edgecolor(REGIME_COLORS[reg])
        status.set_text(
            "pitch = %5.3f cm    rho(H2O) = %4.2f g/cm3    V_m/V_f = %5.2f    "
            "N_H/N_U = %6.2f    k_inf = %7.5f +/- %.5f"
            % (lerp(pitches[i0], pitches[i1], w), lerp(rho[i0], rho[i1], w),
               lerp(vm_vf[i0], vm_vf[i1], w), xc,
               lerp(k[i0], k[i1], w), lerp(k_std[i0], k_std[i1], w)))
        bar.set_width((fi + 1) / float(len(seq)))
        bar.set_color(REGIME_COLORS[reg])
        return []

    frame(0)
    n_frames = len(seq)
    print("Frames: %d  |  fps: %d  |  duration: %.1f s  |  resolution: %dx%d"
          % (n_frames, fps, n_frames / float(fps),
             int(fig.get_figwidth() * dpi), int(fig.get_figheight() * dpi)))

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    root, ext_ = os.path.splitext(out)
    if ext_.lower() != ".mp4":
        out = root + ".mp4"
    try:
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium"])
        anim.save(out, writer=writer, dpi=dpi)
        print("Wrote: %s" % out)
    except Exception as exc:
        print("WARNING: could not write MP4 (%s). Falling back to GIF." % exc)
        gif = True
    if gif:
        anim.save(root + ".gif",
                  writer=animation.PillowWriter(fps=min(fps, 20)), dpi=dpi)
        print("Wrote: %s.gif" % root)

    # still frames for the three regimes
    for tag, idx in (("under", 0), ("optimum", i_peak), ("over", n - 1)):
        frame(min(range(len(seq)), key=lambda q: abs(seq[q][0] - idx)))
        fig.savefig("%s_%s.png" % (root, tag), dpi=dpi)
        print("Wrote: %s_%s.png" % (root, tag))
    plt.close(fig)


# ===========================================================================
#                                   RUN
# ===========================================================================
def main():
    cfg = normalize_cfg(CFG)

    # ---- 1) data ----
    if cfg.demo:
        print("DEMO mode: using synthetic data (OpenMC is not called).")
        data = synth_demo()
    elif cfg.run_sweep:
        data = run_sweep(cfg)
    else:
        npz_path = cfg.npz_out or "results_%s.npz" % cfg.mode
        if not os.path.exists(npz_path):
            sys.exit("ERROR: CFG.run_sweep = False but '%s' was not found.\n"
                     "Run the sweep first with CFG.run_sweep = True." % npz_path)
        print("Reading the existing sweep result: %s" % npz_path)
        data = load_npz(npz_path)

    # ---- 2) animation ----
    if not cfg.make_video:
        print("CFG.make_video = False -> no animation was produced.")
        return

    out = cfg.video_out or ("moderation_demo.mp4" if cfg.demo
                            else "moderation_%s.mp4" % cfg.mode)
    print("")
    render(data, out, fps=cfg.fps, interp=max(1, cfg.interp), hold=cfg.hold,
           pingpong=cfg.pingpong, gif=cfg.gif, dpi=cfg.dpi, scale=cfg.scale)


if __name__ == "__main__":
    main()
