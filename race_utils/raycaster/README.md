# raycaster — 2D LiDAR ray-casting benchmark & comparison

A tool for organizing, comparing, and validating the 2D LiDAR **raycaster** implementations in the UNICORN racing-stack.
Since simulator and particle-filter performance are effectively governed by raycaster efficiency, this measures quantitatively which method suits which workload (especially on low-end hardware).

> This is a **research/benchmark tool**, not a ROS package (it includes `COLCON_IGNORE` and is excluded from colcon builds).
> Going forward, we re-measure on NUC, Mac mini, etc. and accumulate dated results under `results/`.

## Structure
```
tools/raycaster/
├── README.md                 # this document (analysis, conclusions, algorithms, references)
├── SETUP.md                  # reproducible env setup (venv, range_libc build, jax)
├── raycaster.py              # ★ unified library RaycastEngine (shared by sim + PF)
├── benchmarks/
│   ├── bench_raycast.py       # numba + range_libc(bl/rm/cddt/pcddt/glt) + segment
│   ├── bench_jax.py           # jax DT sphere-tracing (single/batch, GPU/CPU)
│   ├── bench_workload.py      # real-time feasibility: SIM vs PF workload validation
│   └── precompute_lut.py      # GLT/CDDT LUT precompute → save → load test
└── results/
    ├── 2026-06-17_ryzen7950x_rtx4080super.md   # per-machine raw results (date + environment)
    └── ray_casting_methods_comparison.html      # interactive visualization
```

## Key Results (Ryzen 7950X + RTX 4080S, 2026-06-17 — full data in `results/`)

`1 scan = one 1080-beam frame`. Single-scan throughput:

| Raycaster | Backend | scans/s | vs numba | One-liner |
|---|---|--:|--:|---|
| jax DT-march, batched | GPU | 155,300 | 9.7× | Best for massive parallelism (needs NVIDIA) |
| range_libc **GLT** | CPU | 147,700 | 9.2× | Fastest single, large init & memory |
| range_libc **PCDDT/CDDT** | CPU | ~52,000 | ~3× | **Balanced — recommended** |
| range_libc RM | CPU | 23,500 | 1.5× | Simple & robust |
| **numba f1tenth_gym** *(current sim)* | CPU | 16,100 | 1.0× | Baseline |
| range_libc BL (Bresenham) | CPU | 3,600 | 0.2× | Slow |

### Real-time Feasibility — Simulator vs Particle Filter (low-end focus)

| Workload | Pattern | Demand |
|---|---|--:|
| Simulator | 2 lidar × 40 Hz × 2200 beams | 176 k rays/s (latency) |
| Particle Filter | 4000 particles × 100 beams × 40 Hz | 16 M rays/s (throughput) |

25 ms/cycle budget, headroom = 25 ms / measured time (≈ how many × slower HW still runs real-time):

| Method | SIM ↑ | PF ↑ | Verdict |
|---|--:|--:|---|
| numba / jax-CPU | 117× / 40× | **0.7× / 0.4×** | **PF fails (even on desktop)** |
| range_libc cddt/pcddt | ~300× | ~3× | PF OK (risky on weak laptops) |
| **range_libc glt** | 1048× | **16×** | **PF safe (even low-end)** |
| jax-GPU | 25× | 9× | PF OK (NVIDIA GPU only) |

**Conclusion**
- **Simulator**: latency-bound, everything has 100×+ headroom → no problem on low-end. Choose it on the basis of *code unification*.
- **Particle Filter**: throughput-bound, the real constraint. numba and jax-CPU fail real-time.
  To guarantee even low-end/portable hardware, use **PF = range_libc GLT (CPU)**. If that's too heavy, use PCDDT + fewer particles/beams.

### jax-GPU Platform Availability

| Platform | jax GPU | PF possible? |
|---|---|---|
| NVIDIA dGPU | ✅ `jax[cuda]` | ◎ |
| Intel iGPU (NUC) | ❌ CPU fallback (`intel-extension-for-openxla` is experimental/Arc-focused) | ❌ |
| Apple Silicon (Mac mini) | △ `jax-metal` experimental & incomplete | ✗ unreliable |
| No GPU | ❌ | ❌ |

→ On NUC, Mac mini, etc. you can't rely on jax-GPU. **range_libc (CPU)** is the right answer for both portability and performance
(being C++, it compiles anywhere on x86/ARM). range_libc `rmgpu` is also CUDA-only, so it shares the same limitation.

## Algorithms at a Glance ("step forward by 0.01" is the naive version)

| Method | Principle |
|---|---|
| **Bresenham/DDA** (`bl`) | Steps through the grid cell by cell with integer increments, checking occupancy (the integer version of a fixed step) |
| **Ray marching = sphere tracing** (`rm`, numba, jax) | Uses a **distance transform (DT)** to jump by the "distance to the nearest obstacle" in one go → adaptive. (The smart version of the "0.01 step") |
| **CDDT / PCDDT** (`cddt`) | Discretizes direction to store a compressed distance LUT; queries are lookup + interpolation. PCDDT adds pruning |
| **Giant LUT** (`glt`) | Precomputes the full (x,y,θ)→distance, O(1) lookup. Fastest, high memory |
| **Segment analytic** | Computes ray-segment intersection analytically. Grid-independent & exact (can be accelerated with a 2D z-buffer) |

> numba, range_libc `rm`, and jax are **all the same DT sphere-tracing algorithm**; only the implementation (numba/C++/XLA) differs.

## Strategy: Unify the Whole Stack on range_libc

particle_filter already uses range_libc (`pcddt`). If the simulator's (`f1tenth_gym_ros`) numba is also swapped for the
range_libc backend, then **sim, localization, and (optionally) planner share a single raycast engine** →
map loading, coordinate conventions, and distance definitions are unified, simplifying maintenance. Share one built `range_libc.so`.
Recommended: swap the sim backend to PCDDT (or GLT/LUT for low-end) → verify output matches numba → unify.

## Unified Library `RaycastEngine` ([raycaster.py](raycaster.py))

A **single raycaster used by both sim and PF**. Self-contained: vendored
[range_libc](range_libc/) + numba [laser_models](vendor/) (no external repo needed).

```python
from raycaster import RaycastEngine
occ, res, origin = RaycastEngine.load_map_yaml("maps/f/f.yaml")
e = RaycastEngine(backend="lut", max_range_m=10.0, theta_disc=720).set_map(occ, res, origin)
ranges = e.scan([x, y, theta], num_beams=1080, fov=4.7)          # simulator
ranges = e.calc_range_repeat_angles(particles[M,3], angles[K])    # particle filter
```
- **Simulator**: `scan(pose, num_beams, fov)` — drop-in for numba `ScanSimulator2D`.
- **Particle filter**: `calc_range_repeat_angles(...)`.

**Backends** — all verified vs the numba sim ([`benchmarks/verify_backends.py`](benchmarks/verify_backends.py), [`results/2026-06-18_range_libc_fixed.md`](results/2026-06-18_range_libc_fixed.md)):
- `rm`/`cddt`/`pcddt`/`glt`/`bl`/**`rmgpu`** — vendored range_libc C++. The numpy
  `PyOMap` ctor was **fixed** (it wrote a *transposed* grid + no world transform → wrong
  on real maps) and the Cython wrapper **modernized for Jazzy** (Cython 3 / numpy 2 /
  py3.12, no `old_build_ext`). Query in **world meters** directly.
- `lut` — precomputed numpy table from the numba oracle → portable (no range_libc / numba
  at query time → NUC / Mac mini).

### Verified vs the existing numba sim (`f` map, 1080 beams)

| backend | sim MAE vs numba | speed | use when |
|---|--:|--:|---|
| **rm** / **rmgpu** | **2.9 cm** (exact DT) | 1.3× CPU / GPU | accuracy; rmgpu for PF batches (4000×60 = **0.31 ms ≈ 80× real-time**) |
| **glt** | 8.5 cm | **6.1×** | fastest CPU single scan (high memory) |
| **pcddt** | 9.0 cm | 2.1× | low memory / init (PF default) |
| **lut** (TD=720) | 1.8 cm | 2.8× | portable, no range_libc needed |

(`cddt`/`pcddt`/`glt` ~9 cm = their `theta_disc=112` quantization — raise it for accuracy.)
→ RaycastEngine matches the current simulator (≤3 cm) while being 1.3–6× faster, and range_libc
can be swapped for `pcddt` (memory) or `rmgpu` (GPU) as needed.

### Precompute → save → load (`backend='lut'`) — low-end / portability
Build the table **once, save `.npz`, then load and query with pure numpy** (no range_libc, no numba):

```python
RaycastEngine(backend="lut", theta_disc=720).set_map(occ, res, origin).save_lut("f_lut.npz")  # once (~build s)
e = RaycastEngine.load_lut("f_lut.npz")    # afterwards: numpy only → NUC / Mac mini OK
```
PF (4000×100) runs at multiple-× real-time from the loaded table. **Copy the `.npz`
to a NUC / Mac mini and both sim and PF run without building range_libc.**
(`theta_disc` trades accuracy vs memory: 112 ≈ PF-grade, 720 ≈ sim-grade.)

## RViz demo ([`examples/`](examples/))

Drives a pose around an F1TENTH map and publishes the live RaycastEngine scan
(`/scan`) + map (`/map`) + TF for RViz — proof the engine runs inside ROS 2 Jazzy.

```bash
# build range_libc for the ROS python (3.12) first:
(cd range_libc/pywrapper && WITH_CUDA=OFF /usr/bin/python3 setup.py build_ext --inplace)
examples/run_demo.sh rm f          # backend=rm, map=f   (try: pcddt / glt / lut ; test / f)
```
`examples/rviz_demo.py` is a plain rclpy node — the same `RaycastEngine.scan(pose, n, fov)`
call is what replaces numba `ScanSimulator2D` in `f1tenth_gym_ros`.

## Reproduce
See [`SETUP.md`](SETUP.md). Summary:
```bash
# deps: creating_autonomous_car's f110_gym / slam/range_libc / stack_master/maps
export CAC_DIR=/path/to/creating_autonomous_car      # default is the HMCL desktop path
$VENV/python benchmarks/bench_raycast.py             # MAP=test|f
$VENV/python benchmarks/bench_jax.py                 # JAX_PLATFORMS=cpu for CPU
$VENV/python benchmarks/bench_workload.py            # SIM vs PF real-time check
```

## References
- range_libc / CDDT: C. Walsh, S. Karaman, *"CDDT: Fast Approximate 2D Ray Casting for Accelerated Localization,"* ICRA 2018. [arXiv:1705.01167](https://arxiv.org/abs/1705.01167) · [kctess5/range_libc](https://github.com/kctess5/range_libc) · [f1tenth/range_libc](https://github.com/f1tenth/range_libc)
- F1TENTH gym: O'Kelly et al., NeurIPS 2019. [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) · [f1tenth/f1tenth_gym_jax](https://github.com/f1tenth/f1tenth_gym_jax)
- JAX: Bradbury et al., 2018. [google/jax](https://github.com/google/jax)
- Sphere tracing: J. C. Hart, *The Visual Computer*, 1996. · Bresenham: IBM Sys. J., 1965. · Z-buffer: Catmull, 1974.
