"""Local copy of trajectory_planning_helpers + prep_track from helper_funcs_glob.

All files are flat in this directory so the planner package is fully self-contained
(no pip install or git submodule required).

The sys.modules alias makes `import trajectory_planning_helpers as tph` inside
each submodule resolve to THIS package automatically — no edits to copied files needed.
"""
import sys

# Register this package as 'trajectory_planning_helpers' BEFORE importing any
# submodule, so their internal `import trajectory_planning_helpers.X` calls
# resolve to local files instead of the pip-installed version.
sys.modules['trajectory_planning_helpers'] = sys.modules[__name__]

# Mirror the original tph import order (dependencies before dependents).
from . import interp_splines
from . import calc_spline_lengths
from . import calc_splines
from . import calc_normal_vectors
from . import normalize_psi
from . import calc_head_curv_an
from . import calc_head_curv_num
from . import calc_t_profile
from . import import_veh_dyn_info
from . import calc_ax_profile
from . import angle3pt
from . import progressbar
from . import calc_vel_profile
from . import calc_vel_profile_brake
from . import spline_approximation
from . import side_of_line
from . import conv_filt
from . import path_matching_global
from . import path_matching_local
from . import get_rel_path_part
from . import create_raceline
from . import iqp_handler
from . import opt_min_curv
from . import opt_shortest_path
from . import interp_track_widths
from . import check_normals_crossing
from . import calc_tangent_vectors
from . import calc_normal_vectors_ahead
from . import import_veh_dyn_info_2
from . import nonreg_sampling
from . import interp_track

# prep_track from helper_funcs_glob/src — uses tph internally, must come after
from . import prep_track
