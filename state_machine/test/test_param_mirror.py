#!/usr/bin/env python3
"""Does every parameter the node READS off itself actually get set on the node?

The state machine keeps its parameters on `self.params` and mirrors the dynamic ones onto itself,
so `self.<name>` is what the loop reads and `parameters_callback` updates. Adding a parameter to
_NODE_MIRRORED_PARAMS without the `self.<name> = self.params.<name>` line leaves the node reading
an attribute that does not exist -- and rclpy lets an AttributeError out of a timer callback, so
the first loop iteration takes the whole state machine down: no /behavior_strategy, no
/local_waypoints, no /ot_section_check, and every planner downstream reporting its own gate stale.

That is not hypothetical; it shipped, and this file is here so it cannot ship twice.

  ~/miniforge3/envs/unicorn/bin/python3 state_machine/test/test_param_mirror.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "state_machine"
NODE = (ROOT / "state_machine_node.py").read_text()
PARAMS = (ROOT / "state_machine_params.py").read_text()


def mirrored():
    m = re.search(r"_NODE_MIRRORED_PARAMS = \{(.*?)\}", PARAMS, re.S)
    assert m, "could not find _NODE_MIRRORED_PARAMS"
    return re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))


def test_every_mirrored_param_is_initialised_on_the_node():
    names = mirrored()
    assert names, "the mirror list is empty -- the regex is wrong"
    missing = [n for n in names
               if f"self.{n} = self.params.{n}" not in NODE
               and f"self.{n} = self.params." not in NODE]
    assert not missing, (
        f"in _NODE_MIRRORED_PARAMS but never initialised on the node: {missing}. "
        f"parameters_callback only assigns when hasattr(node, name) is already true, so these "
        f"stay missing forever and the first read raises.")
    print(f"PASS all {len(names)} mirrored parameters are initialised on the node")


def test_the_node_does_not_read_a_parameter_it_never_sets():
    # The other direction, restricted to the names the params object exposes: anything the node
    # reads as self.<name> must be assigned somewhere in the node.
    declared = set(re.findall(r'self\._declare\(\s*"([A-Za-z0-9_]+)"', PARAMS))
    read = set(re.findall(r"self\.([a-z][A-Za-z0-9_]*)", NODE))
    assigned = set(re.findall(r"self\.([A-Za-z0-9_]+)\s*=", NODE))
    dangling = sorted((declared & read) - assigned)
    assert not dangling, (
        f"the node reads {dangling} off itself but never assigns them -- they live on "
        f"self.params only")
    print(f"PASS the node assigns every declared parameter it reads off itself")


if __name__ == "__main__":
    test_every_mirrored_param_is_initialised_on_the_node()
    test_the_node_does_not_read_a_parameter_it_never_sets()
    print("ALL PASS")
