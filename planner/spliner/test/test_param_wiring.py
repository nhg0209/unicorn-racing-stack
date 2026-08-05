#!/usr/bin/env python3
"""Does every declared parameter actually reach the member the code reads?

static_near_zero_mps, static_promote_sec, static_demote_mps and static_demote_sec were declared,
listed in the yaml, and listed in the startup sync -- and dyn_param_cb had no branch for any of
them. The startup sync goes through that same callback, so the YAML values had never been applied
in the node's life: the four parameters that decide whether a box is treated as STATIC at all were
whatever the constructor happened to set. `ros2 param set` on them returned successful=True and
changed nothing, which is the part that makes it a trap rather than a bug -- it is invisible until
someone tunes one, and today the yaml happens to agree with the code defaults, so nothing is
wrong RIGHT NOW and everything would be wrong the moment either side moved.

This is a static audit of the file, deliberately: it needs no ROS context, and the failure it
guards is exactly "someone added a declare_parameter and forgot the branch".

  ~/miniforge3/envs/unicorn/bin/python3 planner/spliner/test/test_param_wiring.py
"""
import re
from pathlib import Path

NODE = Path(__file__).resolve().parents[1] / "spliner" / "static_avoidance_node.py"
YAML = (Path(__file__).resolve().parents[3]
        / "stack_master" / "config" / "static_avoidance_params.yaml")
SRC = NODE.read_text()

# Parameters deliberately read once at construction rather than through dyn_param_cb: they pick
# what gets CREATED (a publisher, a bag-replay mode), so changing them later would not do anything
# a live reconfigure implies.
EXEMPT = {"from_bag", "measure"}


def declared():
    """Both spellings: declare_parameter('x', ...) and declare_parameters(parameters=[('x', ...)])."""
    out = set(re.findall(r"declare_parameter\(\s*'([a-z0-9_]+)'", SRC))
    for block in re.findall(r"declare_parameters\((.*?)\]\)", SRC, re.S):
        out |= set(re.findall(r"\(\s*'([a-z0-9_]+)'\s*,", block))
    return out


def handled():
    return (set(re.findall(r"elif n == '([a-z0-9_]+)'", SRC))
            | set(re.findall(r"if n == '([a-z0-9_]+)'", SRC)))


def synced():
    m = re.search(r"self\.dyn_param_cb\(self\.get_parameters\(\[(.*?)\]\)\)", SRC, re.S)
    assert m, "could not find the startup parameter sync"
    return set(re.findall(r"'([a-z0-9_]+)'", m.group(1)))


def test_every_declared_parameter_is_applied():
    missing = sorted(declared() - handled() - EXEMPT)
    assert not missing, (
        f"declared with no dyn_param_cb branch: {missing}. The startup sync runs through the same "
        f"callback, so these never take their yaml value, and `ros2 param set` reports success "
        f"while changing nothing.")
    print(f"PASS all {len(declared())} declared parameters have a dyn_param_cb branch")


def test_every_declared_parameter_is_synced_at_startup():
    missing = sorted(declared() - synced() - EXEMPT)
    assert not missing, (
        f"declared but not in the startup sync list: {missing}. A branch that only runs on a live "
        f"`ros2 param set` leaves the yaml value unapplied until someone happens to set it.")
    print(f"PASS all {len(declared())} declared parameters are synced at startup")


def test_the_yaml_only_sets_parameters_the_node_declares():
    import yaml
    data = yaml.safe_load(YAML.read_text())
    block = {}
    for node in data.values():
        if isinstance(node, dict):
            block.update(node.get("ros__parameters", {}) or {})
    unknown = sorted(k for k in block if k not in declared())
    assert not unknown, (
        f"the yaml sets parameters this node never declares: {unknown} -- they are silently "
        f"ignored unless the node runs with automatically_declare_parameters_from_overrides")
    print(f"PASS all {len(block)} yaml keys are declared by the node")


if __name__ == "__main__":
    test_every_declared_parameter_is_applied()
    test_every_declared_parameter_is_synced_at_startup()
    test_the_yaml_only_sets_parameters_the_node_declares()
    print("ALL PASS")
