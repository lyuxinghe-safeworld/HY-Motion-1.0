#!/usr/bin/env python3
"""Convert HY-Motion NPZ files to ProtoMotions .motion format.

Usage:
    # Single file
    python scripts/convert_npz.py \
        --npz-file output/generated/a_person_walks_forward/a_person_walks_forward.npz \
        --output-dir output/generated/a_person_walks_forward/

    # Directory (batch)
    python scripts/convert_npz.py \
        --npz-dir output/generated/a_batch_run/ \
        --output-dir output/generated/a_batch_run/
"""

from __future__ import annotations

import argparse
import sys
import types
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

DEFAULT_PROTOMOTIONS_ROOT = Path.home() / "code" / "ProtoMotions"
_FOOT_HEIGHT_OFFSET = 0.015  # metres, same as convert_amass_to_proto.py for SMPL
_N_BODY_JOINTS = 22
_N_HAND_JOINTS = 2  # synthetic zero-rotation hands
_N_TOTAL_JOINTS = _N_BODY_JOINTS + _N_HAND_JOINTS  # 24
_N_NON_ROOT_JOINTS = 23
_STRICT_CONTACT_VEL_THRESHOLD = 0.15
_STRICT_CONTACT_HEIGHT_THRESHOLD = 0.1
_LENIENT_CONTACT_VEL_THRESHOLD = 0.8
_LENIENT_CONTACT_HEIGHT_THRESHOLD = 0.15


class _MjcfJoint:
    def __init__(
        self,
        name: str | None,
        joint_type: str,
        axis: list[float] | None,
        joint_range: list[float] | None,
    ) -> None:
        self.name = name
        self.type = joint_type
        self.axis = axis
        self.range = joint_range


class _MjcfBody:
    def __init__(
        self,
        name: str | None,
        pos: list[float] | None,
        quat: list[float] | None,
        joints: list[_MjcfJoint],
        freejoint: object | None,
        bodies: list["_MjcfBody"],
    ) -> None:
        self.name = name
        self.pos = pos
        self.quat = quat
        self.joint = joints
        self.freejoint = freejoint
        self.body = bodies


class _MjcfCompiler:
    def __init__(self, angle: str | None = None) -> None:
        self.angle = angle


class _MjcfWorldBody:
    def __init__(self, bodies: list[_MjcfBody]) -> None:
        self.body = bodies


class _MjcfModel:
    def __init__(self, compiler: _MjcfCompiler, worldbody: _MjcfWorldBody) -> None:
        self.compiler = compiler
        self.worldbody = worldbody


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Convert HY-Motion NPZ files to ProtoMotions .motion format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--npz-file",
        type=str,
        help="Path to a single HY-Motion .npz file",
    )
    group.add_argument(
        "--npz-dir",
        type=str,
        help="Path to a directory of HY-Motion .npz files (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for .motion files (default: output/)",
    )
    parser.add_argument(
        "--protomotions-root",
        type=str,
        default=str(DEFAULT_PROTOMOTIONS_ROOT),
        help="Path to ProtoMotions repo root (default: %(default)s)",
    )
    return parser.parse_args(argv)


def resolve_conversion_jobs(
    npz_file: str | None,
    npz_dir: str | None,
    output_dir: str,
) -> list[tuple[Path, Path]]:
    output_root = Path(output_dir)
    if npz_file is not None:
        npz_paths = [Path(npz_file)]
    else:
        input_dir = Path(npz_dir) if npz_dir is not None else None
        if input_dir is None:
            raise ValueError("Either npz_file or npz_dir must be provided.")
        npz_paths = sorted(input_dir.glob("*.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"No .npz files found in {input_dir}")

    return [
        (npz_path, output_root / f"{npz_path.stem}.motion")
        for npz_path in npz_paths
    ]


def prepare_protomotions_imports(protomotions_root: Path) -> None:
    if not protomotions_root.exists():
        raise FileNotFoundError(
            f"ProtoMotions root does not exist: {protomotions_root}"
        )

    path_entries = [
        protomotions_root,
        protomotions_root / "data" / "scripts",
    ]
    for entry in reversed(path_entries):
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)


def _parse_float_list(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    return [float(item) for item in raw_value.split()]


def _compute_contact_labels_with_fallback(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    compute_contact_labels,
) -> torch.Tensor:
    strict_contacts = compute_contact_labels(
        positions=positions,
        velocity=velocity,
        vel_thres=_STRICT_CONTACT_VEL_THRESHOLD,
        height_thresh=_STRICT_CONTACT_HEIGHT_THRESHOLD,
    )
    if strict_contacts.any():
        return strict_contacts.to(torch.bool)

    warnings.warn(
        "No contact candidates found with strict thresholds; "
        "retrying with lenient thresholds (height<0.15, vel<0.8).",
        RuntimeWarning,
        stacklevel=2,
    )
    lenient_contacts = compute_contact_labels(
        positions=positions,
        velocity=velocity,
        vel_thres=_LENIENT_CONTACT_VEL_THRESHOLD,
        height_thresh=_LENIENT_CONTACT_HEIGHT_THRESHOLD,
    )
    return lenient_contacts.to(torch.bool)


def _parse_mjcf_joint(joint_element: ET.Element) -> _MjcfJoint:
    return _MjcfJoint(
        name=joint_element.get("name"),
        joint_type=joint_element.get("type", "hinge"),
        axis=_parse_float_list(joint_element.get("axis")),
        joint_range=_parse_float_list(joint_element.get("range")),
    )


def _parse_mjcf_body(body_element: ET.Element) -> _MjcfBody:
    return _MjcfBody(
        name=body_element.get("name"),
        pos=_parse_float_list(body_element.get("pos")),
        quat=_parse_float_list(body_element.get("quat")),
        joints=[
            _parse_mjcf_joint(joint_element)
            for joint_element in body_element.findall("joint")
        ],
        freejoint=object() if body_element.find("freejoint") is not None else None,
        bodies=[
            _parse_mjcf_body(child_body)
            for child_body in body_element.findall("body")
        ],
    )


def _load_mjcf_model(mjcf_path: str | Path) -> _MjcfModel:
    root = ET.parse(mjcf_path).getroot()
    compiler_element = root.find("compiler")
    worldbody_element = root.find("worldbody")
    if worldbody_element is None:
        raise ValueError(f"MJCF has no <worldbody>: {mjcf_path}")

    return _MjcfModel(
        compiler=_MjcfCompiler(
            angle=compiler_element.get("angle") if compiler_element is not None else None
        ),
        worldbody=_MjcfWorldBody(
            bodies=[
                _parse_mjcf_body(body_element)
                for body_element in worldbody_element.findall("body")
            ]
        ),
    )


def install_pose_lib_import_shims() -> None:
    """Provide lightweight optional deps so ProtoMotions pose_lib can import."""
    try:
        __import__("dm_control")
    except ModuleNotFoundError:
        dm_control_module = types.ModuleType("dm_control")
        dm_control_module.__path__ = []
        mjcf_module = types.ModuleType("dm_control.mjcf")
        mjcf_module.from_path = _load_mjcf_model
        mjcf_module.__package__ = "dm_control"
        dm_control_module.mjcf = mjcf_module
        sys.modules["dm_control"] = dm_control_module
        sys.modules["dm_control.mjcf"] = mjcf_module

    try:
        __import__("mujoco")
    except ModuleNotFoundError:
        sys.modules["mujoco"] = types.ModuleType("mujoco")


def convert_hymotion_npz(
    npz_path: str | Path,
    output_path: str | Path,
    protomotions_root: str | Path,
    device: str = "cpu",
    dtype=None,
) -> None:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as sRot

    if dtype is None:
        dtype = torch.float32

    protomotions_root = Path(protomotions_root)
    prepare_protomotions_imports(protomotions_root)
    install_pose_lib_import_shims()

    from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
    from protomotions.components.pose_lib import (
        compute_angular_velocity,
        compute_forward_kinematics_from_transforms,
        compute_joint_rot_mats_from_global_mats,
        extract_kinematic_info,
        extract_qpos_from_transforms,
        fk_from_transforms_with_velocities,
    )
    from protomotions.utils.rotations import (
        matrix_to_quaternion,
        quat_mul,
        quaternion_to_matrix,
    )

    npz_path = Path(npz_path)
    output_path = Path(output_path)

    npz_data = np.load(npz_path)
    poses = npz_data["poses"]
    amass_trans = npz_data["trans"]
    if "mocap_framerate" in npz_data:
        output_fps = int(np.round(npz_data["mocap_framerate"]))
    else:
        output_fps = 30

    n_frames = poses.shape[0]
    if n_frames < 2:
        raise ValueError(f"Motion too short: {n_frames} frames, need >= 2")

    pose_aa = np.concatenate(
        [poses[:, :66], np.zeros((n_frames, 6))],
        axis=1,
    )

    smpl_to_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(joint_name)
        for joint_name in SMPL_MUJOCO_NAMES
        if joint_name in SMPL_BONE_ORDER_NAMES
    ]
    pose_aa_mj = pose_aa.reshape(n_frames, _N_TOTAL_JOINTS, 3)[:, smpl_to_mujoco]
    pose_quat = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(n_frames, _N_TOTAL_JOINTS, 4)
    )

    amass_trans = torch.from_numpy(amass_trans).to(device, dtype)
    pose_quat = torch.from_numpy(pose_quat).to(device, dtype)
    local_rot_mats = quaternion_to_matrix(pose_quat, w_last=True)

    mjcf_path = protomotions_root / "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    kinematic_info = extract_kinematic_info(str(mjcf_path))

    _, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, amass_trans, local_rot_mats
    )
    global_quat = matrix_to_quaternion(world_rot_mat, w_last=True)

    rot1 = sRot.from_euler(
        "xyz", np.array([-np.pi / 2, -np.pi / 2, 0]), degrees=False
    )
    rot1_quat = torch.from_numpy(rot1.as_quat()).to(device, dtype).expand(n_frames, -1)

    rot_y2z = sRot.from_euler("x", np.pi / 2, degrees=False)
    rot_y2z_quat = (
        torch.from_numpy(rot_y2z.as_quat()).to(device, dtype).expand(n_frames, -1)
    )

    for joint_index in range(_N_TOTAL_JOINTS):
        global_quat[:, joint_index, :] = quat_mul(
            global_quat[:, joint_index, :], rot1_quat, w_last=True
        )
        global_quat[:, joint_index, :] = quat_mul(
            rot_y2z_quat, global_quat[:, joint_index, :], w_last=True
        )

    rot_y2z_mat = torch.from_numpy(rot_y2z.as_matrix()).to(device, dtype)
    amass_trans = (rot_y2z_mat @ amass_trans.unsqueeze(-1)).squeeze(-1)

    local_rot_mats_rotated = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    pose_quat_rotated = matrix_to_quaternion(local_rot_mats_rotated, w_last=True)
    motion.local_rigid_body_rot = pose_quat_rotated.clone()

    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]

    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats_rotated[:, 1:, :, :],
        fps=output_fps,
    )
    if local_angular_vels.shape[1] != _N_NON_ROOT_JOINTS:
        raise ValueError(
            "Unexpected local angular velocity shape: "
            f"{local_angular_vels.shape}"
        )
    motion.dof_vel = local_angular_vels.reshape(-1, _N_NON_ROOT_JOINTS * 3)

    motion.fix_height(height_offset=_FOOT_HEIGHT_OFFSET)

    from contact_detection import compute_contact_labels_from_pos_and_vel

    motion.rigid_body_contacts = _compute_contact_labels_with_fallback(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        compute_contact_labels=compute_contact_labels_from_pos_and_vel,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), str(output_path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protomotions_root = Path(args.protomotions_root)

    try:
        jobs = resolve_conversion_jobs(
            npz_file=args.npz_file,
            npz_dir=args.npz_dir,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"Converting {len(jobs)} file(s) to {output_dir}/")

    had_error = False
    for npz_path, out_path in jobs:
        print(f"  {npz_path.name} -> {out_path.name}")
        try:
            convert_hymotion_npz(
                npz_path=npz_path,
                output_path=out_path,
                protomotions_root=protomotions_root,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            had_error = True
            continue

    print("Done.")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
