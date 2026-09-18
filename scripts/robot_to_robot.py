"""
Robot-to-robot retargeting for GMR.

Instead of loading an SMPL-X file and letting GMR turn it into a canonical
{body_name: (pos, quat)} dict per frame, this script produces that SAME
canonical dict by running forward kinematics on a SOURCE ROBOT's own saved
motion (e.g. a Unitree G1 trajectory) and reading off the poses of the
robot links that correspond to each canonical (human/SMPL-X) body name.

------------------------------------------------------------------------
HOW THIS WORKS, given GeneralMotionRetargeting.__init__:

    with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
        ik_config = json.load(f)
    ...
    self.ik_match_table1 = ik_config["ik_match_table1"]   # {frame_name: [body_name, pos_w, rot_w, pos_off, rot_off]}
    ...
    task = mink.FrameTask(frame_name=frame_name, frame_type="body", ...)
    self.human_body_to_task1[body_name] = task

Two things fall out of that:

1. `src_human` only selects which JSON config to load (IK_CONFIG_DICT[src_human][tgt_robot]).
   It is NOT a "loader mode" switch. Retargeting itself only ever consumes
   `human_data`, a plain {canonical_body_name: (pos, quat_wxyz)} dict, and
   matches its keys against `body_name` entries in the config. So to
   retarget FROM a robot, we still call GMR with src_human="smplx" (the
   canonical namespace) and tgt_robot=<target robot> -- we just build the
   human_data dict ourselves from the source robot's FK instead of from
   an SMPL-X file. GMR never needs to know the data actually came from a
   robot.

2. `frame_name` in the JSON config is a body name in the TARGET robot's
   own MJCF (mink's FrameTask uses frame_type="body"), and it maps to a
   canonical `body_name`. So IK_CONFIG_DICT["smplx"][source_robot] --
   the exact same config GMR would use if source_robot were a normal
   retargeting target -- already contains a
   {source_robot_body_name: canonical_body_name} correspondence. We just
   invert it to get {canonical_body_name: source_robot_body_name}, then
   read the source robot's own FK results through that mapping.

That's the whole trick: reuse GMR's existing smplx -> source_robot config
in reverse to build canonical keypoints, then feed them into a completely
normal smplx -> target_robot GMR instance.

------------------------------------------------------------------------
THINGS TO VERIFY AGAINST YOUR ACTUAL GMR INSTALL (marked "VERIFY" below):

1. CANONICAL_NAMESPACE below is set to "smplx". Confirm this is the exact
   key your install's IK_CONFIG_DICT uses for the human/SMPL-X side, e.g.:
       from general_motion_retargeting.params import IK_CONFIG_DICT
       print(list(IK_CONFIG_DICT.keys()))
   If GMR also supports other source namespaces (bvh, fbx, ...), use
   whichever one actually has a config for your source_robot as a target.

2. root_joint_is_free assumes the source robot's MJCF root is a 7-dof
   free joint at qpos[0:7] (3 pos + 4 quat), with dof_pos filling qpos[7:]
   in the model's own joint order. If a source robot's model differs
   (fixed base, 6-dof joint, etc.) you'll need a per-robot qpos-slicing
   rule -- the code below raises NotImplementedError in that case rather
   than silently writing to the wrong slots.

3. dof_pos in the saved .pkl must already be in the SOURCE robot's own
   MuJoCo qpos/joint order (same requirement as before -- unchanged).
------------------------------------------------------------------------
"""

import argparse
import pathlib
import os
import time
import json
import pickle

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT

from rich import print

ROBOT_CHOICES = [
    "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
    "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
    "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
    "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
    "tienkung", "fourier_gr3", "pal_kangaroo", "pal_kangaroo_lower_body",
    "pal_kangaroo_lower_body_new_ankle", "pal_kangaroo_hands",
]

# VERIFY (see point 1 above): the IK_CONFIG_DICT key used for the
# human/SMPL-X side of every config. Retargeting always keys off this
# namespace's body names as the canonical vocabulary, regardless of
# whether the actual source of a given frame is SMPL-X or (as here) a
# robot's own FK.
CANONICAL_NAMESPACE = "smplx"


def load_robot_motion_file(path):
    """
    Load a saved SOURCE robot motion .pkl.

    Expected schema (matches what this repo's own smplx_to_robot.py
    --save_path writes out):
        {
          "fps": int,
          "root_pos": (T, 3) float array,
          "root_rot": (T, 4) float array, xyzw quaternion order,
          "dof_pos":  (T, n_dof) float array, in the source robot's own
                      joint ordering,
        }

    If your source data instead comes from something like a G1
    MuJoCo-compatible CSV, write a small converter that reshapes it into
    this same dict before calling this script -- the rest of the
    pipeline doesn't care where root_pos/root_rot/dof_pos originally came
    from, only that they're in the source robot's own qpos layout.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _load_ik_config(namespace, robot):
    if namespace not in IK_CONFIG_DICT or robot not in IK_CONFIG_DICT[namespace]:
        raise KeyError(
            f"No IK config for IK_CONFIG_DICT['{namespace}']['{robot}']. "
            f"Available source namespaces: {list(IK_CONFIG_DICT.keys())}. "
            f"Available targets for '{namespace}': "
            f"{list(IK_CONFIG_DICT.get(namespace, {}).keys())}."
        )
    with open(IK_CONFIG_DICT[namespace][robot]) as f:
        return json.load(f)


def get_source_body_mapping(source_robot):
    """
    Returns {canonical_body_name: {"frame_name": ..., "pos_offset": ...,
    "rot_offset": ...}}, built from
    IK_CONFIG_DICT[CANONICAL_NAMESPACE][source_robot] -- the exact same
    config GMR loads when source_robot is a normal retargeting *target*.
    In that config, each row is:

        frame_name -> [body_name, pos_weight, rot_weight, pos_offset, rot_offset]

    where frame_name is a body name in source_robot's own MJCF, body_name
    is the canonical name it corresponds to, and pos_offset/rot_offset are
    the structural-calibration offset GMR's offset_human_data() bakes in
    when using this robot as a retarget *target*:

        target_quat = human_quat * rot_offset
        target_pos  = human_pos + R(target_quat).apply(pos_offset - ground)

    We keep frame_name AND the offsets here (rather than just inverting
    frame_name -> body_name) because reading a source robot's raw FK pose
    and calling it "canonical" without undoing this offset produces a
    systematically wrong pose for every body that has a nonzero offset --
    limbs come out shifted/rotated relative to where the canonical
    skeleton would actually be. robot_frame_to_canonical_dict() undoes it.

    Rows are included even with zero pos_weight/rot_weight: a row being
    unweighted just means it isn't an active IK task when this robot is a
    target, it's still a valid frame_name<->body_name<->offset
    correspondence, and dropping it only loses coverage for no benefit.
    """
    cfg = _load_ik_config(CANONICAL_NAMESPACE, source_robot)
    ground = cfg.get("ground_height", 0.0) * np.array([0.0, 0.0, 1.0])

    body_mapping = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for frame_name, entry in cfg.get(table_name, {}).items():
            body_name, _pos_weight, _rot_weight, pos_offset, rot_offset = entry
            # If both tables define the same canonical body, last one
            # wins -- mirrors GMR's own behavior of just overwriting the
            # per-body task target when both tables are active.
            body_mapping[body_name] = {
                "frame_name": frame_name,
                "pos_offset": np.array(pos_offset, dtype=float) - ground,
                "rot_offset": R.from_quat(rot_offset, scalar_first=True),
            }

    # Make sure the root is always included even if it wasn't already
    # picked up via a table entry. No calibration offset is known for it
    # in that case, so treat it as identity (no shift/rotation).
    human_root_name = cfg.get("human_root_name")
    robot_root_name = cfg.get("robot_root_name")
    if human_root_name and robot_root_name and human_root_name not in body_mapping:
        body_mapping[human_root_name] = {
            "frame_name": robot_root_name,
            "pos_offset": np.zeros(3),
            "rot_offset": R.identity(),
        }

    return body_mapping


def warn_on_missing_coverage(body_mapping, target_robot):
    """
    Preflight check: does the source robot's mapping cover every
    canonical body the TARGET config actually needs? If not, GMR.retarget
    will KeyError partway through the first frame -- better to fail loud
    and early with a readable message.
    """
    tgt_cfg = _load_ik_config(CANONICAL_NAMESPACE, target_robot)
    needed = set()
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for _frame_name, entry in tgt_cfg.get(table_name, {}).items():
            body_name, pos_weight, rot_weight, *_ = entry
            if pos_weight or rot_weight:
                needed.add(body_name)
    human_root_name = tgt_cfg.get("human_root_name")
    if human_root_name:
        needed.add(human_root_name)

    missing = needed - set(body_mapping.keys())
    if missing:
        print(
            f"[yellow]Warning:[/yellow] source robot's canonical mapping is "
            f"missing bodies the target config needs: {sorted(missing)}. "
            f"These canonical keys won't exist in the per-frame dict, and "
            f"GMR.retarget() will raise a KeyError on them."
        )


def build_mj_model_and_data(robot_name):
    """
    Loads the SOURCE robot's MJCF the same way GMR loads it for the
    target side -- straight from ROBOT_XML_DICT, so there's no risk of
    pointing at a different/stale copy of the model.
    """
    if robot_name not in ROBOT_XML_DICT:
        raise KeyError(
            f"'{robot_name}' not in ROBOT_XML_DICT. "
            f"Available: {list(ROBOT_XML_DICT.keys())}."
        )
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_name]))
    data = mujoco.MjData(model)
    return model, data


def robot_frame_to_canonical_dict(model, data, root_pos, root_rot_xyzw, dof_pos, body_mapping):
    """
    Runs forward kinematics for one frame of the SOURCE robot and reads
    off the canonical keypoints, in the same {body_name: (pos, quat_wxyz)}
    schema GMR's own scale_human_data()/offset_human_data() expect
    (they call R.from_quat(quat, scalar_first=True), i.e. wxyz).

    Looks up bodies first (every FrameTask in GMR uses frame_type="body"),
    falling back to sites only if a mapped name isn't a body -- some IK
    configs do reference sites for a handful of robots.

    Each body's raw FK pose is then un-offset back into canonical space
    (see get_source_body_mapping's docstring for the forward formula):
        human_quat = robot_quat * rot_offset^-1
        human_pos  = robot_pos  - R(robot_quat).apply(pos_offset)
    Skipping this step is what causes the retargeted robot to spawn in a
    visibly wrong pose -- every body with a nonzero calibration offset
    would otherwise be read as if it had none.
    """
    n_dof = len(dof_pos)

    root_joint_is_free = (
        model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    )
    if not root_joint_is_free:
        # VERIFY (see point 2 in module docstring): this source robot's
        # root joint isn't a standard 7-dof free joint. Add a
        # robot-specific branch here for how root_pos/root_rot map into
        # this model's qpos instead of assuming qpos[0:7].
        raise NotImplementedError(
            "Source robot's root joint is not a free joint (qpos[0:7]); "
            "add a model-specific qpos-slicing rule for its root pose."
        )

    data.qpos[0:3] = root_pos
    data.qpos[3:7] = np.asarray(root_rot_xyzw)[[3, 0, 1, 2]]  # xyzw -> wxyz for mujoco
    data.qpos[7:7 + n_dof] = dof_pos
    mujoco.mj_forward(model, data)

    frame = {}
    for canonical_name, mapping in body_mapping.items():
        robot_ref_name = mapping["frame_name"]

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_ref_name)
        if body_id != -1:
            pos = data.xpos[body_id].copy()
            mat = data.xmat[body_id].reshape(3, 3)
        else:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot_ref_name)
            if site_id == -1:
                raise KeyError(
                    f"'{robot_ref_name}' (mapped from canonical "
                    f"'{canonical_name}') is neither a body nor a site in "
                    f"the source robot's model."
                )
            pos = data.site_xpos[site_id].copy()
            mat = data.site_xmat[site_id].reshape(3, 3)

        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, mat.flatten())
        robot_rot = R.from_quat(quat_wxyz, scalar_first=True)

        # Undo the config's human -> robot calibration offset to recover
        # an approximate canonical pose from this robot body's actual FK
        # pose (inverse of GeneralMotionRetargeting.offset_human_data).
        canonical_quat = (robot_rot * mapping["rot_offset"].inv()).as_quat(scalar_first=True)
        canonical_pos = pos - robot_rot.apply(mapping["pos_offset"])

        frame[canonical_name] = (canonical_pos, canonical_quat)

    return frame


if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_robot_file",
        help="Path to the SOURCE robot's saved motion (.pkl with "
             "root_pos/root_rot/dof_pos), e.g. a G1 motion saved by this "
             "repo's smplx_to_robot.py --save_path, or converted from a "
             "source CSV.",
        type=str,
        required=True,
    )
    parser.add_argument("--source-robot", choices=ROBOT_CHOICES, default="unitree_g1")
    parser.add_argument("--target-robot", choices=ROBOT_CHOICES, default="pal_kangaroo")
    parser.add_argument("--save_path", default=None, help="Path to save the retargeted TARGET robot motion.")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--loop", default=False, action="store_true")
    parser.add_argument("--record_video", default=False, action="store_true")
    parser.add_argument("--rate_limit", default=False, action="store_true")
    args = parser.parse_args()

    # ---- Load source robot motion + model, compute canonical keypoints ----
    source_motion = load_robot_motion_file(args.source_robot_file)
    fps = source_motion.get("fps", 30)
    root_pos_all = source_motion["root_pos"][args.start_frame:args.end_frame]
    root_rot_all = source_motion["root_rot"][args.start_frame:args.end_frame]  # xyzw
    dof_pos_all = source_motion["dof_pos"][args.start_frame:args.end_frame]

    body_mapping = get_source_body_mapping(args.source_robot)
    warn_on_missing_coverage(body_mapping, args.target_robot)
    src_model, src_data = build_mj_model_and_data(args.source_robot)

    canonical_frames = []
    for t in range(len(root_pos_all)):
        canonical_frames.append(
            robot_frame_to_canonical_dict(
                src_model, src_data,
                root_pos_all[t], root_rot_all[t], dof_pos_all[t],
                body_mapping,
            )
        )

    # ---- Standard GMR retargeting: canonical keypoints -> target robot ----
    # actual_human_height normally rescales an SMPL-X body to match the
    # robot; it has no physical meaning here since the "human" is really
    # a robot skeleton already at the right scale, so this is set to None
    # (ratio stays 1.0, no rescaling applied).
    #
    # src_human is CANONICAL_NAMESPACE ("smplx"), NOT args.source_robot --
    # it only selects which IK config to load (paired with tgt_robot), and
    # that config's body_name vocabulary is exactly what our canonical
    # dict above already uses as keys. Passing the robot name here would
    # make GMR look for a config that doesn't exist.
    retarget = GMR(
        actual_human_height=None,
        src_human=CANONICAL_NAMESPACE,
        tgt_robot=args.target_robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.target_robot,
        motion_fps=fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.source_robot}_to_{args.target_robot}_"
                    f"{pathlib.Path(args.source_robot_file).stem}.mp4",
    )

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    i = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    while True:
        if args.loop:
            i = (i + 1) % len(canonical_frames)
        else:
            i += 1
            if i >= len(canonical_frames):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        frame = canonical_frames[i]
        qpos = retarget.retarget(frame)

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )

        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])
        motion_data = {
            "fps": fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    robot_motion_viewer.close()