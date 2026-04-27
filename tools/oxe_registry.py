"""OXE dataset registry: one (or few) dataset per supported embodiment.

Each entry maps an OXE dataset name to:
  - embodiment: one of RobotSeg's 10 supported robot embodiments
  - version: TFDS version string under gs://gresearch/robotics/<name>/<version>
  - rgb_keys: candidate image keys inside steps/observation to try in order
  - state_key: key under steps/observation holding proprio state (for EE)
  - action_key: key under steps holding action
  - state_schema: human-readable note describing what state dims mean.
    Critical for PnP: which dims are EE xyz in the ROBOT BASE frame.
  - fps_note: native frequency if known (for choosing frame_stride).

Notes:
- "WindowX" in the RobotSeg README is WidowX (Bridge).
- State/action semantics are copied from each dataset's RLDS feature docs.
  When unsure, the script dumps the full feature spec on first access so
  you can correct these offline.
"""

OXE_DATASETS = {
    # --- Franka ---
    "taco_play": {
        "embodiment": "Franka",
        "version": "0.1.0",
        "rgb_keys": ["rgb_static", "rgb_gripper"],
        "state_key": "robot_obs",
        "action_key": "rel_actions_world",
        "state_schema": (
            "robot_obs[15]: tcp_xyz(0:3), tcp_euler(3:6), gripper_width(6), "
            "arm_joints(7:14), gripper_action(14). EE xyz for PnP = [0:3]."
        ),
        "fps_note": "~15 Hz",
    },
    # --- Fanuc Mate ---
    "fanuc_manipulation_v2": {
        "embodiment": "Fanuc Mate",
        "version": "0.1.0",
        "rgb_keys": ["image"],
        "state_key": "end_effector_state",
        "action_key": "action",
        "state_schema": (
            "end_effector_state[7]: xyz(0:3), quat_xyzw(3:7). EE xyz = [0:3]."
        ),
        "fps_note": "~10 Hz",
    },
    # --- UR5 ---
    "berkeley_autolab_ur5": {
        "embodiment": "UR5",
        "version": "0.1.0",
        "rgb_keys": ["image", "hand_image"],
        "state_key": "robot_state",
        "action_key": "action",
        "state_schema": (
            "robot_state[15]: joints(0:6), gripper_is_closed(6), "
            "tcp_xyz(7:10), tcp_quat_wxyz(10:14), gripper_state(14). "
            "EE xyz = [7:10]."
        ),
        "fps_note": "5 Hz",
    },
    # --- Kuka iiwa ---
    "kuka": {
        "embodiment": "Kuka iiwa",
        "version": "0.1.0",
        "rgb_keys": ["image"],
        "state_key": None,  # dataset has no proprio state field
        "action_key": "action",
        "state_schema": (
            "No proprio state; action[7] = delta ee pose + gripper. "
            "For PnP you need EE xyz accumulated from actions — NOT RECOMMENDED."
        ),
        "fps_note": "10 Hz",
    },
    # --- Google Everyday Robot ---
    # Per AugE (process_fractal20220817_data): EE pose is published as
    #   observation['base_pose_tool_reached'] = [x, y, z, qw, qx, qy, qz]
    # in the robot base frame (suitable for PnP).
    "fractal20220817_data": {
        "embodiment": "Google Everyday Robot",
        "version": "0.1.0",
        "rgb_keys": ["image"],
        "state_key": "base_pose_tool_reached",
        "action_key": "action",
        "state_schema": (
            "base_pose_tool_reached[7]: xyz(0:3), quat(3:7). "
            "EE xyz for PnP = [0:3]."
        ),
        "fps_note": "3 Hz",
    },
    # --- DROID (Franka) ---
    # DROID publishes cartesian_position (xyz+euler) and joint_position
    # directly under steps.observation. Multiple cameras are available;
    # exterior_image_1_left is fixed third-person.
    "droid": {
        "embodiment": "Franka",
        "version": "1.0.0",
        "rgb_keys": [
            "exterior_image_1_left", "exterior_image_2_left",
            "wrist_image_left",
        ],
        "state_key": "cartesian_position",
        "action_key": "action",
        "state_schema": (
            "cartesian_position[6]: xyz(0:3), euler(3:6). "
            "EE xyz for PnP = [0:3]. (joint_position[7] also in observation.)"
        ),
        "fps_note": "15 Hz",
    },
    # --- MobileALOHA ---
    # Dataset published as part of OXE under this name.
    "aloha_mobile": {
        "embodiment": "MobileALOHA",
        "version": "0.1.0",
        "rgb_keys": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        "state_key": "state",
        "action_key": "action",
        "state_schema": (
            "state[14]: left_arm_joints(0:6), left_gripper(6), "
            "right_arm_joints(7:13), right_gripper(13). "
            "EE xyz requires FK — not directly in state."
        ),
        "fps_note": "50 Hz",
    },
    # --- xArm ---
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "embodiment": "xArm",
        "version": "0.1.0",
        "rgb_keys": ["image"],
        "state_key": "state",
        "action_key": "action",
        "state_schema": (
            "state[7]: ee_xyz(0:3), ee_euler(3:6), gripper(6). EE xyz = [0:3]."
        ),
        "fps_note": "~5 Hz",
    },
    # --- WidowX (Bridge) ---
    "bridge": {
        "embodiment": "WidowX",
        "version": "0.1.0",
        "rgb_keys": ["image_0", "image_1", "image_2", "image_3"],
        "state_key": "state",
        "action_key": "action",
        "state_schema": (
            "state[7]: ee_xyz(0:3), ee_euler(3:6), gripper(6). EE xyz = [0:3]."
        ),
        "fps_note": "5 Hz",
    },
    # --- Sawyer ---
    "roboturk": {
        "embodiment": "Sawyer",
        "version": "0.1.0",
        "rgb_keys": ["front_rgb"],
        "state_key": None,
        "action_key": "action",
        "state_schema": (
            "No proprio state in RLDS. action[7] = delta ee pose + gripper. "
            "For PnP you must integrate actions — NOT RECOMMENDED."
        ),
        "fps_note": "~10 Hz",
    },
    # --- Hello Stretch ---
    "cmu_stretch": {
        "embodiment": "Hello Stretch",
        "version": "0.1.0",
        "rgb_keys": ["image"],
        "state_key": "state",
        "action_key": "action",
        "state_schema": (
            "state[4]: ee_xyz(0:3), gripper(3). EE xyz = [0:3]."
        ),
        "fps_note": "~5 Hz",
    },
}

# A conservative default subset that have usable EE xyz directly in state
# (best for PnP stress-testing). Override with --datasets on the CLI.
DEFAULT_DATASETS_FOR_PNP = [
    "taco_play",
    "fanuc_manipulation_v2",
    "berkeley_autolab_ur5",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "bridge",
    "cmu_stretch",
    "fractal20220817_data",
    "droid",
]
