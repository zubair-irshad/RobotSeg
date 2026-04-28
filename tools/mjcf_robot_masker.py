"""MuJoCo Menagerie–style robot silhouette renderer.

Mirrors the URDFRobotMasker API but uses a MuJoCo MJCF for kinematics +
mesh assets. The advantage over URDF: Menagerie models follow the real
robot controller's joint conventions, so dataset joint angles can be
fed directly into qpos without sign flips, π/2 offsets, or guessing
which UR/ROS variant the URDF came from.

Reference: AugE-Toolkit's `robot_xml/universal_robots_ur5e/ur5e.xml`,
known to work with Berkeley AutoLab UR5 joint angles via MuJoCo qpos.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class MuJoCoRobotMasker:
    """Rasterize a robot silhouette from an MJCF + dataset joint angles.

    Parameters
    ----------
    mjcf_path : path to the .xml MJCF (e.g. AugE's ur5e.xml).
    arm_dof : number of arm joints to set from joint_positions (the rest
        of qpos is left at the MJCF's defaults / mimic-driven). For UR5e
        this is 6; Franka 7; xArm7 7.
    downsample : rasterize at 1/s and nearest-upsample (s>=1).
    dilate_px : grow the mask by N pixels after rasterization.
    """

    def __init__(self, mjcf_path: str | Path,
                 arm_dof: int = 6,
                 downsample: int = 2,
                 dilate_px: int = 2,
                 verbose: bool = True):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.arm_dof = int(arm_dof)
        self.downsample = max(1, int(downsample))
        self.dilate_px = int(dilate_px)
        # Pre-cache per-geom mesh data + the body each geom hangs off.
        # We only render visual geoms (group 0/1/2 by Menagerie convention).
        self._geoms = self._collect_geoms()
        if verbose:
            n_bodies = len({g["body_id"] for g in self._geoms})
            print(f"[mjcf_masker] loaded {mjcf_path}")
            print(f"[mjcf_masker] {len(self._geoms)} visual mesh geoms over "
                  f"{n_bodies} bodies; arm_dof={self.arm_dof}; nq={self.model.nq}")

    def _collect_geoms(self) -> list[dict]:
        mj = self._mj
        out: list[dict] = []
        for gid in range(self.model.ngeom):
            if self.model.geom_type[gid] != mj.mjtGeom.mjGEOM_MESH:
                continue
            # Skip explicitly-collision-only geoms; Menagerie usually puts
            # collision in group 3.
            if int(self.model.geom_group[gid]) >= 3:
                continue
            mesh_id = int(self.model.geom_dataid[gid])
            v0 = int(self.model.mesh_vertadr[mesh_id])
            vn = int(self.model.mesh_vertnum[mesh_id])
            f0 = int(self.model.mesh_faceadr[mesh_id])
            fn = int(self.model.mesh_facenum[mesh_id])
            if vn == 0 or fn == 0:
                continue
            verts = np.asarray(self.model.mesh_vert[v0:v0 + vn],
                               dtype=np.float64).reshape(-1, 3).copy()
            faces = np.asarray(self.model.mesh_face[f0:f0 + fn],
                               dtype=np.int32).reshape(-1, 3).copy()
            # Geom pose within its parent body:
            T_geom_in_body = np.eye(4)
            T_geom_in_body[:3, 3] = self.model.geom_pos[gid]
            R = np.zeros(9, dtype=np.float64)
            mj.mju_quat2Mat(R, self.model.geom_quat[gid])
            T_geom_in_body[:3, :3] = R.reshape(3, 3)
            out.append({
                "body_id": int(self.model.geom_bodyid[gid]),
                "verts": verts,
                "faces": faces,
                "T_geom_in_body": T_geom_in_body,
            })
        return out

    def _set_joints(self, q_arm: np.ndarray, gripper_position: float) -> None:
        n = min(self.arm_dof, len(q_arm), self.model.nq)
        if n > 0:
            self.data.qpos[:n] = q_arm[:n]
        # Gripper handling: MuJoCo Menagerie's 2f-85 is tendon-driven via
        # an actuator named "fingers_actuator" with ctrl in [0,1]. We set
        # ctrl from gripper_position (0=open, 1=closed) so the simulated
        # actuator pulls the fingers; one mj_forward propagates through
        # the equality constraints / mimics.
        try:
            actuator_id = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
            )
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = float(np.clip(gripper_position, 0, 1))
        except Exception:
            pass
        self._mj.mj_forward(self.model, self.data)

    def render(self, K: dict, T_cam2base: np.ndarray,
               joint_positions: np.ndarray,
               gripper_position: float = 0.0,
               image_hw: tuple[int, int] | None = None) -> np.ndarray:
        H = int(image_hw[0] if image_hw is not None else K["height"])
        W = int(image_hw[1] if image_hw is not None else K["width"])
        s = self.downsample
        Hs, Ws = H // s, W // s
        fx, fy = K["fx"] / s, K["fy"] / s
        cx, cy = K["cx"] / s, K["cy"] / s

        q_arm = np.asarray(joint_positions, dtype=float).flatten()
        self._set_joints(q_arm, gripper_position)

        T_base2cam = np.linalg.inv(np.asarray(T_cam2base, dtype=float))

        mask = np.zeros((Hs, Ws), dtype=np.uint8)
        for g in self._geoms:
            bid = g["body_id"]
            T_body_world = np.eye(4)
            T_body_world[:3, 3] = self.data.xpos[bid]
            T_body_world[:3, :3] = self.data.xmat[bid].reshape(3, 3)
            T_geom_world = T_body_world @ g["T_geom_in_body"]
            T_geom_cam = T_base2cam @ T_geom_world

            verts = g["verts"]
            vh = np.hstack([verts, np.ones((len(verts), 1))])
            vc = (T_geom_cam @ vh.T).T[:, :3]

            z = vc[:, 2]
            if (z <= 0.05).all():
                continue
            zsafe = np.maximum(z, 1e-6)
            u = fx * vc[:, 0] / zsafe + cx
            v = fy * vc[:, 1] / zsafe + cy

            tri = g["faces"]
            keep = ((z[tri[:, 0]] > 0.05)
                    & (z[tri[:, 1]] > 0.05)
                    & (z[tri[:, 2]] > 0.05))
            tri = tri[keep]
            if len(tri) == 0:
                continue
            poly = np.stack([u[tri], v[tri]], axis=-1).astype(np.int32)
            cv2.fillPoly(mask, poly, color=1)

        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.dilate_px > 0:
            k = 2 * self.dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask.astype(bool)
