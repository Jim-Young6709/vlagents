"""vlagents Agent "drp": the DRP / IMPACT goal-conditioned neural motion planner (MPTransformer).

Model loading, observation layout and depth handling follow the DRP deployment wrapper
(DrpPolicy.load / build_obs / infer, DepthMask.apply, to_policy_units). The model code (`nmp`) comes from an
impact checkout, passed as `impact_root` (default ~/drp/impact_eefgoal).

    python -m vlagents start-server drp --host localhost --port 20000 \
        --kwargs '{"checkpoint_path": "~/drp/ckpt/<exp_name>"}'

Contract with the RCS client (examples/inference/franka.py):
  SingleObs.joints                   (7,)    measured joint angles [rad]
  SingleObs.info["goal_joints"]      (7,)    goal joint angles [rad]
  SingleObs.info["intrinsics"]       {slot: 3x3 K of the frame}, needed by pointmap (pm_cam) checkpoints
  SingleObs.cameras["<slot>"]        HxWx3 uint8 RGB
  SingleObs.cameras["<slot>_depth"]  HxW float32 z-depth in metres aligned to the RGB frame (0 = no return)
  slots: front_left, front_right (robot's left/right, both behind the base)
Returns: Act chunk of `chunk_size` absolute joint targets [rad] (no gripper; the client keeps its own gripper cmd).
"""

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from vlagents import register_agent
from vlagents.policies.interface import Act, Agent, Obs

logger = logging.getLogger("drp_agent")

DEFAULT_IMPACT_ROOT = Path.home() / "drp" / "impact_eefgoal"
POLICY_HW = (240, 320)  # the img encoders expect 240x320
SIM_NEAR_M = 0.01  # renderer clipping range [m]
SIM_FAR_M = 3.0
EEF_GOAL_REPRS = ("eef", "eef_elbow", "eef_joint")

# old ImageEncoder weights sat directly on the encoder; the new one nests them under `.trunk`
_LEGACY_KEY = re.compile(r"^(encoders\.[^.]+_(?:rgb|depth))\.(backbone|proj|pos_embedding)(.*)$")


def _upgrade_legacy_keys(state: dict, expected) -> dict:
    if set(state) <= set(expected):
        return state
    return {_LEGACY_KEY.sub(r"\1.trunk.\2\3", k): v for k, v in state.items()}


def panda_flange_pose(q: np.ndarray) -> np.ndarray:
    """panda_link8 in panda_link0 (Craig / modified DH of the Panda, same kinematics as the FR3)."""
    dh = [  # a, d, alpha
        (0.0, 0.333, 0.0),
        (0.0, 0.0, -np.pi / 2),
        (0.0, 0.316, np.pi / 2),
        (0.0825, 0.0, np.pi / 2),
        (-0.0825, 0.384, -np.pi / 2),
        (0.0, 0.0, np.pi / 2),
        (0.088, 0.0, np.pi / 2),
    ]
    T = np.eye(4)
    for (a, d, alpha), theta in zip(dh, np.reshape(q, 7), strict=True):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        T = T @ np.array(
            [[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]]
        )
    return T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.107], [0, 0, 0, 1]])


def goal_ee_pose(goal_angles) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = panda_flange_pose(goal_angles)
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()])


def _resize(img: np.ndarray, hw: tuple[int, int], nearest: bool) -> np.ndarray:
    if img.shape[:2] == hw:
        return img
    from PIL import Image

    h, w = hw
    if nearest:
        return np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize((w, h), Image.NEAREST))
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


class DrpAgent(Agent):
    def __init__(
        self,
        checkpoint_path: str,
        impact_root: str | None = None,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        ckpt = Path(checkpoint_path).expanduser()
        if ckpt.is_dir():
            ckpt = ckpt / "latest.pt"
        super().__init__(default_checkpoint_path=str(ckpt), checkpoint_path=str(ckpt), **kwargs)
        self.ckpt = ckpt
        self.impact_root = Path(impact_root).expanduser() if impact_root else DEFAULT_IMPACT_ROOT
        self.device = device
        self.model = None
        self.chunk_size = None
        self.rgb_slots: list[str] = []
        self.depth_slots: list[str] = []
        self.depth_type: dict[str, str] = {}
        self.depth_scale = 500.0
        self.goal_repr = "joint"
        self.epoch = None

    # ------------------------------------------------------------------ loading (DrpPolicy.load)
    def initialize(self):
        import torch
        from omegaconf import OmegaConf

        if not self.ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.ckpt}")
        if str(self.impact_root) not in sys.path:
            sys.path.insert(0, str(self.impact_root))
        from nmp.model.transformer import MPTransformer, strip_prefix_from_state_dict

        cfg = OmegaConf.to_container(OmegaConf.load(self.ckpt.parent / "model_config.yaml"), resolve=True)
        cfg.pop("_target_", None)
        cfg.pop("defaults", None)
        if any(c.get("use_pcd") for c in (cfg.get("pcd_encoders_cfg") or {}).values()):
            raise ValueError(f"{self.ckpt.parent.name} reads a robot-frame point cloud (needs extrinsics)")
        self.chunk_size = cfg.get("chunk_size")
        self.goal_repr = cfg.get("goal_repr", "joint")
        for key, enc in (cfg.get("img_encoders_cfg") or {}).items():
            if not enc.get("use_img", True):
                continue
            # the checkpoint overwrites every weight; pretrained=True would only download ImageNet weights
            enc["pretrained"] = False
            if key.endswith("_rgb"):
                self.rgb_slots.append(key[:-4])
            elif key.endswith("_depth"):
                slot, kind = key[:-6], enc.get("type", "resnet")
                if kind == "pointmap" and enc.get("frame", "camera") != "camera":
                    raise ValueError(f"{self.ckpt.parent.name} unprojects depth into the robot frame (needs extrinsics)")
                if float(enc.get("depth_clip_m", SIM_FAR_M)) != SIM_FAR_M:
                    raise ValueError(f"{key}: depth_clip_m {enc.get('depth_clip_m')} != {SIM_FAR_M}")
                self.depth_slots.append(slot)
                self.depth_type[slot] = kind
                self.depth_scale = float(enc["depth_scale"])

        model = MPTransformer(**cfg)
        state = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        weights = strip_prefix_from_state_dict(state["model_state_dict"])
        model.load_state_dict(_upgrade_legacy_keys(weights, model.state_dict()))
        self.model = model.to(self.device).float().eval()
        self.epoch = state.get("epoch")
        self._torch = torch
        logger.info(
            "loaded %s (epoch %s) on %s | chunk=%s goal_repr=%s rgb=%s depth=%s depth_scale=%s",
            self.ckpt.parent.name,
            self.epoch,
            self.device,
            self.chunk_size,
            self.goal_repr,
            self.rgb_slots,
            self.depth_type,
            self.depth_scale,
        )

    # ------------------------------------------------------------------ obs (DrpPolicy.build_obs)
    def build_obs(self, q: np.ndarray, goal: np.ndarray, cameras: dict, intrinsics: dict | None) -> dict:
        def batched(a) -> np.ndarray:
            return np.asarray(a, dtype=np.float32)[None, None]

        obs = {"current_angles": batched(np.reshape(q, 7)), "goal_angles": batched(np.reshape(goal, 7))}
        if self.goal_repr in EEF_GOAL_REPRS:
            obs["goal_ee"] = batched(goal_ee_pose(goal))
        for slot in self.rgb_slots:
            rgb = np.asarray(cameras[slot])
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            # 0-255, the encoder divides by 255 and ImageNet-normalises
            obs[f"{slot}_rgb"] = batched(_resize(rgb, POLICY_HW, nearest=False))
        for slot in self.depth_slots:
            depth_m = _resize(np.asarray(cameras[f"{slot}_depth"], dtype=np.float32), POLICY_HW, nearest=True)
            # DepthMask.apply: keep SIM_NEAR <= d < SIM_FAR (non-finite -> 0), everything else means "no return"
            depth_m = np.where(np.isfinite(depth_m) & (depth_m >= SIM_NEAR_M) & (depth_m < SIM_FAR_M), depth_m, 0.0)
            # to_policy_units: metres -> renderer's stored integer (2 mm steps at depth_scale 500)
            obs[f"{slot}_depth"] = batched(np.round(depth_m.astype(np.float64) * self.depth_scale))
            if self.depth_type[slot] == "pointmap":
                if not intrinsics or slot not in intrinsics:
                    raise ValueError(f"{self.ckpt.parent.name} needs intrinsics for slot {slot}")
                obs[f"{slot}_intrinsics"] = batched(np.asarray(intrinsics[slot]).reshape(3, 3))
        return obs

    # ------------------------------------------------------------------ act (DrpPolicy.infer)
    def act(self, obs: Obs) -> Act:
        super().act(obs)  # decodes shared memory / jpeg payloads into numpy
        robot_name, single_obs = self._require_single_arm(obs)
        goal = single_obs.info.get("goal_joints")
        if goal is None:
            raise ValueError("DRP is goal conditioned: SingleObs.info['goal_joints'] is required")
        missing = [s for s in self.rgb_slots if s not in single_obs.cameras] + [
            f"{s}_depth" for s in self.depth_slots if f"{s}_depth" not in single_obs.cameras
        ]
        if missing:
            raise ValueError(f"missing camera frames {missing}, got {sorted(single_obs.cameras)}")
        obs_np = self.build_obs(single_obs.joints, goal, single_obs.cameras, single_obs.info.get("intrinsics"))
        torch = self._torch
        t0 = time.perf_counter()
        with torch.no_grad():
            batch = {k: torch.as_tensor(v).to(self.device) for k, v in obs_np.items()}
            chunk = self.model.get_action(batch)[0].float().cpu().numpy().astype(np.float64)
        ms = (time.perf_counter() - t0) * 1e3
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.all(np.isfinite(chunk)):
            raise ValueError(f"bad action chunk shape={chunk.shape} finite={np.all(np.isfinite(chunk))}")
        if single_obs.info.get("step", 0) == 0:
            logger.info(
                "episode %s: q=%s goal=%s -> chunk[0]-q=%s (%.1f ms)",
                single_obs.info.get("episode"),
                np.round(single_obs.joints, 3).tolist(),
                np.round(goal, 3).tolist(),
                np.round(chunk[0] - single_obs.joints, 3).tolist(),
                ms,
            )
        infos = [{"inference_ms": ms, "epoch": self.epoch}] + [None] * (len(chunk) - 1)
        return self._chunk_act(robot_name, chunk, infos=infos)


register_agent("drp", DrpAgent)
