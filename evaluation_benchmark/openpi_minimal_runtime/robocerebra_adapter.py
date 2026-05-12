# examples/robocerebra/robocerebra_adapter.py
# -*- coding: utf-8 -*-
"""
Adapter utilities to bridge RoboCerebra env observations to pi-0.5 policy inputs.
Matched exactly with DataRecorder logic:
  1. Only np.flipud (No horizontal flip/mirroring)
  2. Force resize (Squish to square, no padding)
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import cv2


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to axis-angle (robosuite-compatible)."""
    quat = quat.astype(np.float64).copy()
    # Normalize quaternion first to avoid numerical errors
    norm = np.linalg.norm(quat)
    if norm > 1e-12:
        quat = quat / norm
    
    # Robosuite convention: w is last [x, y, z, w]
    # Check if we need to flip sign to ensure w is positive (canonical representation)
    if quat[3] < 0:
        quat = -quat

    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    
    # Calculate angle and axis
    angle = 2.0 * np.arccos(quat[3])
    out = (quat[:3] * angle) / den
    return out.astype(np.float32)


def _process_image_match_training(x: np.ndarray, size: int) -> np.ndarray:
    """
    Process image EXACTLY as done in DataRecorder:
    1. np.flipud (Fix Robosuite upside-down)
    2. cv2.resize (Force resize/squish to target size, ignoring aspect ratio)
    """
    # 1. 确保是 HWC 格式
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    
    # 2. 转换为 uint8
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)

    # 3. 🔥 关键修正 A: 只做垂直翻转 (flipud)，不做水平翻转
    # 对应 DataRecorder: img = np.flipud(img)
    x = np.flipud(x)

    # 4. 🔥 关键修正 B: 暴力缩放 (Squish)，不留黑边
    # 对应 DataRecorder: cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    if x.shape[0] != size or x.shape[1] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
        
    return x


def obs_to_pi_element(
    obs: Dict[str, Any],
    resize_size: int,
    prompt: str | None = None,
) -> Dict[str, Any]:
    """
    Build the single policy input element for pi-0.5 server.
    """
    # --- 1. 获取图像 ---
    img_main = obs.get("agentview_image", None)
    if img_main is None:
        img_main = obs.get("agentview_rgb", None)
    if img_main is None:
        raise KeyError("Neither 'agentview_image' nor 'agentview_rgb' found in obs")

    img_wrist = obs.get("robot0_eye_in_hand_image", None)
    if img_wrist is None:
        img_wrist = obs.get("wrist_image", None)
    if img_wrist is None:
        raise KeyError("Neither 'robot0_eye_in_hand_image' nor 'wrist_image' found in obs")

    # --- 2. 获取状态 (Proprio) ---
    eef_pos = obs.get("robot0_eef_pos")
    eef_quat = obs.get("robot0_eef_quat")
    gripper = obs.get("robot0_gripper_qpos")

    # 容错处理
    if eef_pos is None:
        eef_pos = obs.get("eef_pos") or np.zeros(3, dtype=np.float32)
    if eef_quat is None:
        eef_quat = obs.get("eef_quat") or np.array([0, 0, 0, 1], dtype=np.float32)
    if gripper is None:
        gripper = obs.get("gripper_qpos") or np.zeros(1, dtype=np.float32)

    # 拼接状态向量
    state = np.concatenate(
        [np.asarray(eef_pos, dtype=np.float32),
         _quat2axisangle(np.asarray(eef_quat, dtype=np.float32)),
         np.asarray(gripper, dtype=np.float32)]
    )

    # --- 3. 处理图像 (使用修正后的函数) ---
    # 注意：这里不需要传 rotate_180 参数了，因为函数内部已经固定为 flipud
    processed_main = _process_image_match_training(np.asarray(img_main), resize_size)
    processed_wrist = _process_image_match_training(np.asarray(img_wrist), resize_size)

    # --- 4. 构造 OpenPI 需要的字典 ---
    # 为了兼容 pi0 模型的 3 个 image slot，我们手动填充
    element = {
        # 对应 Config 里的映射 (通常 base -> agentview)
        "observation/image": processed_main,
        "observation/wrist_image": processed_wrist,
        
        # 显式提供具体的 key 名字，防止 server 端映射出错
        "base_0_rgb": processed_main,
        "left_wrist_0_rgb": np.zeros_like(processed_wrist), # 填充全黑 (单臂机器人无左手)
        "right_wrist_0_rgb": processed_wrist,               # 假设 eye_in_hand 映射为右手
        
        "observation/state": state,
        "prompt": "" if prompt is None else str(prompt),
    }
    
    return element