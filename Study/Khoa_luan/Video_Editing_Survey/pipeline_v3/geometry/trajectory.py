"""
Camera trajectory generation and pose transformations for Pipeline v3.
Supports spherical coordinate projection (theta, phi, radius) and Euclidean translation (dx, dy).
"""

import copy
import torch
import numpy as np
from typing import Optional, Union, Tuple


def sphere2pose(
    c2ws_input: torch.Tensor,
    theta: float,
    phi: float,
    r: float,
    device: Union[str, torch.device],
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> torch.Tensor:
    """
    Transforms camera-to-world (c2w) matrices using spherical coordinate shifts and Euclidean translations.
    
    Args:
        c2ws_input: (N, 4, 4) tensor of initial camera poses
        theta: pitch angle in degrees (rotation around X)
        phi: yaw angle in degrees (rotation around Y - Pan)
        r: camera distance shift along Z
        device: target torch device
        x: lateral translation
        y: vertical translation
    Returns:
        (N, 4, 4) transformed camera-to-world matrices
    """
    c2ws = copy.deepcopy(c2ws_input).to(device)

    # Translate along Z
    c2ws[:, 2, 3] -= r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] -= x

    # Rotation around X axis (Pitch)
    theta_rad = torch.deg2rad(torch.tensor(theta, dtype=torch.float32, device=device))
    sin_x = torch.sin(theta_rad)
    cos_x = torch.cos(theta_rad)
    rot_mat_x = (
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos_x, -sin_x, 0.0],
                [0.0, sin_x, cos_x, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
    )

    # Rotation around Y axis (Yaw / Pan)
    phi_rad = torch.deg2rad(torch.tensor(phi, dtype=torch.float32, device=device))
    sin_y = torch.sin(phi_rad)
    cos_y = torch.cos(phi_rad)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_y, 0.0, sin_y, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sin_y, 0.0, cos_y, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
    )

    c2ws = torch.matmul(c2ws, rot_mat_x)
    c2ws = torch.matmul(c2ws, rot_mat_y)
    return c2ws


def generate_camera_trajectory(
    c2ws_anchor: torch.Tensor,
    theta: float = 0.0,
    phi: float = 0.0,
    d_r: float = 0.0,
    d_x: float = 0.0,
    d_y: float = 0.0,
    num_frames: int = 49,
    device: Union[str, torch.device] = "cuda",
    mode: str = "gradual",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates source and target camera trajectory sequences.

    Modes:
      - 'gradual': Camera moves smoothly from 0 to target pose over num_frames.
      - 'direct': Camera instantly jumps to target pose (or anchor).
      - 'bullet': Fixed freeze-time trajectory.

    Returns:
        pose_s: (num_frames, 4, 4) source poses (usually identity or static initial pose)
        pose_t: (num_frames, 4, 4) target camera trajectory poses
    """
    if mode == "gradual":
        thetas = np.linspace(0, theta, num_frames)
        phis = np.linspace(0, phi, num_frames)
        rs = np.linspace(0, d_r, num_frames)
        xs = np.linspace(0, d_x, num_frames)
        ys = np.linspace(0, d_y, num_frames)
    elif mode == "direct":
        thetas = np.full(num_frames, theta)
        phis = np.full(num_frames, phi)
        rs = np.full(num_frames, d_r)
        xs = np.full(num_frames, d_x)
        ys = np.full(num_frames, d_y)
    else:
        thetas = np.linspace(0, theta, num_frames)
        phis = np.linspace(0, phi, num_frames)
        rs = np.linspace(0, d_r, num_frames)
        xs = np.linspace(0, d_x, num_frames)
        ys = np.linspace(0, d_y, num_frames)

    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            float(th),
            float(ph),
            float(r),
            device,
            float(x),
            float(y),
        )
        c2ws_list.append(c2w_new)

    pose_t = torch.cat(c2ws_list, dim=0)
    pose_s = c2ws_anchor.repeat(num_frames, 1, 1)
    return pose_s, pose_t
