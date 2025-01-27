from typing import Tuple
import numpy as np

def euler_to_homo(Xe: np.array) -> np.array:
    """
    convert euler to homogenous coordinates
    """
    num_points = np.shape(Xe)[0]
    Xh = np.column_stack((Xe, np.ones(num_points)))
    return Xh


def homo_to_euler(Xh: np.array) -> np.array:
    """
    convert homogenous to euler coordinates
    """
    Xe = np.column_stack(
        (Xh[:, 0] / Xh[:, 3],
         Xh[:, 1] / Xh[:, 3],
         Xh[:, 2] / Xh[:, 3])
    )
    return Xe


def euler_to_lin_rot(alpha1: float, alpha2: float,
                     alpha3: float) -> np.array:
    """
    compute linearized rotation matrix from
    euler angles
    """
    R = np.array([[1, -alpha3, alpha2],
                  [alpha3, 1, -alpha1],
                  [-alpha2, alpha1, 1]])
    return R

def euler_to_rot(alpha1: float, alpha2:float,
                 alpha3: float) -> np.array:
    """
    compute rotation matrix from euler angles
    """
    R = np.array(
        [
            [
                np.cos(alpha2) * np.cos(alpha3),
                -np.cos(alpha2) * np.sin(alpha3),
                np.sin(alpha2),
            ],
            [
                np.cos(alpha1) * np.sin(alpha3)
                + np.sin(alpha1) * np.sin(alpha2) * np.cos(alpha3),
                np.cos(alpha1) * np.cos(alpha3)
                - np.sin(alpha1) * np.sin(alpha2) * np.sin(alpha3),
                -np.sin(alpha1) * np.cos(alpha2),
            ],
            [
                np.sin(alpha1) * np.sin(alpha3)
                - np.cos(alpha1) * np.sin(alpha2) * np.cos(alpha3),
                np.sin(alpha1) * np.cos(alpha3)
                + np.cos(alpha1) * np.sin(alpha2) * np.sin(alpha3),
                np.cos(alpha1) * np.cos(alpha2),
            ]
        ]
    )
    return R

def rot_to_euler(R: np.array) -> Tuple[float, float, float]:
    """
    euler angles from rotation matrix
    """
    alpha1 = np.arctan2(-R[1, 2], R[2, 2])
    alpha2 = np.arcsin(R[0, 2])
    alpha3 = np.arctan2(-R[0, 1], R[0, 0])
    return alpha1, alpha2, alpha3

def homo_t_matrix(R: np.array, t: np.array) -> np.array:
    """
    create homogenous transformation matrix from
    rotation matrix and translation vector
    """
    H = np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], t[0]],
            [R[1, 0], R[1, 1], R[1, 2], t[1]],
            [R[2, 0], R[2, 1], R[2, 2], t[2]],
            [0, 0, 0, 1],
        ]
    )
    return H

