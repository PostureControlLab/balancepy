from cmath import pi
from dataclasses import dataclass
from numbers import Number
import numpy as np
from numpy.typing import NDArray


def calculate_com_2segmentmodel(
    shoulder_t: NDArray[np.number],
    shoulder_marker_height: np.number,
    hip_t: NDArray[np.number],
    hip_marker_height: np.number,
    height_m: float,
    rotation: bool = True,
) -> NDArray:
    """Calculates center of mass from hip and shoulder movement using a 2 segment model and anthropometric tables.

    Args:
        shoulder_t (NDArray[np.number]): 1D shoulder translation in meters
        shoulder_marker_height (np.number): shoulder marker height above support surface in meters
        hip_t (NDArray[np.number]): 1D hip AP translation in meters
        hip_t_height (np.number): hip marker height above support surface in meters
        height_m (float): Height of subject in meters
        rotation (bool): False: COM translation in m; True COM rotation about ankle joints in degrees

    Returns:
        NDArray: 1D center of mass
    """
    """
    This function calculates the center of mass (COM) using a 2-segment model based on anthropometric data.
    """
    assert shoulder_t.ndim == 1
    assert hip_t.ndim == 1

    # AP is Z
    # ML is X

    h_sm = np.mean(shoulder_marker_height)
    h_hm = np.mean(hip_marker_height)

    # Needs to be in meters (m) for correct moment of inertia calculations
    # mass cancels out in com calculation and is set to one here.
    wt = WinterTable(1, height_m)

    sho = shoulder_t - np.mean(shoulder_t)
    hip = hip_t - np.mean(hip_t)

    hT = wt.smh_trunk
    hL = wt.smh_legs
    mT = wt.sm_trunk
    mL = wt.sm_legs

    comL = hip * hL / h_hm

    #  Movement of trunk above hip is taken relative to hip marker (i.e. hip
    #  marker is used as approximate centre of rotation for the hip joint).
    comT = hip + ((sho - hip) * hT / (h_sm - h_hm))

    comB = (comL * mL + comT * mT) / (mL + mT)

    if rotation:
        out = np.arcsin(comB / (wt.smh_body)) * 180 / pi
    else:
        out = comB

    return out


@dataclass
class WinterTable:
    """
    Based on the anthropometric table as published in 'Biomechanics and Motor Control of Human Movement', second Ed. (1990) by David A. Winter

    Attributes:
        sm_*    Segment mass
        sl_*    Segment length
        smh_*   Segment center of mass height above supporting joints
        Jcom_*  Moment of inertia for rotation around segment center of mass
        Jj_*    Moment of inertia for rotation around proximal joint
    """

    height_m: Number
    mass_kg: Number
    J: Number
    mgh: Number
    sm_foot: Number
    sm_legs: Number
    sm_trunk: Number
    sl_foot: Number
    sl_legs: Number
    sl_trunk: Number
    smh_legs: Number
    smh_trunk: Number
    smh_body: Number
    Jcom_legs: Number
    Jcom_trunk: Number
    Jcom_body: Number
    Jj_legs: Number
    Jj_trunk: Number
    Jj_body: Number
    eyes_above_ground_m: Number
    shoulder_above_ankle_m: Number

    def __init__(self, mass_kg: Number, height_m: Number):
        m_F = 2 * 0.0145 * mass_kg
        m_L = 2 * (0.0465 + 0.1) * mass_kg  # relative mass of shank+thigh
        m_T = 0.678 * mass_kg

        l_F = 0.039 * height_m  # vertical length...
        l_L = 0.53 * height_m - l_F
        l_T = (0.87 - 0.53) * height_m  # heigth of glenohumeral joint above hip

        h_L = 0.553 * l_L
        h_T = 0.626 * (0.818 - 0.53) * height_m  # heigth trunk com above hip
        h_com = (h_L * m_L + (l_L + h_T) * m_T) / (m_L + m_T)  # com height above ankle

        # Moment of inertia for rotation about segment COM
        J_Fc = m_F * (0.475 * l_F) ** 2
        J_Lc = m_L * (0.326 * l_L) ** 2
        J_Tc = m_T * (0.496 * l_T) ** 2

        # Moment of inertia for rotation about joints
        J_L = J_Lc + m_L * h_L**2
        J_T = J_Tc + m_T * h_T**2
        J_B = J_Lc + J_Tc + m_L * h_L**2 + m_T * (l_L + h_T) ** 2

        self.height_m = height_m
        self.mass_kg = mass_kg
        self.J = J_B
        self.mgh = (mass_kg - m_F) * 9.81 * h_com

        self.sm_foot = (m_F,)
        self.sm_legs = m_L
        self.sm_trunk = m_T

        self.sl_foot = l_F
        self.sl_leg = l_L
        self.sl_trunk = l_T

        self.smh_legs = h_L
        self.smh_trunk = h_T
        self.smh_body = h_com

        self.Jcom_foot = J_Fc
        self.Jcom_leg = J_Lc
        self.Jcom_trunk = J_Tc

        self.Jj_legs = J_L
        self.Jj_trunk = J_T
        self.Jj_body = J_B

        self.eyes_above_ground_m = 0.936 * height_m
        self.shoulder_above_ankle_m = (0.818 - 0.039) * height_m
