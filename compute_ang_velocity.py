import argparse
import itertools
import math
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fps = 15.01
CM_PER_PIXEL = 2.54 / 96

def _gen_calc_angle_speed_deg(angles):
    for i in range(len(angles) - 1):

        angle1 = angles[i]
        angle1 = angle1 % 360
        if angle1 < 0:
            angle1 += 360

        angle2 = angles[i + 1]
        angle2 = angle2 % 360
        if angle2 < 0:
            angle2 += 360

        diff1 = angle2 - angle1
        abs_diff1 = abs(diff1)
        diff2 = (360 + angle2) - angle1
        abs_diff2 = abs(diff2)
        diff3 = angle2 - (360 + angle1)
        abs_diff3 = abs(diff3)

        if abs_diff1 <= abs_diff2 and abs_diff1 <= abs_diff3:
            yield diff1
        elif abs_diff2 <= abs_diff3:
            yield diff2
        else:
            yield diff3

    yield 0

def _smooth(vec, smoothing_window):
    if smoothing_window <= 1 or len(vec) == 0:
        return vec.astype(float)
    else:
        assert smoothing_window % 2 == 1
        half_conv_len = smoothing_window // 2
        smooth_tgt = np.concatenate([
            np.full(half_conv_len, vec[0], dtype=vec.dtype),
            vec,
            np.full(half_conv_len, vec[-1], dtype=vec.dtype),
        ])

        smoothing_val = 1 / smoothing_window
        conv_arr = np.full(smoothing_window, smoothing_val)

        return np.convolve(smooth_tgt, conv_arr, mode='valid')


def calc_angle_speed_deg(angles, smoothing_window=1):
    """
    Calculate angular velocity from the given angles.
    """
    speed_deg = np.array(list(_gen_calc_angle_speed_deg(angles))) * fps
    speed_deg = _smooth(speed_deg, smoothing_window)

    return speed_deg

def calc_angle_deg(x,y):
    """
    calculates the angle of the orientation of the mouse in degrees
    """
    angle_rad = np.arctan2(y, x)

    return angle_rad * (180 / math.pi)


data = pd.read_csv(r"C:\Downloads\MATLAB_DATA_FOLDERS\2_MONTHS\AD\Test 349.csv")
neck_x = data["Neck_x"].values
neck_y = data["Neck_y"].values
tail_1_x = data["Tail_1_x"].values
tail_1_y = data["Tail_1_y"].values

x = neck_x - tail_1_x
y = neck_y - tail_1_y

ang_deg = calc_angle_deg(x,y)
ang_speed = calc_angle_speed_deg(ang_deg, smoothing_window=15)

# print(type(ang_deg))
print(ang_speed.shape)
