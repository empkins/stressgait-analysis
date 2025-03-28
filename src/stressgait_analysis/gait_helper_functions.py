import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


from scipy.signal import find_peaks
def compute_HS(keypoints):
    dist = keypoints.left_ankle_x - keypoints.left_hip_x
    hs_idx, _ = find_peaks(dist, width=15, prominence=50)
    hs_v = keypoints.iloc[hs_idx].index
    return np.array(hs_v)


def compute_leg_kinematics(keypoints_o):
        keypoints = keypoints_o.copy()
        keypoint_names = ['left_shoulder_x', 'left_shoulder_y', 'left_hip_x', 'left_hip_y', 'left_knee_x', 'left_knee_y',
                     'left_ankle_x', 'left_ankle_y', 'left_heel_x', 'left_heel_y', 'left_big_toe_x', 'left_big_toe_y']
        for k in keypoint_names:
            if k not in keypoints.columns:
                keypoints.loc[:,k] = np.nan
        df_ang = keypoints.loc[:, keypoint_names]

        angles = leg2d(np.array(df_ang))
        angles = pd.DataFrame(angles, index = keypoints.index, columns = ['hip_flexion', 'knee_flexion', 'ankle_plantarflexion'])


        return angles

def leg2d(mocap_f):

    """
    % Coordinate system:
    %	X is forward (direction of walking), Y is up
    %
    % Markers:
    %	1: Shoulder
    %	2: Greater trochanter
    %	3: Lateral epicondyle of knee
    %	4: Lateral malleolus
    %	5: Heel (placed at same height as marker 6)
    %	6: Head of 5th metatarsal
    %
    % Joints:
    %	hip, knee, ankle
    %   sign convention for angles and moments: hip flexion, knee flexion, ankle plantarflexion are positive
    %

    """
    np.seterr(divide='ignore', invalid='ignore')
    # some constants
    nr_markers = 6
    g = 9.80665
    nr_samples = len(mocap_f)
    segments = np.array([[1, 2, np.nan, np.nan, np.nan, np.nan, np.nan],  # HAT (head-arms-trunk)
                         [2, 3, 2, 0.100, 0.433, 0.323, +1],  # (thigh)
                         [3, 4, 3, 0.0465, 0.433, 0.302, -1],  # shank
                         [5, 6, 4, 0.0145, 0.500, 0.475, -1]])  # foot
    nr_segments = len(segments)

    seg_com_x = np.zeros((nr_samples, nr_segments))  # e.g. 101x4
    seg_com_y = np.zeros((nr_samples, nr_segments))
    seg_a = np.zeros((nr_samples, nr_segments))  # angle of the segment
    angles = np.zeros((nr_samples, nr_segments - 1))

    for i in range(nr_segments):
        iprox = int(segments[i, 0])
        idist = int(segments[i, 1])  # distal marker
        try:
            com = int(segments[i, 4])  # center of mass location RELATIVE to line from prox to distal marker
        except:
            com = segments[i, 4]
        prox_x = mocap_f[:, (2 * iprox - 2)]  # x wert
        prox_y = mocap_f[:, (2 * iprox - 1)]  # y wert
        dist_x = mocap_f[:, (2 * idist - 2)]  # x wert
        dist_y = mocap_f[:, (2 * idist - 1)]  # y wert
        r_x = dist_x - prox_x
        r_y = dist_y - prox_y
        seg_com_x[:, i] = prox_x + com * r_x
        seg_com_y[:, i] = prox_y + com * r_y
        seg_a[:, i] = np.unwrap(
            np.arctan2(r_y, r_x))  # orientation of the vector R, unwrap removes -pi to pi discontinuities

    for i in range(nr_segments - 1, 0, -1):
        g = i - 1
        sign = segments[i, 6]
        angles[:, g] = sign * (seg_a[:, i] - seg_a[:, i - 1])
        max_iter = 10  # this should be never reached
        count_iter = 1
        while count_iter < max_iter:
            isabove = np.max(angles[:, g] > np.pi) == 1
            isbelow = np.max(angles[:, g] < -np.pi) == 1
            if isabove:
                angles[:, g] = angles[:, g] - 2 * np.pi
            elif isbelow:
                angles[:, g] = angles[:, g] + 2 * np.pi
            else:
                break
            count_iter += 1

    angles[:, 2] += np.pi / 2

    return np.rad2deg(angles)


def compute_arm_kinematics(keypoints):
        u_x = keypoints['left_elbow_x'] - keypoints['left_shoulder_x']
        u_y = keypoints['left_elbow_y'] - keypoints['left_shoulder_y']
        v_x = keypoints['left_wrist_x'] - keypoints['left_elbow_x']
        v_y = keypoints['left_wrist_y'] - keypoints['left_elbow_y']

        # Dot products and magnitudes
        dot_product = u_x * v_x + u_y * v_y
        magnitude_u = np.sqrt(u_x ** 2 + u_y ** 2)
        magnitude_v = np.sqrt(v_x ** 2 + v_y ** 2)

        # Compute cosine of angles
        cos_theta = dot_product / (magnitude_u * magnitude_v)

        # Clip values to [-1, 1] to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Compute angles in degrees
        elbow_flexion_angles = np.degrees(np.arccos(cos_theta))

        u_x = keypoints['left_wrist_x'] - keypoints['left_shoulder_x']
        u_y = keypoints['left_wrist_y'] - keypoints['left_shoulder_y']

        # Vertical reference vector (unit vector pointing straight up)
        v_x = 0
        v_y = -1  # Assuming screen coordinate system where y increases downward

        # Dot product and magnitudes
        dot_product = u_x * v_x + u_y * v_y
        magnitude_u = np.sqrt(u_x ** 2 + u_y ** 2)
        magnitude_v = 1.0  # Unit vector, so magnitude is 1

        # Compute cosine of angle
        cos_theta = dot_product / (magnitude_u * magnitude_v)

        # Clip values to [-1, 1] to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Compute angle in degrees
        arm_swing_angles = np.degrees(np.arccos(cos_theta))

        arm_angles = pd.DataFrame([elbow_flexion_angles, arm_swing_angles]).T
        arm_angles.columns = ['elbow_flexion_angles', 'arm_swing_angles']
        arm_angles.index = keypoints.index
        return arm_angles

def resample(data, n=100):
    x_new = np.linspace(data.index[0], data.index[-1], num=n, endpoint=True)
    resampled = {}

    for column in data.columns:
        f = interp1d(data.index, data[column])  # or 'cubic'
        resampled[column] = f(x_new)
    return pd.DataFrame(resampled, index=np.arange(n))