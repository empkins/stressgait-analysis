from os.path import exists
from typing import Sequence, ClassVar

import pandas as pd
from biopsykit.utils._types_internal import path_t
from pandas import DataFrame
from tpcp import Dataset
import numpy as np
from scipy import signal
import json
import os

class StressGaitDataset(Dataset):

    PARTICIPANTS_EXCLUDED: ClassVar[Sequence[str]] = [
        "VP_03",
        "VP_04",
        "VP_09",
        "VP_14",
        "VP_21",
    ]

    PARTICIPANTS_NO_SALIVA: ClassVar[Sequence[str]] = ["VP_04", "VP_48"]

    PARTICIPANTS_HIGH_S0_CORTISOL: ClassVar[Sequence[str]] = ["VP_39"]

    SAMPLE_TIMES: ClassVar[Sequence[int]] = [0, 30, 34, 38, 48, 58]

    PARTICIPANTS_SPEED_INVERSE = ["VP11", "VP12", "VP15"]

    GRF_DICT = { 'ground_force1_vx': 'GRF_x_1', 'ground_force1_vy': 'GRF_y_1', 'ground_force1_vz': 'GRF_z_1',
                 'ground_force1_px': 'COP_x_1', 'ground_force1_py': 'COP_y_1', 'ground_force1_pz': 'COP_z_1',
    'ground_torque1_x': 'M_x_1', 'ground_torque1_y': 'M_y_1', 'ground_torque1_z': 'M_z_1',
    'ground_force2_vx': 'GRF_x_2', 'ground_force2_vy': 'GRF_y_2', 'ground_force2_vz': 'GRF_z_2',
    'ground_force2_px': 'COP_x_2', 'ground_force2_py': 'COP_y_2', 'ground_force2_pz': 'COP_z_2',
    'ground_torque2_x': 'M_x_2', 'ground_torque2_y': 'M_y_2', 'ground_torque2_z': 'M_z_2'}

    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "model_keypoints.json")

    # Open and load the JSON file
    with open(json_path, "r") as f:
        MODEL_KEYPOINTS = json.load(f)


    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        coarse_condition: bool = False,
        exclude_missing_data: bool = True,
        specify_bouts: bool = False,
        use_cache: bool = True,
        gait_data_path = None,
        specify_speed = False
    ) -> None:
        self.base_path = base_path
        self.coarse_condition = coarse_condition
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.specify_bouts = specify_bouts
        self.specify_speed = specify_speed
        self.gait_data_path = gait_data_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"
        index = pd.read_csv(self.base_path.joinpath("metadata/condition_list.csv"))
        index = index[["participant", "condition"]].set_index("participant")

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self._find_data_to_exclude():
                index = index.drop(index=p_id, errors="ignore")

        if self.coarse_condition:
            index["condition"] = index["condition"].str.split("_").str[0]



        index = index.reset_index()

        if self.specify_bouts:
            columns = index.columns
            index = pd.DataFrame(np.repeat(index.values, 2, axis=0))
            index.columns = columns
            index['bout'] = '1'
            index.loc[1::2, 'bout'] = '2'

        if self.specify_speed:
            normal = ~index["participant"].isin(self.PARTICIPANTS_SPEED_INVERSE)
            index.loc[normal, "speed"] = index.loc[normal, "bout"].apply(lambda x: "slow" if x == "1" else "fast")
            inverse = index["participant"].isin(self.PARTICIPANTS_SPEED_INVERSE)
            index.loc[inverse, "speed"] = index.loc[inverse, "bout"].apply(lambda x: "fast" if x == "1" else "slow")

        return index

    def _find_data_to_exclude(self) -> Sequence[str]:
        data_to_exclude = []
        data_to_exclude.extend(self.PARTICIPANTS_EXCLUDED)
        data_to_exclude.extend(self.PARTICIPANTS_NO_SALIVA)
        data_to_exclude.extend(self.PARTICIPANTS_HIGH_S0_CORTISOL)
        data_to_exclude = sorted(set(data_to_exclude))
        return data_to_exclude

    @property
    def excluded_participants(self) -> Sequence[str]:
        return self._find_data_to_exclude()

    @property
    def condition_fine(self) -> pd.DataFrame:
        data = pd.read_csv(self.base_path.joinpath("metadata/condition_list.csv"))
        data = data[["participant", "condition"]]
        data = data.set_index(["participant"])

        return data

    @property
    def condition_coarse(self) -> pd.DataFrame:
        data = pd.read_csv(self.base_path.joinpath("metadata/condition_list.csv"))
        data = data[["participant", "condition"]]
        data["condition"] = data["condition"].str.split("_").str[0]
        data = data.set_index(["participant"])
        return data

    @property
    def condition(self):
        if self.coarse_condition:
            return self.condition_coarse
        else:
            return self.condition_fine

    @property
    def sample_times(self) -> Sequence[int]:
        return self.SAMPLE_TIMES

    @property
    def cortisol(self) -> pd.DataFrame:
        return self._load_saliva("cortisol")

    @property
    def amylase(self) -> pd.DataFrame:
        return self._load_saliva("amylase")

    def _load_saliva(self, saliva_type: str) -> pd.DataFrame:
        file_path = self.base_path.joinpath(
            f"saliva/processed/stressgait_{saliva_type}_samples.csv"
        )
        data = pd.read_csv(file_path)
        data = data.rename(columns={"subject": "participant"}).set_index(
            ["participant", "sample"]
        )
        data = data.join(self.condition)
        data = data.reset_index()
        data = data.rename(columns={"participant": "subject"})
        data = data.set_index(["subject", "condition", "sample"])

        return data.loc[self.index["participant"].unique()]

    @property
    def cortisol_features(self) -> pd.DataFrame:
        return self._load_saliva_features("cortisol")

    @property
    def amylase_features(self) -> pd.DataFrame:
        return self._load_saliva_features("amylase")

    def _load_saliva_features(self, saliva_type: str) -> pd.DataFrame:
        file_path = self.base_path.joinpath(
            f"saliva/processed/stressgait_{saliva_type}_features.csv"
        )
        data = pd.read_csv(file_path)
        data = data.rename(columns={"subject": "participant"}).set_index(
            ["participant", "saliva_feature"]
        )
        data = data.join(self.condition)
        data = data.reset_index()
        data = data.rename(columns={"participant": "subject"})
        data = data.set_index(["subject", "condition", "saliva_feature"])

        return data.loc[self.index["participant"].unique()]

    def load_force_plate_data(self: str, video_framerate = False) -> pd.DataFrame:
        if not 'bout' in self.index.columns:
            raise KeyError('specifiy bout in dataset index')
        self.assert_is_single(None, "bout")
        id = self.index['participant'][0].replace("_", "")
        bout = self.index['bout'][0]
        file_path = next(self.gait_data_path.joinpath(f"vicon/{id}").rglob(f"bout{bout}.mot"))

        grf = pd.read_csv(file_path, index_col=None, delimiter='\t', skiprows=6)
        columns = grf.columns
        grf = grf.reset_index(inplace=False).iloc[:, :-1]
        grf.columns = columns

        time = grf.time
        grf = self._filter_data(grf, 1000)
        grf['time'] = time
        grf.rename(columns=self.GRF_DICT, inplace=True)
        if video_framerate:
            return grf.set_index('time')[::20]

        return grf.set_index('time')


    def _filter_data(self, data, fs):
        b, a = signal.butter(2, 10, fs=fs)
        column_names = data.columns
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        filtered_data = pd.DataFrame(filtered_data, columns=column_names)
        filtered_data.index = data.index
        return filtered_data


    def load_keypoint_trajectories(self, model: object = 'rtmo') -> DataFrame:
        """
        the function computes the keypoint trajectories given of a single trial at 50Hz in the coordinate system where X is forward and Y is vertical.
        currently, left sagital video data is used.
        :param model: specify the pose estimation model to use
        :return: a DataFrame containing the keypoint trajectories of one trial, rows are time points and columns are keypoint names
        """
        if not 'bout' in self.index.columns:
            raise KeyError('specifiy bout in dataset index')
        self.assert_is_single(None, "bout")
        id = self.index['participant'][0].replace("_", "")
        bout = self.index['bout'][0]
        json_file = next(self.gait_data_path.joinpath(f"pred_{model}/{id}").rglob(f"bout{bout}_0.json"))

        #load the json file

        with open(json_file) as f:
            d = json.load(f)  # d is a list that has every frame as a list entry
        n_keypoints = len(self.MODEL_KEYPOINTS[f"{model}"]["forward"].keys())
        traj = np.empty((len(d), n_keypoints * 2))

        # iterate over all frames
        for i, frame in enumerate(d):
            try:
                instance = frame['instances'][0]
                frame_keypoints = instance['keypoints']

                for n in range(n_keypoints):
                    kp_name = self.MODEL_KEYPOINTS[f"{model}"]["forward"][str(n)]
                    kp_values = frame_keypoints[self.MODEL_KEYPOINTS[f"{model}"]["inverse"][kp_name]]
                    traj[i, n * 2:n * 2 + 2] = kp_values
            except:
                traj[i, :] = np.nan

        traj = pd.DataFrame(traj)
        column_names = []
        for n in range(n_keypoints):
            kp_name = self.MODEL_KEYPOINTS[f"{model}"]["forward"][str(n)]
            column_names.extend([kp_name + '_x', kp_name + '_y'])

        traj.columns = column_names
        traj['time'] = traj.index / 50
        traj.set_index('time', inplace=True)


        #interpolate missing NaNs
        traj_interpolated = traj.interpolate(axis=0, limit=4)

        #filter the trajectories
        traj_filtered = self._filter_data(traj_interpolated, fs=50)

        #convert the KOS
        pixel = [460, 640]  # return breite * höhe
       #assume we have the left side view
        for i in traj_filtered.columns:
            if '_x' in i:
                traj_filtered.loc[:, i] = pixel[0] - traj_filtered.loc[:, i]
            elif '_y' in i:
                traj_filtered.loc[:, i] = pixel[1] - traj_filtered.loc[:, i]
            else:
                pass

        return traj_filtered

    @property
    def gait_cycle_kinematics(self) -> pd.DataFrame:

        if not 'bout' in self.index.columns:
            raise KeyError('specifiy bout in dataset index')
        self.assert_is_single(None, "bout")
        id = self.index['participant'][0].replace("_", "")
        bout = self.index['bout'][0]
        return


    @property
    def confidence_scores(self) -> DataFrame:
        """
        function to compute the confidence score ranging between [0,1] of every keypoint estimation at every time point
        :return: a pd Dataframe for a single bout, where every row is one timepoint (at 50Hz) and the columns are the keypoint names
        """
        raise KeyError('specifiy bout in dataset index')
        self.assert_is_single(None, "bout")
        id = self.index['participant'][0].replace("_", "")
        bout = self.index['bout'][0]
        json_file = next(self.gait_data_path.joinpath(f"pred_{model}/{id}").rglob(f"bout{bout}_0.json"))

        # load the json file

        with open(json_file) as f:
            d = json.load(f)  # d is a list that has every frame as a list entry
        n_keypoints = len(self.MODEL_KEYPOINTS[f"{model}"]["forward"].keys())
        traj = np.empty((len(d), n_keypoints * 2))

        # iterate over all frames
        for i, frame in enumerate(d):
            try:
                instance = frame['instances'][0]
                frame_keypoints = instance['keypoint_scores']

                for n in range(n_keypoints):
                    kp_name = self.MODEL_KEYPOINTS[f"{model}"]["forward"][str(n)]
                    kp_values = frame_keypoints[self.MODEL_KEYPOINTS[f"{model}"]["inverse"][kp_name]]
                    traj[i, n * 2:n * 2 + 2] = kp_values
            except:
                traj[i, :] = np.nan

        traj = pd.DataFrame(traj)
        column_names = []
        for n in range(n_keypoints):
            kp_name = sself.MODEL_KEYPOINTS[f"{model}"]["forward"][str(n)]
            column_names.extend([kp_name + '_x', kp_name + '_y'])

        traj.columns = column_names
        traj['time'] = traj.index / 50
        traj.set_index('time', inplace=True)


        return traj

    @property
    def kinematics(self):
        try:
            kinematics = pd.read_pickle(self.base_path.joinpath('kinematics/kinematics.pkl'))
        except:
            raise FileNotFoundError('kinematics.pkl not found, please run the file "Gait_kinematics.py" first')

        subset = self.index
        flat_kinematics = kinematics.reset_index()
        filtered_kinematics = flat_kinematics.merge(subset, on=subset.columns.tolist(), how='inner')
        return filtered_kinematics.set_index(['participant', 'condition', 'bout', 'speed', 'cycle_idx', 'percentage_of_stride'])