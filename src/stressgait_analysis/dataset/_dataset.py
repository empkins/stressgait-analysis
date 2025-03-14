from typing import Sequence, ClassVar

import pandas as pd
from biopsykit.utils._types_internal import path_t
from tpcp import Dataset


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

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        coarse_condition: bool = False,
        exclude_missing_data: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.base_path = base_path
        self.coarse_condition = coarse_condition
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
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
