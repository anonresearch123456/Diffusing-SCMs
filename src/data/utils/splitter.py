import json
from typing import List, Union
import pandas as pd
import numpy as np
from pathlib import Path

import random


class DataSplitter():
    def __init__(self, out_path: Union[str, Path], tabular_data: pd.DataFrame) -> None:
        if type(out_path) is not Path:
            out_path = Path(out_path)
        self.out_path = out_path
        self.tabular_data: pd.DataFrame = tabular_data
        self.all_ids = list(self.tabular_data["eid"])
        self.all_ids.sort()

    def _write_to_file(self, file_name: str, train: List[int], val: List[int], test: List[int]):
        out_dict = {"train": train, "val": val, "test": test}
        with open(self.out_path / (file_name + ".json"), 'w') as f:
            json.dump(out_dict, f)

    def generate_overfit_split(self, size: int = 10, seed: int = 42):
        random.seed(seed)
        overfit_ids = random.sample(self.all_ids, size)
        self._write_to_file("overfit", overfit_ids, overfit_ids, overfit_ids)

    def generate_split(self, split_name: str, test_ratio: float = 0.2, val_ratio: int = 0.1, seed: int = 42):
        """
        """
        assert test_ratio + val_ratio <= 1.0
        random.seed(seed)
        len_all = len(self.all_ids)
        
        test_size = int(test_ratio * len_all)
        val_size = int(val_ratio * len_all)

        test_ids = random.sample(self.all_ids, test_size)

        train_val_ids = list(set(self.all_ids).difference(set(test_ids)))

        val_ids = random.sample(train_val_ids, val_size)

        train_ids = list(set(train_val_ids).difference(set(val_ids)))

        # check validity
        _all_ids = train_ids + val_ids + test_ids
        _all_ids.sort()
        assert _all_ids == self.all_ids

        self._write_to_file(split_name, train_ids, val_ids, test_ids)


def main():
    tab_path = ".../cf_data.csv"
    out_path = "/path/to/save/splits"
    tabular_data = pd.read_csv(tab_path)
    out_path = Path(out_path)
    splitter = DataSplitter(out_path, tabular_data)
    splitter.generate_split("split", test_ratio=1.0, val_ratio=0.0)


if __name__ == "__main__":
    main()
