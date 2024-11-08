import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data.interface import DataInterface


class BasicData(DataInterface):
    def __init__(self, cfg, data=None):
        super().__init__(cfg, data)

    def data_load(self):
        """
        Parameters
        ----------
        cfg.dataset.data_path : str
            데이터 경로를 설정할 수 있는 parser

        Returns
        -------
        None
           self._data 학습 및 테스트 데이터가 담긴 사전 형식의 데이터로 설정합니다
        """
        _, _, train_df, test_df, sub = self._load_data()

        all_df = pd.concat([train_df, test_df], axis=0)

        sparse_cols = ["user_id", "isbn"]

        # 라벨 인코딩하고 인덱스 정보를 저장
        data = self._get_data_with_encode_label(
            all_df, train_df, test_df, sub, sparse_cols
        )
        data.update({"test": data.get("test").drop(["rating"], axis=1)})
        self._data = data

    def data_split(self):
        """
        Parameters
        ----------
        cfg.dataset.valid_ratio : float
            Train/Valid split 비율을 입력합니다.
        cfg.seed : int
            데이터 셔플 시 사용할 seed 값을 입력합니다.

        Returns
        -------
        None
            self._data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 설정합니다.
        """
        if self._cfg.dataset.valid_ratio == 0:
            self._data["X_train"] = self._data["train"].drop("rating", axis=1)
            self._data["y_train"] = self._data["train"]["rating"]

        else:
            X_train, X_valid, y_train, y_valid = train_test_split(
                self._data["train"].drop(["rating"], axis=1),
                self._data["train"]["rating"],
                test_size=self._cfg.dataset.valid_ratio,
                random_state=self._cfg.seed,
                shuffle=True,
            )

            (
                self._data["X_train"],
                self._data["X_valid"],
                self._data["y_train"],
                self._data["y_valid"],
            ) = (
                X_train,
                X_valid,
                y_train,
                y_valid,
            )

    def data_loader(self):
        """
        Parameters
        ----------
        cfg.dataloader.batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        cfg.dataloader.shuffle : bool
            data shuffle 여부
        cfg.dataloader.num_workers: int
            dataloader에서 사용할 멀티프로세서 수
        cfg.dataset.valid_ratio : float
            Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
        self._data : dict
            basic_data_split 함수에서 반환된 데이터

        Returns
        -------
        None
            self._data를 DataLoader가 추가된 데이터로 설정합니다.
        """

        train_dataset = TensorDataset(
            torch.LongTensor(self._data["X_train"].values),
            torch.LongTensor(self._data["y_train"].values),
        )
        valid_dataset = (
            TensorDataset(
                torch.LongTensor(self._data["X_valid"].values),
                torch.LongTensor(self._data["y_valid"].values),
            )
            if self._cfg.dataset.valid_ratio != 0
            else None
        )
        test_dataset = TensorDataset(torch.LongTensor(self._data["test"].values))
        self._add_data_loaders(train_dataset, valid_dataset, test_dataset)
