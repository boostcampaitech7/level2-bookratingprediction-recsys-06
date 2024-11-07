from abc import abstractmethod, ABC

import pandas as pd
from torch.utils.data import DataLoader


class DataInterface(ABC):

    def __init__(self, cfg, data=None):
        self._cfg = cfg
        if data is None:
            data = {}
        self._data: dict[
            str, pd.DataFrame | pd.Series | dict[str, dict[any, int]] | None
        ] = data

    def _load_data(self):
        users = pd.read_csv(
            self._cfg.dataset.data_path + "users.csv"
        )  # 베이스라인 코드에서는 사실상 사용되지 않음
        books = pd.read_csv(self._cfg.dataset.data_path + "books.csv")
        train = pd.read_csv(self._cfg.dataset.data_path + "train_ratings.csv")
        test = pd.read_csv(self._cfg.dataset.data_path + "test_ratings.csv")
        sub = pd.read_csv(self._cfg.dataset.data_path + "sample_submission.csv")
        return users, books, train, test, sub

    @staticmethod
    def _get_data_with_encode_label(
            all_df: pd.DataFrame,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            sub: pd.DataFrame,
            sparse_cols: list[str],
    ) -> dict[str, pd.DataFrame | pd.Series | dict[str, dict[any, int]] | None]:
        label2idx, idx2label = {}, {}
        for col in sparse_cols:
            all_df[col] = all_df[col].fillna("unknown")
            train_df[col] = train_df[col].fillna("unknown")
            test_df[col] = test_df[col].fillna("unknown")
            unique_labels = all_df[col].astype("category").cat.categories
            label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
            idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}
            train_df[col] = train_df[col].map(label2idx[col])
            test_df[col] = test_df[col].map(label2idx[col])

        field_dims = [len(label2idx[col]) for col in sparse_cols]
        return {
            "train": train_df,
            "test": test_df,
            "field_dims": field_dims,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "sub": sub,
        }

    def get_data(
            self,
    ) -> dict[str, pd.DataFrame | pd.Series | dict[str, dict[any, int]] | None]:
        return self._data

    @abstractmethod
    def data_load(self):
        raise NotImplementedError

    @abstractmethod
    def data_split(self):
        raise NotImplementedError

    @abstractmethod
    def data_loader(self):
        raise NotImplementedError

    def _add_data_loaders(self, train_dataset, valid_dataset, test_dataset):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._cfg.dataloader.batch_size,
            shuffle=self._cfg.dataloader.shuffle,
            num_workers=self._cfg.dataloader.num_workers,
        )
        valid_dataloader = (
            DataLoader(
                valid_dataset,
                batch_size=self._cfg.dataloader.batch_size,
                shuffle=False,
                num_workers=self._cfg.dataloader.num_workers,
            )
            if self._cfg.dataset.valid_ratio != 0
            else None
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self._cfg.dataloader.batch_size,
            shuffle=False,
            num_workers=self._cfg.dataloader.num_workers,
        )
        (
            self._data["train_dataloader"],
            self._data["valid_dataloader"],
            self._data["test_dataloader"],
        ) = (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
        )

    # @abstractmethod
    # def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
    #     pass
    #
    # @abstractmethod
    # def _preprocess(self):
    #     pass
    #
    # @abstractmethod
    # def get_data(self) -> pd.DataFrame:
    #     pass
