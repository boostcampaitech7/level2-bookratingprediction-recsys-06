import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.basic_data import BasicData
from src.data.interface import DataInterface


class ImageData(DataInterface):
    def __init__(self, cfg, data=None):
        super().__init__(cfg, data)

    def data_load(self):
        """
        Parameters
        ----------
        self._cfg.dataset.data_path : str
            데이터 경로를 설정할 수 있는 parser
        data : dict
            image_data_split로 부터 학습/평가/테스트 데이터가 담긴 사전 형식의 데이터를 입력합니다.

        Returns
        -------
        data : Dict
            학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
        """
        users, books, train, test, sub = self._load_data()

        # 이미지를 벡터화하여 데이터 프레임에 추가
        books_ = ImageData.__process_img_data(books, self._cfg)

        # 유저 및 책 정보를 합쳐서 데이터 프레임 생성 (단, 베이스라인에서는 user_id, isbn, img_vector만 사용함)
        # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
        user_features = []
        book_features = []
        sparse_cols = ["user_id", "isbn"] + list(
            set(user_features + book_features) - {"user_id", "isbn"}
        )

        train_df = train.merge(books_, on="isbn", how="left").merge(
            users, on="user_id", how="left"
        )[sparse_cols + ["img_vector", "rating"]]
        test_df = test.merge(books_, on="isbn", how="left").merge(
            users, on="user_id", how="left"
        )[sparse_cols + ["img_vector"]]
        all_df = pd.concat([train_df, test_df], axis=0)

        # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
        self._data.update(
            self._encode_label(all_df, train_df, test_df, sub, sparse_cols)
        )
        self._data.update({"field_names": sparse_cols})

    def data_split(self):
        """학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다."""
        basic_data = BasicData(self._cfg, self._data)
        basic_data.data_split()
        self._data = basic_data.get_data()

    def data_loader(self):
        """
        Parameters
        ----------
        self._cfg.dataloader.batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        self._cfg.dataloader.shuffle : bool
            data shuffle 여부
        self._cfg.dataloader.num_workers: int
            dataloader에서 사용할 멀티프로세서 수
        self._cfg.dataset.valid_ratio : float
            Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용
        data : Dict
            image_data_split()에서 반환된 데이터

        Returns
        -------
        data : Dict
            Image_Dataset 형태의 학습/검증/테스트 데이터를 DataLoader로 변환하여 추가한 후 반환합니다.
        """
        train_dataset = ImageData.__ImageDataset(
            self._data["X_train"][self._data["field_names"]].values,
            self._data["X_train"]["img_vector"].values,
            self._data["y_train"].values,
        )
        valid_dataset = (
            ImageData.__ImageDataset(
                self._data["X_valid"][self._data["field_names"]].values,
                self._data["X_valid"]["img_vector"].values,
                self._data["y_valid"].values,
            )
            if self._cfg.dataset.valid_ratio != 0
            else None
        )
        test_dataset = ImageData.__ImageDataset(
            self._data["test"][self._data["field_names"]].values,
            self._data["test"]["img_vector"].values,
        )

        self._add_data_loaders(train_dataset, valid_dataset, test_dataset)

    @staticmethod
    def __image_vector(path, img_size):
        """
        Parameters
        ----------
        path : str
            이미지가 존재하는 경로를 입력합니다.

        Returns
        -------
        img_fe : np.ndarray
            이미지를 벡터화한 결과를 반환합니다.
            베이스라인에서는 grayscale일 경우 RGB로 변경한 뒤, img_size x img_size 로 사이즈를 맞추어 numpy로 반환합니다.
        """
        img = Image.open(path)
        transform = v2.Compose(
            [
                v2.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                v2.Resize((img_size, img_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return transform(img).numpy()

    @staticmethod
    def __process_img_data(books, args):
        """
        Parameters
        ----------
        books : pd.DataFrame
            책 정보에 대한 데이터 프레임을 입력합니다.

        Returns
        -------
        books_ : pd.DataFrame
            이미지 정보를 벡터화하여 추가한 데이터 프레임을 반환합니다.
        """
        books_ = books.copy()
        books_["img_path"] = books_["img_path"].apply(lambda x: f"data/{x}")
        img_vecs = []
        for idx in tqdm(books_.index):
            img_vec = ImageData.__image_vector(
                books_.loc[idx, "img_path"], args.model_args[args.model].img_size
            )
            img_vecs.append(img_vec)

        books_["img_vector"] = img_vecs

        return books_

    class __ImageDataset(Dataset):
        def __init__(self, user_book_vector, img_vector, rating=None):
            """
            Parameters
            ----------
            user_book_vector : np.ndarray
                모델 학습에 사용할 유저 및 책 정보(범주형 데이터)를 입력합니다.
            img_vector : np.ndarray
                벡터화된 이미지 데이터를 입력합니다.
            rating : np.ndarray
                정답 데이터를 입력합니다.
            """
            self.user_book_vector = user_book_vector
            self.img_vector = img_vector
            self.rating = rating

        def __len__(self):
            return self.user_book_vector.shape[0]

        def __getitem__(self, i):
            return (
                {
                    "user_book_vector": torch.tensor(
                        self.user_book_vector[i], dtype=torch.long
                    ),
                    "img_vector": torch.tensor(self.img_vector[i], dtype=torch.float32),
                    "rating": torch.tensor(self.rating[i], dtype=torch.float32),
                }
                if self.rating is not None
                else {
                    "user_book_vector": torch.tensor(
                        self.user_book_vector[i], dtype=torch.long
                    ),
                    "img_vector": torch.tensor(self.img_vector[i], dtype=torch.float32),
                }
            )
