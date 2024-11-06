import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from .basic_data import BasicData
from .interface import DataInterface


class TextData(DataInterface):
    def __init__(self, cfg, data=None):
        super().__init__(cfg, data)

    def data_load(self):
        """
        Parameters
        ----------
        self._cfg.dataset.data_path : str
            데이터 경로를 설정할 수 있는 parser
        self._cfg.model_args[self._cfg.model].pretrained_model : str
            사전학습된 모델을 설정할 수 있는 parser
        self._cfg.model_args[self._cfg.model].vector_create : bool
            텍스트 데이터 벡터화 및 저장 여부를 설정할 수 있는 parser
            False로 설정하면 기존에 저장된 벡터를 불러옵니다.

        Returns
        -------
        data : dict
            학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
        """
        users, books, train, test, sub = self._load_data()

        tokenizer = AutoTokenizer.from_pretrained(
            self._cfg.model_args[self._cfg.model].pretrained_model
        )
        model = AutoModel.from_pretrained(
            self._cfg.model_args[self._cfg.model].pretrained_model
        ).to(device=self._cfg.device)
        model.eval()
        users_, books_ = self.__process_text_data(
            train,
            users,
            books,
            tokenizer,
            model,
            self._cfg.model_args[self._cfg.model].vector_create,
        )

        # 유저 및 책 정보를 합쳐서 데이터 프레임 생성 (단, 베이스라인에서는 user_id, isbn, user_summary_merge_vector, book_summary_vector만 사용함)
        # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
        user_features = []
        book_features = []
        sparse_cols = ["user_id", "isbn"] + list(
            set(user_features + book_features) - {"user_id", "isbn"}
        )

        train_df = train.merge(books_, on="isbn", how="left").merge(
            users_, on="user_id", how="left"
        )[sparse_cols + ["user_summary_merge_vector", "book_summary_vector", "rating"]]
        test_df = test.merge(books_, on="isbn", how="left").merge(
            users_, on="user_id", how="left"
        )[sparse_cols + ["user_summary_merge_vector", "book_summary_vector"]]
        all_df = pd.concat([train, test], axis=0)

        self._data = self._encode_label(all_df, train_df, test_df, sub, sparse_cols)
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
        data : dict
            text_data_load()에서 반환된 데이터

        Returns
        -------
        data : dict
            Text_Dataset 형태의 학습/검증/테스트 데이터를 DataLoader로 변환하여 추가한 후 반환합니다.
        """
        train_dataset = TextData.__Dataset(
            self._data["X_train"][self._data["field_names"]].values,
            self._data["X_train"]["user_summary_merge_vector"].values,
            self._data["X_train"]["book_summary_vector"].values,
            self._data["y_train"].values,
        )
        valid_dataset = (
            TextData.__Dataset(
                self._data["X_valid"][self._data["field_names"]].values,
                self._data["X_valid"]["user_summary_merge_vector"].values,
                self._data["X_valid"]["book_summary_vector"].values,
                self._data["y_valid"].values,
            )
            if self._cfg.dataset.valid_ratio != 0
            else None
        )
        test_dataset = TextData.__Dataset(
            self._data["test"][self._data["field_names"]].values,
            self._data["test"]["user_summary_merge_vector"].values,
            self._data["test"]["book_summary_vector"].values,
        )
        self._add_data_loaders(train_dataset, valid_dataset, test_dataset)

    @staticmethod
    def __text_preprocessing(summary: pd.Series | str):
        """
        Parameters
        ----------
        summary : pd.Series
            정규화와 같은 기본적인 전처리를 하기 위한 텍스트 데이터를 입력합니다.

        Returns
        -------
        summary : pd.Series
            전처리된 텍스트 데이터를 반환합니다.
            베이스라인에서는 특수문자 제거, 공백 제거를 진행합니다.
        """
        summary = re.sub(
            "[^0-9a-zA-Z.,!?]", " ", summary
        )  # .,!?를 제외한 특수문자 제거
        summary = re.sub("\s+", " ", summary)  # 중복 공백 제거

        return summary

    @staticmethod
    def __text_to_vector(text, tokenizer, model):
        """
        Parameters
        ----------
        text : str
            `summary_merge()`를 통해 병합된 요약 데이터
        tokenizer : Tokenizer
            텍스트 데이터를 `model`에 입력하기 위한 토크나이저
        model : 사전학습된 언어 모델
            텍스트 데이터를 벡터로 임베딩하기 위한 모델
        ----------
        """
        text_ = "[CLS] " + text + " [SEP]"
        tokenized = tokenizer.encode(text_, add_special_tokens=True)
        token_tensor = torch.tensor([tokenized], device=model.device)
        with torch.no_grad():
            outputs = model(token_tensor)  # attention_mask를 사용하지 않아도 됨
            ### BERT 모델의 경우, 최종 출력물의 사이즈가 (토큰길이, 임베딩=768)이므로, 이를 평균내어 사용하거나 pooler_output을 사용하여 [CLS] 토큰의 임베딩만 사용
            # sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0)  # 방법1) 모든 토큰의 임베딩을 평균내어 사용
            sentence_embedding = outputs.pooler_output.squeeze(
                0
            )  # 방법2) pooler_output을 사용하여 맨 첫 토큰인 [CLS] 토큰의 임베딩만 사용

        return sentence_embedding.cpu().detach().numpy()

    @staticmethod
    def __process_text_data(
            ratings, users, books, tokenizer, model, vector_create=False
    ):
        """
        Parameters
        ----------
        users : pd.DataFrame
            유저 정보에 대한 데이터 프레임을 입력합니다.
        books : pd.DataFrame
            책 정보에 대한 데이터 프레임을 입력합니다.
        vector_create : bool
            사전에 텍스트 데이터 벡터화가 된 파일이 있는지 여부를 입력합니다.

        Returns
        -------
        `users_` : pd.DataFrame
            각 유저가 읽은 책에 대한 요약 정보를 병합 및 벡터화하여 추가한 데이터 프레임을 반환합니다.

        `books_` : pd.DataFrame
            텍스트 데이터를 벡터화하여 추가한 데이터 프레임을 반환합니다.
        """
        num2txt = ["Zero", "One", "Two", "Three", "Four", "Five"]
        users_ = users.copy()
        books_ = books.copy()
        nan_value = "None"
        books_["summary"] = (
            books_["summary"]
            .fillna(nan_value)
            .apply(lambda x: TextData.__text_preprocessing(x))
            .replace({"": nan_value, " ": nan_value})
        )

        books_["summary_length"] = books_["summary"].apply(lambda x: len(x))
        books_["review_count"] = books_["isbn"].map(ratings["isbn"].value_counts())

        users_["books_read"] = users_["user_id"].map(
            ratings.groupby("user_id")["isbn"].apply(list)
        )

        if vector_create:
            if not os.path.exists("./data/text_vector"):
                os.makedirs("./data/text_vector")

            print("Create Item Summary Vector")
            book_summary_vector_list = []
            for title, summary in tqdm(
                    zip(books_["book_title"], books_["summary"]), total=len(books_)
            ):
                # 책에 대한 텍스트 프롬프트는 아래와 같이 구성됨
                # '''
                # Book Title: {title}
                # Summary: {summary}
                # '''
                prompt_ = f"Book Title: {title}\n Summary: {summary}\n"
                vector = TextData.__text_to_vector(prompt_, tokenizer, model)
                book_summary_vector_list.append(vector)

            book_summary_vector_list = np.concatenate(
                [
                    books_["isbn"].values.reshape(-1, 1),
                    np.asarray(book_summary_vector_list, dtype=np.float32),
                ],
                axis=1,
            )

            np.save(
                "./data/text_vector/book_summary_vector.npy", book_summary_vector_list
            )

            print("Create User Summary Merge Vector")
            user_summary_merge_vector_list = []
            for books_read in tqdm(users_["books_read"]):
                if not isinstance(books_read, list) and pd.isna(
                        books_read
                ):  # 유저가 읽은 책이 없는 경우, 텍스트 임베딩을 0으로 처리
                    user_summary_merge_vector_list.append(np.zeros((768)))
                    continue

                read_books = books_[books_["isbn"].isin(books_read)][
                    ["book_title", "summary", "review_count"]
                ]
                read_books = read_books.sort_values(
                    "review_count", ascending=False
                ).head(
                    5
                )  # review_count가 높은 순으로 5개의 책을 선택
                # 유저에 대한 텍스트 프롬프트는 아래와 같이 구성됨
                # DeepCoNN에서 유저의 리뷰를 요약하여 하나의 벡터로 만들어 사용함을 참고 (https://arxiv.org/abs/1701.04783)
                # '''
                # Five Books That You Read
                # 1. Book Title: {title}
                # Summary: {summary}
                # ...
                # 5. Book Title: {title}
                # Summary: {summary}
                # '''
                prompt_ = f"{num2txt[len(read_books)]} Books That You Read\n"
                for idx, (title, summary) in enumerate(
                        zip(read_books["book_title"], read_books["summary"])
                ):
                    summary = summary if len(summary) < 100 else f"{summary[:100]} ..."
                    prompt_ += f"{idx + 1}. Book Title: {title}\n Summary: {summary}\n"
                vector = TextData.__text_to_vector(prompt_, tokenizer, model)
                user_summary_merge_vector_list.append(vector)

            user_summary_merge_vector_list = np.concatenate(
                [
                    users_["user_id"].values.reshape(-1, 1),
                    np.asarray(user_summary_merge_vector_list, dtype=np.float32),
                ],
                axis=1,
            )

            np.save(
                "./data/text_vector/user_summary_merge_vector.npy",
                user_summary_merge_vector_list,
            )

        else:
            print("Check Vectorizer")
            print("Vector Load")
            book_summary_vector_list = np.load(
                "./data/text_vector/book_summary_vector.npy", allow_pickle=True
            )
            user_summary_merge_vector_list = np.load(
                "./data/text_vector/user_summary_merge_vector.npy", allow_pickle=True
            )

        book_summary_vector_df = pd.DataFrame({"isbn": book_summary_vector_list[:, 0]})
        book_summary_vector_df["book_summary_vector"] = list(
            book_summary_vector_list[:, 1:].astype(np.float32)
        )
        user_summary_vector_df = pd.DataFrame(
            {"user_id": user_summary_merge_vector_list[:, 0]}
        )
        user_summary_vector_df["user_summary_merge_vector"] = list(
            user_summary_merge_vector_list[:, 1:].astype(np.float32)
        )

        books_ = pd.merge(books_, book_summary_vector_df, on="isbn", how="left")
        users_ = pd.merge(users_, user_summary_vector_df, on="user_id", how="left")

        return users_, books_

    class __Dataset(Dataset):
        def __init__(
                self,
                user_book_vector,
                user_summary_vector,
                book_summary_vector,
                rating=None,
        ):
            """
            Parameters
            ----------
            user_book_vector : np.ndarray
                벡터화된 유저와 책 데이터를 입렵합니다.
            user_summary_vector : np.ndarray
                벡터화된 유저에 대한 요약 정보 데이터를 입력합니다.
            book_summary_vector : np.ndarray
                벡터화된 책에 대한 요약 정보 데이터 입력합니다.
            label : np.ndarray
                정답 데이터를 입력합니다.
            ----------
            """
            self.user_book_vector = user_book_vector
            self.user_summary_vector = user_summary_vector
            self.book_summary_vector = book_summary_vector
            self.rating = rating

        def __len__(self):
            return self.user_book_vector.shape[0]

        def __getitem__(self, i):
            return (
                {
                    "user_book_vector": torch.tensor(
                        self.user_book_vector[i], dtype=torch.long
                    ),
                    "user_summary_vector": torch.tensor(
                        self.user_summary_vector[i], dtype=torch.float32
                    ),
                    "book_summary_vector": torch.tensor(
                        self.book_summary_vector[i], dtype=torch.float32
                    ),
                    "rating": torch.tensor(self.rating[i], dtype=torch.float32),
                }
                if self.rating is not None
                else {
                    "user_book_vector": torch.tensor(
                        self.user_book_vector[i], dtype=torch.long
                    ),
                    "user_summary_vector": torch.tensor(
                        self.user_summary_vector[i], dtype=torch.float32
                    ),
                    "book_summary_vector": torch.tensor(
                        self.book_summary_vector[i], dtype=torch.float32
                    ),
                }
            )
