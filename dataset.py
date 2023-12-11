import typing as tp

import torch
import pandas as pd
import numpy as np

from utils import NaiveNormalizer



class TimeSeqSigment(torch.utils.data.Dataset):
    def __init__(self, data:tp.Union[str,pd.DataFrame], feature_columns:tp.List[str], label_column:str, window_width:int):
        
        if isinstance(data, str):
            _df = pd.read_excel(data)
        elif isinstance(data, pd.DataFrame):
            _df = data
        else:
            raise RuntimeError(f"unknown datatype: {type(data)}")
        
        
        self._df = _df.iloc[1:-1].reset_index(drop=True)
        self._feature_columns = feature_columns
        self._label_column = label_column
        self._window_width = window_width-1

        self.__starting_indices = self._df.index[:-self._window_width]
        self.__ending_indices = self._df.index[self._window_width:]


        all_X = [self.__getitem(idx)[0] for idx in range(len(data))]
        all_X = np.stack(all_X)
        self._normalizer = NaiveNormalizer.fit(all_X)


    def __len__(self):
        return len(self.__starting_indices)
    
    def __getitem__(self, idx):
        data, label = self.__getitem(idx)
        normalized_data = self._normalizer(data)
        normalized_data = normalized_data[np.newaxis, :]
        return normalized_data, label
    
    def __getitem(self, idx):
        start = self.__starting_indices[idx]
        end = self.__ending_indices[idx]
        data = self._df.loc[start: end, self._feature_columns].to_numpy()
        label = self._df.loc[end, self._label_column]
        return data, label


class TimeSeqSigmentSign(TimeSeqSigment):
    def __init__(self, data:tp.Union[str,pd.DataFrame], feature_columns:tp.List[str], label_column:str,window_width:int):
        super().__init__(data, feature_columns, label_column, window_width)
        old_label_column = self._label_column
        new_label_column = f"{old_label_column}_sign"
        self._df[new_label_column] = self._df.apply(lambda row: 1 if row[old_label_column]>0 else 0, axis=1)
        self._label_column = new_label_column