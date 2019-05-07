
import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp.data import PadSequence
from model.utils import JamoTokenizer
from typing import Tuple

class Corpus(Dataset):
    """
    Corpus class
    """

    def __init__(self, filepath: str, tokenizer : JamoTokenizer, padder : PadSequence) -> None:
        """
        Instantiating Corpus class

        Args:
            filepath (str) : filepath
            padder (gluonnlp.data.PadSequence) : instance of gluonnlp.data.PadSequence
        """

        self._corpus = pd.read_csv(filepath, sep='\t'.loc[:,['document', 'label']])
        self._padder = padder
        self._tokenizer = tokenizer


    def __len__(self) -> int:
        return len(self._corpus)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized2indices = self._tokenizer.tokenize_and_transform(self._corpus.iloc[idx]['documnet'])
        tokenized2indices = torch.tensor(self._padder(tokenized2indices))

        label = torch.tensor(self._corpus.iloc[idx]['label'])

        return tokenized2indices, label


#-------------------------
# gluonnlp - 상세 내용은 좀 더 확인 필요 / 참고 링크 - https://gist.github.com/haven-jeon/6b508f4547418ab26f6e56b7a831dd9a

#-------------------------
