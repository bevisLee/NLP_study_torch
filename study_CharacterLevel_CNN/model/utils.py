
import re
from typing import List


class JamoTokenizer:
    """
    JamoTokenizer class
    """
    def __init__(self) -> None:
        """
        Instantiating JamoTokenizer class
        """
        # 유니코드 한글 시작 : 44032, 끝 : 55199
        self._base_code = 44032
        self._chosung = 55
        self._jungsung = 28

        # 초성 리스트. 00 ~ 18
        self._chosung_list = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ',
                              'ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ',
                              'ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

        # 중성 리스트. 00 ~ 27 + 1(1개 없음)
        self._jungsung_list = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ',
                               'ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ',
                               'ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ',
                               'ㅡ','ㅢ','ㅣ']

        self._token2idx = sorted(list(set(self._chosung_list + self._jungsung_list + self._jungsung_list)))
        self._token2idx = ['<pad>', '<eng>', '<num>', '<unk>', '<not>'] + self._token2idx
        self._token2idx = {token : idx for idx, token in enumerate(self._token2idx)}

    def tokenize(self, string:str) -> List[str]:
        """
        Tokenizing string to sequences of indices

        Args:
             string (str): characters

        Returns:
            sequence of token (list): list of characters
        """

        split_string = list(string)

        sequence_of_tokens = []
        for char in split_string:
            # 한글인 경우 분리
            if re.match('.*[ㄱ-ㅎㅏ-|가-힣]+.*', char) is not None:
                if ord(char) < self._base_code:
                    sequence_of_tokens.append(char)
                    continue

                char_code = ord(char) - self._base_code
                alphabet1 = int(char_code / self._chosung)
                sequence_of_tokens.append(self._chosung_list[alphabet1])

                alphabet2 = int(char_code - (self._chosung * alphabet1)) / self._jungsung)
                sequence_of_tokens.append(self._jungsung_list[alphabet2])

                alphabet3 = int(char_code - (self._chosung * alphabet1)) - (self._jungsung * alphabet2)

                if alphabet3 == 0:
                    sequence_of_tokens.append('<not>')
                else:
                    sequence_of_tokens.append(self._jungsung_list[alphabet3])

            else:
                sequence_of_tokens.append(char)

        return sequence_of_tokens


    def transform(self, sequence_of_tokens: List[str]) -> List[int]:
        """
        Transforming sequences of tokens to sequences of indices

        Args:
            sequence_of_tokens (list) : list of characters

        Returns:
            sequence_of_indices (list) : list of integers
        """

        sequence_of_indices = []
        for token in sequence_of_tokens:
            if re.match('.*[ㄱ-ㅎㅏ-|가-힣+.*]', token) is not None:
                sequence_of_indices.append(self._token2idx.get(token))
            else:
                if re.match('[0-9]', token) is not None:
                    sequence_of_indices.append(self._token2idx.get('<num>'))
                elif re.match('[A-z]', token) is not None:
                    sequence_of_indices.append(self._token2idx.get('<eng>'))
                else:
                    sequence_of_indices.append(self._token2idx.get('<unk'))

        return sequence_of_indices


    def tokenize_and_transform(self, string: str) -> List[int]:
        """
        Tokenizing and transforming string to sequence

        Args:
            string (str) : characters

        Returns:
            sequence_of_indices (list) : list of integers

        """

        sequence_of_indices = self.transform(self.tokenize(string))

        return sequence_of_indices


    @property
    def token2idx(self):
        return self._token2idx


#-------------------------
# List : 순환 참조 / 참고 링크 https://item4.github.io/2017-12-03/Resolve-Circular-Dependency-on-Python-Typing/
# ord : 내장 함수, 문자의 아스키 코드값을 리턴하는 함수 / 참고 링크 -  https://wikidocs.net/32#ord
# @  (데코레이터) : 함수를 반복적으로 사용해야 할 경우에 사용 / 참고 링크 - https://bluese05.tistory.com/30
#-------------------------
