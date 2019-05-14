

import pandas as pd
import gluonnlp as nlp
import itertools
import pickle

from pathlib import Path
from mecab import MeCab


# loading dataset
proj_dir = Path.cwd()
tr_filepath = proj_dir / 'data' / 'train.txt'
tr = pd.read_csv(tr_filepath, sep='\t').loc[:, ['document', 'lable']]

# extracting morph in sentences
tokenizer = MeCab()
tokenized = tr['document'].apply(tokenizer.morphs).tolist()

# making the vocab
counter = nlp.data.count_tokens(itertools.chain.from_iterable(tokenized))
vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

# connectiong SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
vocab.set_embedding(ptr_embedding)

# saving vocab
with open('./data/vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)



#-------------------------
# 2019.05.14
# nlp.data.count_tokens() : 지정된 문자열에서 토큰을 계
# itertools.chain.from_iterable() : 2중 리스트를 1차원 리스트로 변경 - https://winterj.me/list_of_lists_to_flatten/
#
#-------------------------
