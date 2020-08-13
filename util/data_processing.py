# coding:utf-8

import pandas as pd
import numpy as np
import re, sys
import json
import logging
from keras.preprocessing.sequence import pad_sequences
from tqdm._tqdm import tqdm
from numpy import array, zeros, ones, hstack, vstack, squeeze, empty, expand_dims
import pickle as pkl

logger = logging.getLogger(__name__)

seed=13
np.random.seed(seed)

abbr_ext = {
    u" it's ": u" it is ",
    u" that's ": u" that is ",
    u" there's ": u" there is ",
    u" here's ": u" here is ",
    u" he's ": u" he is ",
    u" she's ": u" she is ",
    u" what's ": u" what is ",
    u" who's ": u" who is ",
    u" how's ": u" how is ",
    u" where's ": u" where is ",
    u" let's ": u" let us ",
    u" won't ": u" will not ",
    u" ain't ": u" am not ",
    u" i'm ": u" i am ",
}

prtl_abbr_ext = {
    u"'d ": u" would ",
    u"'ve ": u" have ",
    u"'ll ": u" will ",
    u"n't ": u" not ",
    u"'re ": u" are ",
    u"'s ": u" 's ",
    u"' ": u" ",
}

def text_cleaner(text, uncase=True, reabbr=True):
    """
    simple preprocess
    """
    text = ' ' + text.strip() + ' '
    if uncase:
        text = text.lower()

    # Clean the text
    try:
        text = text.replace('<br>', ' ')
    except UnicodeDecodeError:
        print(text)
        raise
    if reabbr:
        text = re.sub("[,^!:;+=\.\/\(\)\"\-\?\\\n\r]", " ", text)
        # usual abbr whole word sub
        for ky, vl in abbr_ext.items():
            text = text.replace(ky, vl)
        # partial word sub
        for ky, vl in prtl_abbr_ext.items():
            text = text.replace(ky, vl)
    # blanks
    text = re.sub(u'\s+', u' ', text).strip()
    return text


def table_tokenizer(table, uncase=True):
    textTable = []
    # maxLen = 0
    for text in tqdm(table, file=sys.stdout):
        text = text_cleaner(text, uncase=uncase)
        textTable.append(text)
    return textTable
