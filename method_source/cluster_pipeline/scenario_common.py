"""Shared routing and text processing for training and inference."""
import os
from pathlib import Path
import re
import pandas as pd

ROOT = Path(os.environ.get('FOREDROID_OUTPUT_ROOT','/home/lb/data/Dataset/ForeDroid_output/2014-2017'))
MODELS = Path(os.environ.get('FOREDROID_MODELS','/home/lb/models/foredroid'))
INVALID_GUI = {'','.','nan','none','null','noapi','no_rawtext'}
PRESERVED = {'in','out','up','down','on','off','to','for','with','by','at','from','of','into','onto','over','not','yes','no'}


def valid_gui_text(value):
    return isinstance(value,str) and value.strip().lower() not in INVALID_GUI


def gui_mask(df):
    mask=pd.Series(False,index=df.index)
    for col in ['label','context','rawtext']:
        if col in df:mask |= df[col].notna() & ~df[col].astype(str).str.strip().str.lower().isin(INVALID_GUI)
    return mask


def gui_raw_text(label,context,rawtext):
    # Retain the original order: label, rawtext, context.
    parts=[value.strip() for value in [label,rawtext,context] if valid_gui_text(value)]
    return re.sub(r'\s+',' ',re.sub(r'[^a-zA-Z\s]+',' ',' '.join(parts))).strip().lower()


def gui_doc_text(doc):
    return ' '.join(token.lemma_.lower().strip() for token in doc
        if token.lemma_.strip() and (token.lemma_.lower().strip() in PRESERVED or token.pos_ in {'NOUN','VERB','ADJ','ADV','PROPN'}))
