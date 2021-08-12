#!/usr/bin/env python
# coding: utf-8

### Imports

import os, re

import spacy
from spacy.tokens import Doc, Span, Token
from spacy import displacy

import IPython
from IPython.display import display, Markdown

import nltk
from nltk.tree import Tree
from nltk.draw.tree import TreeView

import pandas as pd
import numpy as np

import logging

### Logging
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    
    blue = "\x1b[34;1m"
    green = "\x1b[32;1m"
    grey = "\x1b[38;1m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def init_logging():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)

### Detect jupyter notebook
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

jupyter = isnotebook()

### Debug

def tokens_infos(sent):
    
    infos = []
    for token in sent:
        infos.append([token.text, token.lemma_, token.pos_, token.morph, token._.labels, token.dep_])
    return pd.DataFrame(np.array(infos,dtype=object),
                   columns=['form', 'lemma', 'pos', 'morph', 'labels', 'dep'])

