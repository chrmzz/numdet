#!/usr/bin/env python
# coding: utf-8

# imports
import re
import sys
import logging

import spacy
import benepar

# models
nlp = spacy.load("fr_core_news_lg")
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_fr2"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_fr2", "disable_tagger": True})

# helpers
from helpers.debug import init_logging, tokens_infos

logger = logging.getLogger(__package__)
init_logging()

# usage
if len(sys.argv) < 2:
    print("Usage: {} file.txt".format(sys.argv[0]))
    sys.exit(2)
input_filename = sys.argv[1]

# doc
with open(input_filename) as f:
    text = f.read()
    text = text.replace('\n', ' ')
    doc = nlp(text)


# numerical data detection
# dates

def detect_date(doc):
    for match in re.finditer(r"([0-3][0-9]) (janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)( [12][0-9][0-9][0-9])?", doc.text): # 21 juillet 2021, 21 juillet
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"([0-3][0-9])\/([01][0-9])\/([12]?[0-9]?[0-9][0-9])", doc.text): # 21/07/2021, 21/07/21
        start, end = match.span()
        span = doc.char_span(start, end) 
        if span is not None:
            yield start, end, span.text
 
    for match in re.finditer(r"([0-3][0-9]) (jan\.?|févr\.?|avr\.?|juill\.?|sept\.?|oct\.?|nov\.?|déc\.?) ?([12][0-9][0-9][0-9])?", doc.text): # 21 juill. 2021, 21 juill 2021, 21 juill., 21 juill
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# time

def detect_time(doc):
    for match in re.finditer(r"[0-2][0-9][hH]([0-5][0-9])?", doc.text): # 13h43, 13h, 13H43, 13H
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"[0-2][0-9]:[0-5][0-9](:[0-5][0-9])?", doc.text): # 13:43, 13:43:45
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# percent

def detect_percent(doc):
    for match in re.finditer(r"(\d{1,4})([,\.]\d{1,2})?\%", doc.text): # 2% 2,3% 2.3% 25% 25,4% 234% 234,56% 1234% 1234,56%
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text
    
    for match in re.finditer(r"(\d{1,4})([,\.]\d{1,2})? (p\.cent|p\. cent|p\. 100)", doc.text): # 25 p. cent/25 p.cent/25 p. 100
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# currency

def detect_currency(doc):
    for match in re.finditer(r"(\d+)([,\.]\d{1,2})? (millions?|milliards?|billions?)? ?(de|d'|d’)? ?(euros?|dollars?|livres?|yen)", doc.text): # 1 euro, 3,5 dollars, 4 millions de livres 
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"(\d+)([,\.]\d{1,2})? ?[€$£¥]", doc.text): # 5 €, 5€, 10,5 £, 15.00 $, 235 ¥
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"(\d+)([,\.]\d{1,2})? (millions?|milliards?|billions?) ?(de|d'|d’)? ?[€$£¥]", doc.text): # 1 million $, 3 milliards d'€, 4,56 millions de ¥
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# temperature

def detect_temperature(doc):
    for match in re.finditer(r"(\d{1,2})([,\.]\d{1,2})?°[CF]?", doc.text): # 25°C, 25°F, 25°, 23,4°
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# measure

def detect_mesure(doc):
    for match in re.finditer(r"(\d+)([,\.]\d+)? ([cdm]l|[GMk]o|Gy|ha|M?Hz|k[gW]|km(\/h)?|L|l[bmx]?|m(\/s|t)?|m|mol|oz?|Pa|po|s|t|Wb)", doc.text): # 58 cl, 54,345 dl, 3456 Go, Gy, ha, Hz, kg, km, km/h, kW, ko, l, L, lb, lm, lx, m, m/s, MHz, ml, Mo, mol, mt, o, oz, Pa, po, s, t, Wb
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# fractions, cardinal and ordinal numerals

def detect_num(doc):
    for match in re.finditer(r"(\d+)(ème|er|e|ère|nde|nd)", doc.text): # 1e, 1er, 1ère, 2e, 2nd, 2nde, 3e, 3ème, 45ème, 1000ème
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"(\d{0,3}) (\d{0,3}) (\d{0,3}) (\d{1,3}) (\d{1,3})", doc.text): # 1 758, 1 758 625, 758 625 758 625, jusqu'à 999 billions
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"(\d+)([,\.\/]\d+)?", doc.text): # 345, 23,2345, 1.2, 56/8
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# cardinal and ordinal numerals in letters

def detect_alphanum(doc):
    for match in re.finditer(r"\b(deux-|trois-|quatre-|cinq-|six-|sept-|huit-|neuf-)?(mille-)?(deux-|trois-|quatre-|cinq-|six-|sept-|huit-|neuf-)cent-(quatre-vingts|quatre-vingt(-et-un|-deux|-trois|-quatre|-cinq|-six|-sept|-huit|-neuf|-onze|-douze|-treize|-quatorze|-quince|-seize|-dix-(sept|huit|neuf)|-dix)|soixante(-et-onze|-douze|-treize|-quatorze|-quince|-seize|-dix-(sept|huit|neuf))?|soixante-dix|(vingt|trente|quarante|cinquante|soixante)(-et-un|-deux|-trois|-quatre|-cinq|-six|-sept|-huit|-neuf)?)\b", doc.text): # deux-cent-vingt à neuf-cent-quatre-vingt-dix-neuf, mille-deux-cent-vingt à neuf-mille-neuf-cent-quatre-vingt-dix-neuf
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(deux-|trois-|quatre-|cinq-|six-|sept-|huit-|neuf-)?(mille-)(onze|douze|treize|quatorze|quince|seize|dix-(sept|huit|neuf)|dix)\b", doc.text): # mille-onze à mille-dix-neuf, deux-mille-onze à deux-mille-dix-neuf, ...
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(deux-|trois-|quatre-|cinq-|six-|sept-|huit-|neuf-)?(mille-)(un|deux|trois|quatre|cinq|six|sept|huit|neuf)\b", doc.text): # mille-un à mille-dix, deux-mille-un à deux-mille-dix, ...
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(mille-)?(deux-|trois-|quatre-|cinq-|six-|sept-|huit-|neuf-)(cents?)\b", doc.text): # deux-cents/deux-cent + NOUN à neuf-cents/neuf-cent + NOUN, mille-deux-cents à mille-neuf-cents
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(mille-)?cent|(deux-|trois-|quatre-|cinq-|six-|sept-|huit-|neuf-)?mille\b", doc.text): # cent, mille, mille-cent, deux-mille, trois-mille, ...
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(mille-)?(cent-)?(quatre-vingts|quatre-vingt(-et-un|-deux|-trois|-quatre|-cinq|-six|-sept|-huit|-neuf|-onze|-douze|-treize|-quatorze|-quince|-seize|-dix-(sept|huit|neuf)|-dix)|soixante(-et-onze|-douze|-treize|-quatorze|-quince|-seize|-dix-(sept|huit|neuf))?|soixante-dix|(vingt|trente|quarante|cinquante|soixante)(-et-un|-deux|-trois|-quatre|-cinq|-six|sept|-huit|-neuf)?)\b", doc.text): # vingt à quatre-vingt-dix-neuf, cent-vingt à cent-quatre-vingt-dix-neuf, mille-vingt à mille-quatre-vingt-dix-neuf
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(mille-)?(cent-)?(onze|douze|treize|quatorze|quince|seize|dix-(sept|huit|neuf)|dix)\b", doc.text): # once à dix-neuf, cent-onze à cent-dix-neuf, mille-onze à mille-dix-neuf
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(mille-)?(cent-)?(un|deux|trois|quatre|cinq|six|sept|huit|neuf)\b", doc.text): # cent-un à cent-neuf, mille-un à mille-neuf
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"\b(zéro|deux|trois|quatre|cinq|six|sept|huit|neuf)\b", doc.text): # zéro, deux, trois, quatre, cinq, six, huit, neuf
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"premi(er|ère)|seconde?|\b[a-zA-Z-]+i?ème\b", doc.text): # premier, première, second, seconde, troisième, dix-huitième, millième, ...
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text


# roman numerals

def detect_roman_num(doc):
    for match in re.finditer(r"((M{1,4}(CM|(CD|(D?C{0,3})))?(XC|(XL|(L?X{0,3})))?(IX|(IV|(V?I{0,3})))?)|((CM|(CD|(DC{0,3})|(C{1,3})))(XC|(XL|(L?X{0,3})))?(IX|(IV|(V?I{0,3})))?)|((XC|(XL|(LX{0,3})|(X{1,3})))(IX|(IV|(V?I{0,3})))?)|((IX|(IV|(VI{0,3})|(I{1,3})))))(ème|er|e|ère|nde|nd)", doc.text): # Ier, Ière, IInd, IInde, IVème, Vème, VIIème, XIXème, Ie, Ve, XIXe
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"(M{1,4}(CM|(CD|(D?C{0,3})))?(XC|(XL|(L?X{0,3})))?(IX|(IV|(V?I{0,3})))?)|((CM|(CD|(DC{0,3})|(C{1,3})))(XC|(XL|(L?X{0,3})))?(IX|(IV|(V?I{0,3})))?)|((XC|(XL|(LX{0,3})|(X{1,3})))(IX|(IV|(V?I{0,3})))?)|((IX|(IV|(VI{0,3})|(I{1,3}))))", doc.text): # I, II, IV, V, VII, MC, DC
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text

    for match in re.finditer(r"(m{1,4}(cm|(cd|(d?c{0,3})))?(xc|(xl|(l?x{0,3})))?(ix|(iv|(v?i{0,3})))?)|((cm|(cd|(dc{0,3})|(c{1,3})))(xc|(xl|(l?x{0,3})))?(ix|(iv|(v?i{0,3})))?)|((xc|(xl|(lx{0,3})|(x{1,3})))(ix|(iv|(v?i{0,3})))?)|((ix|(iv|(vi{0,3})|(i{1,3}))))", doc.text): # i, ii, iv, v, vii, mc, dc
        start, end = match.span()
        span = doc.char_span(start, end)
        if span is not None:
            yield start, end, span.text
    

# print of matches

for sent in doc.sents:
    logger.debug(tokens_infos(sent))

for match in detect_date(doc):
    print("date", match[2])
for match in detect_time(doc):
    print("time", match[2])
for match in detect_percent(doc):
    print("percent", match[2])
for match in detect_currency(doc):
    print("currency", match[2])
for match in detect_temperature(doc):
    print("temperature", match[2])
for match in detect_mesure(doc):
    print("mesure", match[2])
for match in detect_num(doc):
    print("number", match[2])
for match in detect_alphanum(doc):
    print("alpha", match[2])
for match in detect_roman_num(doc):
    print("roman", match[2])
