"""Text pre-processing module

Some of this code is sourced from Internet sources (given in ref below)
Ref: https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
"""

import re
from symspellpy.symspellpy import SymSpell, Verbosity

from constants import DICT_DIR

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

def clean_puncts(x):
    """ Return the text with punctuations spaced out

        Args:
            x (str): Full text which has punctuations in usual place

        Returns:
            x (str): text with punctuations spaced out
    """
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x

def remove_linebreaks(x):
    x = str(x)
    if "\n" in x:
        x = x.replace("\n", "")
    if "\t" in x:
        x = x.replace("\t", "")
    return x


def remove_puncts(x):
    """ Replace punctuations, '\n' and '...' from the text

        Args:
            x (str): Full text which has numbers

        Returns:
            x (str): text with numbers replaced by '#'
    """
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, "")
    if "\n" in x:
        x = x.replace("\n", "")
    if "..." in x:
        x = x.replace("...", "")
    return x

def clean_numbers(x):
    """ Replace numbers with '#' as most word vectors use that notation to indicate number

        Args:
            x (str): Full text which has numbers

        Returns:
            x (str): text with numbers replaced by '#'
    """
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not",
                    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                    "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
                    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
                    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",
                    "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                    "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                    "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                    "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is",
                    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
                    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                    "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                    "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    """ Replace contractions with expanded version from the dictionary.

    Usage::
    replace_contractions("this's a text with contraction")

    Args:
        text (str): text which has the contractions

    Returns:
        text (str): text with contractions replaced with expanded form
    """

    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

def remove_stop_short_words(txt):
    """ Remove stop and short words from the text.

    Usage::
    remove_stop_short_words("I run a business")

    Args:
        txt (str): Full text which has all the words

    Returns:
        txt (str): text with stop (a, an, is) and short words (less than 3 letters) removed from it
    """
    txt = gensim.parsing.preprocessing.remove_stopwords(txt)
    filtered_words = [word for word in txt.split()]
    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
    txt = " ".join(filtered_words)
    return (txt)

def symspell_dict(max_edit_dist, prefix_len):
    dictfile = DICT_DIR / "big.txt"  #downloaded from Peter Norvig's site
    sym_spell = SymSpell(max_edit_dist, prefix_len)

    #create the symspell dictionary using the dictfile
    if not sym_spell.create_dictionary(str(dictfile)):
        print("corpus file not found")
    return sym_spell

def correction(symspellobj, word):
    suggested_word = word
    suggestions = symspellobj.lookup(word, Verbosity.CLOSEST, 2)
    if suggestions:
        suggested_word = suggestions[0].term
        for suggestion in suggestions:
            if suggestion.term[0:1] == word[0:1]:
                suggested_word = suggestion.term
                break;
    return suggested_word