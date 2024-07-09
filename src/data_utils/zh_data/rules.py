import re
import jieba
from zhon import hanzi
from hanziconv import HanziConv
import string 

def jieba_tokenize(text):
    return list(jieba.cut(text))

def filter_too_short(string):
    tokenized = jieba_tokenize(string)
    return len(tokenized) >= 8

EMOJI_REGEX = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "\\*\u20e3"
    "#\u20e3"
    "]+",
    flags=re.UNICODE,
)
COLON_REGEX = re.compile(r"[:\s]{4,}")
def remove_emoji(text):
    text = EMOJI_REGEX.sub(r"", text)
    text = COLON_REGEX.sub(r"", text)
    return text.strip()

OTHER_BRACKETS = re.compile(r"【(.*?)】|「(.*?)」|『(.*?)』|<(.*?)>")
def remove_other_brackets(text: str):
    '''针对知乎数据：如 有哪些年轻人「千万不能碰」的东西 
        将【】,「」,『』,<>删除，保留中间内容'''
    return OTHER_BRACKETS.sub("\\1\\2\\3\\4", text)

def to_simplified(text):
    return HanziConv.toSimplified(text)

def punc_regularized(text):
    '''如果有多个标点符号连在一起，调整为最后一个出现的符号'''
    punctuation = string.punctuation + hanzi.punctuation
    return re.sub(r"(?!</)([%s]+)([%s])" %(punctuation, punctuation), "\\2", text)

EMAIL_ADDRESS = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
def remove_email(text):
    text = EMAIL_ADDRESS.sub(r"", text)
    return text.strip() 

IP_ADDRESS = re.compile(r"(\d{1,3}\.){3}\d{1,3}")
def remove_ip(text):
    text = IP_ADDRESS.sub(r"", text)
    return text.strip()

PHONE_NUM = re.compile(r"\d{11,}")
def remove_phone_number(text):
    text = PHONE_NUM.sub(r"", text)
    return text.strip() 

QQ_NUM = re.compile(r"[qQ]{2,}\d{5,12}\D")
def remove_qq_number(text):
    text = PHONE_NUM.sub(r"", text)
    return text.strip() 

HTML_TAG = re.compile(r"<[^>]+>")
BR_TAG = re.compile(r"<br>")
NBSP_TAG = re.compile(r"&nbsp")
def remove_html_tags(text):
    text = HTML_TAG.sub("", text)
    text = BR_TAG.sub("", text)
    text = NBSP_TAG.sub("", text)
    return text.strip()