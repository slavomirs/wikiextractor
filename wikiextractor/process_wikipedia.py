import re
import glob
import nltk
import gensim
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm
from uuid import uuid4
from multiprocessing import Pool, cpu_count

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub(' ', string)


def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case 
    return re.sub('\s+',' ',string).strip().lower()

def clean_string(string,
                 min_len=2,
                 max_len=30):

    string = _remove_non_printed_chars(string)
    string = _trim_string(string)
    # also remove short words, most likely containing addresses / crap / left-overs / etc remaining after removal
    # gensim mostly does the same as above, it is used here for simplicity
    string = ' '.join(gensim.utils.simple_preprocess(string,
                                                     min_len=min_len,
                                                     max_len=max_len))
    return string

def splitkeepsep(s, sep):
    cleaned = []
    s = re.split("(%s)" % re.escape(sep), s)
    for _ in s:
        if _!='' and _!=sep:
            cleaned.append(sep+_)
    return cleaned

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_chars(text,char_list):
    for char in char_list:
        text=text.replace(char,'')
    return text.replace(u'\xa0', u' ')   

def process_wiki_files(wiki_file):
    chars = ['\n']
    global sw
    
    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()

    articles = splitkeepsep(content,'<doc id=')
    df = pd.DataFrame(columns=['article_uuid','sentence','proc_sentence','proc_len'])
    
    for article in articles:
        uuid = uuid4()
        
        article = remove_special_chars(remove_html_tags(article),
                                       chars)        
        
        sentences = nltk.sent_tokenize(article)
        proc_sentences = [clean_string(sentence) for sentence in sentences]
        proc_lens = [len(sentence.split(' ')) for sentence in proc_sentences]
        
        temp_df = pd.DataFrame(
            {'article_uuid': [uuid]*len(sentences),
             'sentence': sentences,
             'proc_sentence':proc_sentences,
             'proc_len':proc_lens
            })
        df = df.append(temp_df)
    
    return df

def list_multiprocessing(param_lst,
                         func,
                         **kwargs):
    
    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]

def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)


if __name__ == '__main__':
    nltk.download('punkt')
    wiki_files = []

    for filename in glob.iglob('data/wiki/*/*', recursive=True):
        wiki_files.append(filename)

    df = list_multiprocessing(wiki_files,
                            process_wiki_files,
                            workers=cpu_count() - 1)

    df = pd.concat(df).reset_index(drop=True)
    df.article_uuid = df.article_uuid.astype(str)
    df.to_csv('data/skwiki_2024_03_29.csv')