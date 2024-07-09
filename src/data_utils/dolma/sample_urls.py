import json 
import os
from pathlib import Path
import random 


data_detail = {
    'cc': {
        'name': 'CommonCrawl',
        'tokens': '1195.5',
        'urls': []
    },
    'refinedweb':{
        'name': 'RefinedWeb',
        'tokens': '456.4',
        'urls': []
    },
    'starcoder':{
        'name': 'StarCoder',
        'tokens': '263.8',
        'urls': []
    },
    'c4': {
        'name': 'C4', 
        'tokens': '138.4',
        'urls': []
    },
    'reddit': {
        'name': 'Reddit', 
        'tokens': '79.9',
        'urls': []
    },
    'pes2o': {
        'name': 'PeS2o', 
        'tokens': '57.2',
        'urls': []
    },
    'arxiv':{
        'name': 'arXiv', 
        'tokens': '28.0',
        'urls': []
    },
    'stackexchange':{
        'name': 'StackExchange', 
        'tokens': '19.6',
        'urls': []
    },
    'tulu_flan':{
        'name': 'Flan', 
        'tokens': '16.5',
        'urls': []
    },
    'cc_news':{
        'name': 'CCNews', 
        'tokens': '14.3',
        'urls': []
    },
    'open_web_math':{
        'name': 'OpenWebMath', 
        'tokens': '12.6',
        'urls': []
    },
    'algebraic_stack':{
        'name': 'AlgebraicStack', 
        'tokens': '12.6',
        'urls': []
    },
    'books': {
        'name': 'Project Gutenberg', 
        'tokens': '5.3',
        'urls': []
    },
    'wikiref_megawika': {
        'name': 'MegaWika', 
        'tokens': '4.6',
        'urls': []
    },
    'wiki': {
        'name': 'Wikipedia', 
        'tokens': '3.7',
        'urls': []
    },
}


def group_type():
    urls = 'src/data_utils/dolma/v1_7_all.txt'
    
    with open(urls, 'r') as f:
        urls = [i.split('\n')[0] for i in f.readlines()]
    
    data_type = data_detail.keys()
    
    cnt = 0
    for line in urls:
        for dt in data_type:
            if dt in line:
                if dt == 'wiki' and 'wikiref_megawika' in line:
                    continue
                if dt == 'cc' and 'cc_news' in line:
                    continue
                cnt += 1
                data_detail[dt]['urls'].append(line)
    
    for k, v in data_detail.items():
        urls = data_detail[k]['urls']
        print(f'name:{k} num of url:{len(urls)}')
    
    json.dump(data_detail, open('src/data_utils/dolma/grouped_urls.json', 'w'), indent=1)


def sample_urls():
    data_url_num = {
        # 网页数据
        'cc': 3, # 一个文件1B tokens左右
        'refinedweb': 2, 
        'c4': 2,
        'cc_news': 2,
        
        # 代码相关
        'starcoder': 0, # 一个文件5B tokens左右
               
        'reddit': 1, # 一个文件1B tokens左右
        
        # 学术论文相关
        'pes2o': 0, # 一个文件2B tokens左右
        'arxiv': 4, # 一个文件0.3B tokens左右
        
        # 数学相关
        'stackexchange': 1, # 也有代码 一个文件1B tokens左右
        'tulu_flan': 0, # 一个文件0.3B tokens左右
        'open_web_math': 1, # 一个文件1B tokens左右
        'algebraic_stack': 0, # 一个文件1B tokens左右
        
        # 维基百科相关
        'books': 2, # 一个文件2B tokens左右
        'wiki': 2, # 一个文件2B tokens左右
        'wikiref_megawika': 0, # 一个文件0.02B tokens左右
    }
    data_detail = json.load(open('src/data_utils/dolma/grouped_urls.json', 'r'))
    
    url_to_download = []
    for dt, num in data_url_num.items():
        urls = data_detail[dt]['urls']
        
        url_to_download.extend(random.sample(urls, num))
        
    with open('src/data_utils/dolma/sampled_urls.txt', 'w') as f:
        for item in url_to_download:
            f.write(item + '\n')


if __name__ == "__main__":
    group_type()
    sample_urls()