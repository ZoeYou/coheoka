"""
sample randomly 100k patent applications (abstracts and claim sets) for entity grid and coherence probability evaluation
"""
import os, json, csv, psutil, sys
import random
from glob import glob
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re
import multiprocessing
from nltk.tokenize import word_tokenize

csv.field_size_limit(sys.maxsize)


def extract_data(filename):
    res = {}
    with open(filename) as json_file:
        data = json.load(json_file)

    res['date'] = data['filing_date']
    res['decision'] = data['decision']
    res['domain'] = data['main_ipcr_label']
    res['claims'] = data['claims']
    res['abstract'] = data['abstract']
    return res


#=======================================================================================#
df_dict = defaultdict(list)
domain_counter = {'A': 0, 'G': 0}
target_domains = ['A', 'G']

# sample 500 accepted patents filed in 2017 and 2018 (40 for each domain)
with open('../../data/data_15_18.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['date', 'decision', 'domain', 'claims', 'abstract'])
    cnt = 0
    for row in tqdm(reader):
        domain, decision = row['domain'][0], row['decision']
        if decision == 'ACCEPTED' and domain in target_domains and row['date'][:4] in ['2015', '2016', '2017', '2018'] and ('(canceled)' not in row['claims']):
            df_dict['domain'].append(domain)
            df_dict['claims'].append(row['claims'])

            domain_counter[domain] += 1
            cnt += 1
            if cnt % 1000 == 0:
                print('domain counter:', domain_counter)

            if domain_counter[domain] >= 5000:
                target_domains.remove(domain)
            if target_domains == []:
                break

df = pd.DataFrame(df_dict)
df.to_csv('./corpus/training_corpus.csv', index=False)
#=======================================================================================#

#=======================================================================================#
# # statistics 
# df = pd.read_csv('./corpus/training_corpus.csv')
# domains = ['A', 'G']
# for d in domains:
#     sub_df = df[df['domain']==d]
#     nb_claims = sub_df['claims'].apply(lambda c: len(re.findall('\d+\. [AT]', c)))
#     nb_words_claims = sub_df['claims'].apply(lambda x: len(word_tokenize(x)))
#     print('domain:', d)
#     print('number of claims:', sum(nb_claims)/len(nb_claims))
#     print('number of words of claims:', sum(nb_words_claims)/len(nb_words_claims))
#=======================================================================================#
