# -*- coding: utf-8 -*-
"""
Assess model performance
"""
from __future__ import print_function, division

import os

from nltk import sent_tokenize
from utils import replace_sents, pk_load
from evaluator import Evaluator
from coherence_probability import ProbabilityVector

import pandas as pd
import re
import pickle


class Assessment(object):
    def __init__(self, corpus, pv):#, ev):
        self.corpus = self._preprocess(corpus) + self._label_corpus(corpus)
        assert type(pv) == ProbabilityVector
        # assert type(ev) == Evaluator
        self.pv = pv
        # self.ev = ev

    def _preprocess(self, corpus):
        res = []
        for text in corpus:
            text = '. '.join([t.strip() for t in re.split(r'\.[\s\n]*', text)])
            res.append((text, 1))
        return res

    def _label_corpus(self, corpus):
        res = []
        for text in corpus:
            text = '. '.join([t.strip() for t in re.split(r'\.[\s\n]*', text)])
            remove_one = replace_sents(text, 1)[0]
            res.append((remove_one, -1))
        return res

    def assess_pv(self, text):
        if len(sent_tokenize(text)) <= 1:
            return -1
        pb = self.pv.evaluate_coherence(text)[0]
        return pb
        # if pb < self.pv.mean:
        #     return -1
        # elif self.pv.mean <= pb <= self.pv.mean + 2 * self.pv.std:
        #     return 1
        # else:
        #     return 1
        

    # def assess_ev(self, text):
    #     rank = self.ev.evaluate_coherence(text)[0]
    #     return rank
    #     # if rank < 0.2:
    #     #     return -1
    #     # elif 0.2 <= rank < 1:
    #     #     return 1
    #     # else:
    #     #     return 1

    # def assess_all(self):
    #     ev_right, pv_right, length = 0, 0, len(self.corpus)
    #     cnt = 0
    #     for text, label in self.corpus:
    #         ev_res, pv_res = None, None
    #         cnt += 1
    #         try:
    #             ev_res = self.assess_ev(text)
    #             pv_res = self.assess_pv(text)
    #         except Exception:
    #             print(text)
    #         else:
    #             print('{}/{}'.format(cnt, length))
    #         if ev_res == label:
    #             ev_right += 1
    #         if pv_res == label:
    #             pv_right += 1
    #     return ev_right / length, pv_right / length


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))

    # load training corpus
    df = pd.read_csv(os.path.join(cur_dir, 'corpus', 'training_corpus.csv'))

    claims_A = df[df['domain'] == 'A']['claims'].tolist()
    claims_G = df[df['domain'] == 'G']['claims'].tolist()

    if os.path.exists(os.path.join(cur_dir, 'pickles', 'pv_A.pkl')) and os.path.exists(os.path.join(cur_dir, 'pickles', 'pv_G.pkl')):
        # load pv_A and pv_G from pickle
        with open(os.path.join(cur_dir, 'pickles', 'pv_A.pkl'), 'rb') as f:
            pv_A = pickle.load(f)
        with open(os.path.join(cur_dir, 'pickles', 'pv_G.pkl'), 'rb') as f:
            pv_G = pickle.load(f)
    else:
        if not os.path.exists(os.path.join(cur_dir, 'pickles')):
            os.makedirs(os.path.join(cur_dir, 'pickles'))

        # make pv_A and pv_G
        pv_A = ProbabilityVector(claims_A).make_probs()
        print("pv_A made")
        pv_G = ProbabilityVector(claims_G).make_probs()
        print("pv_G made")
        
        # save pv_A and pv_G to pickle
        with open(os.path.join(cur_dir, 'pickles', 'pv_A.pkl'), 'wb') as f:
            pickle.dump(pv_A, f)
        with open(os.path.join(cur_dir, 'pickles', 'pv_G.pkl'), 'wb') as f:
            pickle.dump(pv_G, f)

    

    with open(os.path.join(cur_dir, 'corpus', 'test.txt')) as f:
        testtxt = f.read().split('////')
        assess = Assessment(testtxt[:2], pv_A)
        print(assess.assess_pv(testtxt[0]))
        print(assess.assess_pv(testtxt[1]))
        print(assess.assess_pv(testtxt[2]))
