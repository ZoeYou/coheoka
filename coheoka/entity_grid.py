# -*- coding: utf-8 -*-
'''
Build entity grid using StanfordCoreNLP
Reference: Barzilay, R., & Lapata, M. (2008).
    Modeling local coherence: An entity-based approach.
    Computational Linguistics, 34(1), 1-34.
'''
from __future__ import print_function, division
from collections import defaultdict
from functools import reduce
from pprint import pprint
import doctest
import os

import pandas as pd


import spacy
import neuralcoref




# Load spaCy's language model
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

class Constants(object):
    REMOVE_ABBR = {'Inc.', 'Inc', 'Corp.', 'Corp'}
    _NOUNS = {'NN', 'NNS', 'NNP', 'NNPS', 'PRP'}
    SUB, OBJ, OTHER, NOSHOW = 'S', 'O', 'X', '-'

    @staticmethod
    def noun_tags():
        """Get noun POS tags"""
        return Constants._NOUNS

    @staticmethod
    def get_role(dep):
        """Indentify an entity's grammatical role"""
        if 'subj' in dep:
            return Constants.SUB
        elif 'obj' in dep:
            return Constants.OBJ
        else:
            return Constants.OTHER



class EntityGrid(object):
    '''
    Entity grid
    >>> eg = EntityGrid('My friend is Bob. He loves playing basketball.')
    >>> 'friend' in eg.grid.columns and 'bob' in eg.grid.columns
    True
    >>> 'he' not in eg.grid.columns
    True
    '''

    def __init__(self, text):
        self.text = ' '.join([token
                              for token in text.split(' ')
                              if token not in Constants.REMOVE_ABBR])
        self.doc = nlp(self.text)
        self.resolved_text = self._resolve_coreferences()
        self.resolved_doc = nlp(self.resolved_text)  # Re-process resolved text to create a new doc
        self._grid = self._set_up_grid()

    def _resolve_coreferences(self):
        """Resolve coreferences in the text using NeuralCoref, making entity references explicit."""
        return self.doc._.coref_resolved if self.doc._.has_coref else self.doc.text


    def _set_up_grid(self):
        """Set up the entity grid based on the resolved text."""
        grid = defaultdict(lambda: [Constants.NOSHOW for _ in range(len(list(self.resolved_doc.sents)))])
        
        for i, sentence in enumerate(self.resolved_doc.sents):
            for token in sentence:
                if token.pos_ in ['NOUN', 'PROPN', 'PRON']:  # Include nouns, proper nouns, and pronouns
                    entity = token.lemma_.lower() if token.pos_ in ['NOUN', 'PROPN'] else token.text.lower()
                    role = Constants.get_role(token.dep_)
                    grid[entity][i] = role

        # Transpose the grid so that rows represent sentences and columns represent entities
        return pd.DataFrame(grid, index=[f"Sentence {i+1}" for i in range(len(list(self.resolved_doc.sents)))])


    @property
    def grid(self):
        """Entity grid"""
        return self._grid



if __name__ == '__main__':
    doctest.testmod()