# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from io import StringIO
from typing import Iterable, List, Optional, Union

import penman

from supar.utils import Field
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence, Transform

from .utils import decode_amr, tokenize_encoded_graph, noop_model


class AMR(Transform):

    fields = ['SRC', 'TGT', 'POS', 'HEAD', 'DEPREL']

    def __init__(
        self,
        SRC: Optional[Union[Field, Iterable[Field]]] = None,
        TGT: Optional[Union[Field, Iterable[Field]]] = None,
        POS: Optional[Union[Field, Iterable[Field]]] = None,
        HEAD: Optional[Union[Field, Iterable[Field]]] = None,
        DEPREL: Optional[Union[Field, Iterable[Field]]] = None,
    ) -> AMR:
        super().__init__()

        self.SRC = SRC
        self.TGT = TGT

        self.POS = POS
        self.HEAD = HEAD
        self.DEPREL = DEPREL

    @property
    def src(self):
        return self.SRC, self.POS, self.HEAD, self.DEPREL

    @property
    def tgt(self):
        return self.TGT,

    def decode_graph(self, tokens):
        tokens = self.TGT.postprocess(tokens)
        graph, state = decode_amr(tokens)
        return penman.encode(graph), state

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> Iterable[AMRSentence]:
        r"""
        Loads the data in Text-X format.
        Also supports for loading data from Text-U file with comments and non-integer IDs.

        Args:
            data (str or Iterable):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`TextSentence` instances.
        """

        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data, str) and os.path.exists(data):
            f = open(data)
            if data.endswith('.txt'):
                lines = (i
                         for s in f
                         if len(s) > 1
                         for i in StringIO((s.split() if lang is None else tokenizer(s)) + '\n'))
            else:
                lines = f
        else:
            if lang is not None:
                data = [tokenizer(s) for s in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = (i for s in data for i in StringIO(s + '\n'))

        index, sentence = 0, []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                sentence = AMRSentence(self, sentence, index)
                yield sentence
                index += 1
                sentence = []
            else:
                sentence.append(line)


class AMRSentence(Sentence):
    def __init__(self, transform: AMR, lines: List[str], index: Optional[int] = None) -> AMRSentence:
        super().__init__(transform, index)
        
        try:
            amr = penman.decode('\n'.join(lines), model=noop_model)
        except Exception:
            print('\n'.join(lines))
        self.metadata = amr.metadata
        amr.metadata = {}
        self.cands = [tokenize_encoded_graph(penman.encode(amr))]
        # wid偶尔存在最后一个词是空，需要去除
        wp = self.metadata['wid'].split()
        if wp[-1].endswith('_'):
            wp = wp[:-1]
        wid = ' '.join([x.replace('_', ' ') for x in wp])
        self.values = [wid, self.cands[0]]
        
        if 'pos' in self.metadata.keys():
            self.values.append(self.metadata['pos'].split())
        else:
            self.values.append([])
            
        if 'arc' in self.metadata.keys():
            self.values.append([int(i) for i in self.metadata['arc'].split()])
        else:
            self.values.append([])
            
        if 'rel' in self.metadata.keys():
            self.values.append(self.metadata['rel'].split())
        else:
            self.values.append([])

    def __repr__(self):
        amr = penman.decode(self.values[1])
        amr.metadata = self.metadata
        return penman.encode(amr) + '\n'

