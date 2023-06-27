# coding:utf-8
# this is a simplified version of "https://github.com/SapienzaNLP/spring/blob/main/spring_amr/tokenization_bart.py"
import os
import sys
import json
from typing import Any, List, Union
import numpy as np
import penman
import regex as re
import torch
from transformers import BertTokenizer
import networkx as nx
from transformers.tokenization_utils_base import TextInput


DEFAULT = penman.Graph(
    [
        penman.Triple("x1", ":instance", "狗"),
        penman.Triple("x2", ":instance", "叫-01"),
        penman.Triple("x1", ":arg0", "x2"),
    ]
)


class SimVocab:
    def __init__(self, vocab: dict, unk_token_id: int) -> None:
        self.vocab = vocab
        self.unk_token_id = unk_token_id
    
    def __getitem__(self, key: str):
        return self.vocab.get(key, self.unk_token_id)
    
    def __len__(self):
        return len(self.vocab)

class SimDecoder(dict):
    def __call__(self, tokens: List[int]) -> List[str]:
        return [self.get(t) for t in tokens]


class AMRBertTokenizer(BertTokenizer):
    WP = '##'

    def __init__(self, vocab_file, errors="replace", add_prefix_space=False, **kwargs):
        super().__init__(vocab_file=vocab_file, errors=errors,
                         add_prefix_space=add_prefix_space, **kwargs)
        self.modified = 0
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.remove_pars = False

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary()
        return inst

    def init_amr_vocabulary(self):
        self.old_enc_size = len(self.vocab)
        raw_special_tokens = json.load(
            open(f"{os.path.dirname(__file__)}/additional-tokens-chinese.json", "r", encoding="utf-8")
        )
        tokens = [t for t in raw_special_tokens if t not in self.vocab]

        # for i, t in enumerate(tokens, start=old_enc_size):
        # self.encoder[t] = i
        self.add_tokens(tokens)

        self.encoder = {**self.vocab, **self.added_tokens_encoder}
        self.decoder = {v: k for k, v in
                        sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)

        # self.amr_bos_token = "[CLS]"
        # self.amr_bos_token_id = self.encoder[self.amr_bos_token]
        # self.amr_eos_token = "[EOS]"
        # self.amr_eos_token_id = self.encoder[self.amr_eos_token]
        print(f"Added {self.modified} AMR tokens")

    def get_vocab(self):
        return SimVocab(super().get_vocab(), self.unk_token_id)
    
    def get_decoder(self):
        return SimDecoder(self.decoder)
    
    def _tok_wp(self, token):
        tokk = []
        for token in self.basic_tokenizer.tokenize(token, never_split=self.all_special_tokens):
            # If the token is part of the never_split set
            if token in self.basic_tokenizer.never_split:
                tokk.append(token)
            else:
                tokk += self.wordpiece_tokenizer.tokenize(token)

        return tokk
    
    def tokenize_amr(self, amr_text):
        amr_tokens = amr_text.split()
        wp_tokens = []
        for i, tokk in enumerate(amr_tokens):
            is_in_enc = tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_of = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'[^\d\s-]+-\d+', tokk) is not None and \
                        re.match(r'[^\d\s-]+-\d+', tokk).span() == (0, len(tokk))

            if tokk.startswith('x') and '_' in tokk:
                wp_toks = tokk.replace('_', ' _ ').split()

            elif (is_rel or is_frame or is_of):
                if is_in_enc:
                    wp_toks = [tokk]
                elif is_frame:
                    if not tokk[-3:].startswith('-'):
                        wp_toks = self._tok_wp(tokk)
                    else:
                        wp_toks = self._tok_wp(tokk[:-3]) + [self.WP + tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if rel in self.encoder:
                        wp_toks = [rel, self.WP + '-of']
                    else:
                        wp_toks = [':'] + [rel[1:]] + [self.WP + '-of']
                elif is_rel:
                    wp_toks = [tokk]
                else:
                    print("tok:", tokk)
                    print(
                        f"is_rel:{is_rel}, is_frame:{is_frame}, is_of:{is_of}")
                    raise ValueError
            else:
                if is_in_enc:
                    wp_toks = [tokk]
                else:
                    wp_toks = self._tok_wp(tokk)

            wp_tokens.append(wp_toks)
        wp_tokens = [b for bb in wp_tokens for b in bb]
        
        return wp_tokens

    def decode_amr(self, tokens):
        def multi_slash_check(line):
            cnt = 0
            for c in line:
                if c == '/':
                    cnt += 1
            return cnt > 1

        def tokens_linearized(tokens):
            toks = []
            for t in tokens:
                if len(toks) and t.startswith('##'):
                    toks[-1] += t[2:]
                elif len(toks) and toks[-1].startswith(':') and t.startswith("x"):
                    toks.append(f" {t}")
                else:
                    toks.append(t)
            return ''.join(toks)

        raw = str(tokens)
        pieces = tokens_linearized(tokens)
        raw += f"\n{pieces}"

        # match the whole ()
        start, end = 0, len(pieces)-1
        while start < len(pieces):
            if pieces[start] != '(':
                start += 1
            else:
                break
        while end > -1:
            if pieces[end] != ')':
                end -= 1
            else:
                break
        pieces = pieces[start:end+1]

        # deal with the problem that parse multi '/xx/xx/xx/' in concept
        for wp in re.findall(r'[^:\(\)]+', pieces):
            if multi_slash_check(wp):
                ps = wp.split('/')
                wp_n = '/'.join(ps[1:])
                concept = '/'.join(ps[1:]).strip()
                if not concept.startswith('\"'):
                    concept = '\"' + concept
                if not concept.endswith('\"'):
                    concept = concept + '\"'
                wp_n = ps[0] + f"/{concept}"
                pieces = pieces.replace(wp, wp_n)

        # add the ) if necessary
        n_lp = re.findall(r'\(', pieces)
        n_rp = re.findall(r'\)', pieces)
        if len(n_lp) > len(n_rp):
            pieces += ')'*(len(n_lp)-len(n_rp))

        try:
            # 尝试在缺失边的情况下添加边
            pieces_ = ''
            edge_exist = True
            for c in pieces:
                if c == '(':
                    if edge_exist == True:
                        edge_exist = False
                        pieces_ += c
                    else:
                        pieces_ += ':arg0('
                elif c == ':':
                    edge_exist = True
                    pieces_ += c
                else:
                    pieces_ += c
            pieces = pieces_
            
            g = penman.decode(pieces)
            g = self._fix_graph(g)
            if len(g.triples) == 0:
                g = DEFAULT
        except Exception:
            g = DEFAULT

        return raw, g

    def _fix_graph(self, graph):
        triples = []
        node_dict = {}
        raign_node = []
        error_coref = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if rel == ':instance':
                # 记录概念节点
                if x not in node_dict.keys() and x not in error_coref:
                    y = 'thing' if y is None else y
                    node_dict[x] = y
            elif rel == ':ralign':
                # 记录虚词
                # remove x in node_dict
                raign_node.append(y)
            elif rel == ':coref' and y not in node_dict.keys():
                # 记录前文不存在的同指
                error_coref.append(y)
            elif x in error_coref:
                # 同指所关联的节点也标记错误
                error_coref.append(y)
        
        concept_set = []
        for triple in graph.triples:
            x, rel, y = triple
            if x is None or re.match(r'x\d+(_x*\d+)*$', x) is None or x not in node_dict.keys():
                pass
            elif rel == ':instance':
                if x not in concept_set:
                    # 不能重复实例化相同节点
                    concept_set.append(x)
                    triples.append(penman.Triple(x, rel, node_dict[x]))
                elif y is not None:
                    var = f'x{newvars}'
                    newvars += 1
                    triples.append(penman.Triple(var, ':instance', y))       
            elif x == y or y is None or \
                    re.match(r'x\d+(_x*\d+)*$', y) is None or \
                    y not in node_dict.keys():
                # y 不符合规范
                if rel != ':coref':
                    # 非同指情况 new y
                    var = f'x{newvars}'
                    newvars += 1
                    triples.append(penman.Triple(x, rel, var))
                    triples.append(penman.Triple(var, ':instance', 'thing'))
            elif rel == ':coref':
                if y not in raign_node and y in node_dict.keys():
                    triples.append(triple)
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
    
        try:
            penman.encode(graph)
        except Exception:
            # if graph is not connected, use 'and' and 'op' to make it connected.
            nxgraph = nx.MultiGraph()
            variables = graph.variables()
            for v1, _, v2 in graph.triples:
                if v1 in variables and v2 in variables:
                    nxgraph.add_edge(v1, v2)
                elif v1 in variables:
                    nxgraph.add_edge(v1, v1)

            triples = graph.triples.copy()
            new_triples = []
            addition = f"x{len(variables) + 100}"
            triples.append(penman.Triple(addition, ":instance", "and"))
            for i, conn_set in enumerate(nx.connected_components(nxgraph), start=1):
                edge = f":op{i}"
                # for 'x14_x15', the key of it is 14.
                conn_set = sorted(conn_set, key=lambda x: int(
                    x[1:]) if '_' not in x else int(x.split('_')[0][1:]))
                conn_set = [c for c in conn_set if c in variables]
                node = conn_set[0]
                new_triples.append(penman.Triple(addition, edge, node))
            triples = new_triples + triples
            metadata = graph.metadata
            graph = penman.Graph(triples)
            graph.metadata.update(metadata)

        return graph