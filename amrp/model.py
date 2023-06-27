# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.model import Model
from supar.modules.dropout import TokenDropout
from supar.modules.transformer import (TransformerDecoder,
                                       TransformerDecoderLayer)
from supar.config import Config
from supar.utils.common import MIN
from supar.modules import VariationalLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bart.modeling_bart import BartDecoder, BartConfig
from supar.modules.pretrained import ScalarMix


from .criterion import CrossEntropyLoss


class Seq2SeqModel(Model):
    r"""
    The implementation of Semantic Role Labeling Parser using span-constrained CRF.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of y states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the y states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        freeze (bool):
            If ``True``, freezes BERT parameters, required if using BERT features. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of LSTM y states. Default: 600.
        n_encoder_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_encoder_hidden=512,
                 n_encoder_layers=3,
                 encoder_dropout=.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        if self.args.encoder == 'transformer':
            self.token_dropout = TokenDropout(self.args.token_dropout)
            self.decoder = TransformerDecoder(layer=TransformerDecoderLayer(n_heads=self.args.n_decoder_heads,
                                                                            n_model=self.args.n_decoder_hidden,
                                                                            n_inner=self.args.n_decoder_inner,
                                                                            dropout=self.args.decoder_dropout),
                                              n_layers=self.args.n_decoder_layers)
        elif self.args.encoder == 'bart':
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.args.bart, dropout=self.args.dropout)
            if self.args.n_words > self.model.config.vocab_size:
                self.model.resize_token_embeddings(self.args.n_words)
            self.encoder, self.decoder = self.model.encoder, self.model.decoder
        elif self.args.encoder == 'bert':
            if self.args.n_words > self.encoder.model.config.vocab_size:
                self.encoder.model.resize_token_embeddings(self.args.n_words)

            self.embed_shared = self.encoder.model.embeddings.word_embeddings

            # self.decoder = TransformerDecoder(layer=TransformerDecoderLayer(n_heads=self.args.n_decoder_heads,
            #                                                                 n_model=self.args.n_decoder_hidden,
            #                                                                 n_inner=self.args.n_decoder_inner,
            #                                                                 dropout=self.args.decoder_dropout),
            #                                   n_layers=self.args.n_decoder_layers)
            self.args.update({
                'decoder_start_token_id': self.args.eos_index,
                'forced_eos_token_id': self.args.eos_index,
                'eos_token_id': self.args.eos_index,
                'pad_token_id': self.args.pad_index,
                'bos_token_id': self.args.bos_index,
                'vocab_size': self.args.n_words
            })
            self.decoder = BartDecoder(
                BartConfig(**self.args), self.embed_shared)

        self.decoder_dropout = nn.Dropout(self.args.decoder_dropout)
        self.classifier = nn.Linear(
            self.args.n_encoder_hidden, self.args.n_words)

        if self.args.use_syn:
            self.pos_embedding = nn.Embedding(
                num_embeddings=self.args.n_tags, embedding_dim=self.args.syn_embedding_size)
            self.rel_embedding = nn.Embedding(
                num_embeddings=self.args.n_rels, embedding_dim=self.args.syn_embedding_size)
            self.arc_embedding = nn.Embedding(
                num_embeddings=self.args.max_len, embedding_dim=self.args.syn_embedding_size)
            self.fusion = VariationalLSTM(input_size=self.args.syn_embedding_size*3,
                                          hidden_size=self.args.syn_encoder_hidden//2,
                                          num_layers=self.args.syn_encoder_layers,
                                          bidirectional=True,
                                          dropout=self.args.syn_encoder_dropout)
            self.scalar_mix = ScalarMix(n_layers=2)

        if self.args.encoder == 'transformer':
            self.classifier.weight = self.word_embed.embed.weight
        elif self.args.encoder == 'bart':
            self.classifier.weight = self.model.shared.weight
        elif self.args.encoder == 'bert':
            # self.decoder.embed_tokens.weight
            self.classifier.weight = self.embed_shared.weight

        self.criterion = CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing)

    def embed(self, words):
        if self.args.encoder == 'transformer':
            return super().embed(words)

        return self.embed_shared(words)

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.

        Returns:
            ~torch.Tensor:
                Representations for the src sentences of the shape ``[batch_size, seq_len, n_model]``.
        """
        # we need to do token dropout, so the TranformerWordEmbedding layer is not invoked here
        if self.args.encoder == 'transformer':
            embed = self.token_dropout(self.word_embed.embed(words))
            embed = embed * self.word_embed.embed_scale + \
                self.word_embed.pos_embed(embed)
            embed = self.embed_dropout(embed)
            x = self.encoder(embed, words.ne(self.args.pad_index))
        elif self.args.encoder == 'bert':
            x = self.encoder(words).float()
        elif self.args.encoder == 'bart':
            x = self.encoder(input_ids=words, attention_mask=words.ne(
                self.args.pad_index))[0]

        if self.args.use_syn and feats is not None:
            pos_embed = self.pos_embedding(feats['pos'])
            arc_embed = self.arc_embedding(feats['arc'])
            rel_embed = self.rel_embedding(feats['rel'])
            syn_x = pack_padded_sequence(torch.cat(
                (pos_embed, arc_embed, rel_embed), -1), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            syn_x, _ = self.fusion(syn_x)
            syn_x, _ = pad_packed_sequence(
                syn_x, True, total_length=words.shape[1])

            return self.scalar_mix([x, syn_x])

        return x

    def loss(self, x, tgt, src_mask, tgt_mask):
        if self.args.encoder == 'bart' or self.args.encoder == 'bert':
            shifted = torch.full_like(tgt, self.args.eos_index)
            shifted[:, 1:] = tgt[:, :-1]
            y = self.decoder(input_ids=shifted,
                             attention_mask=tgt_mask,
                             encoder_hidden_states=x,
                             encoder_attention_mask=src_mask)[0]
            tgt_mask[:, 0] = 0
        else:
            tgt_mask = tgt_mask[:, 1:]
            shifted, tgt,  = tgt[:, :-1], tgt[:, 1:]
            _, seq_len = tgt.shape
            attn_mask = tgt.new_ones(
                seq_len, seq_len, dtype=torch.bool).tril_()
            y = self.decoder(self.embed(shifted), x,
                             tgt_mask, src_mask, attn_mask)
        y = self.decoder_dropout(y)
        s_y = self.classifier(y)
        return self.criterion(s_y[tgt_mask], tgt[tgt_mask])

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                               for past_state in layer_past),)
        return reordered_past

    def decode(self, x, src_mask):
        batch_size, *_ = x.shape
        beam_size, n_words = self.args.beam_size, self.args.n_words

        # repeat the src inputs beam_size times
        # [batch_size * beam_size, ...]
        x = x.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *x.shape[1:])
        src_mask = src_mask.unsqueeze(1).repeat(
            1, beam_size, 1).view(-1, *src_mask.shape[1:])
        # initialize the tgt inputs by <bos>
        # [batch_size * beam_size, seq_len]
        tgt = x.new_full((batch_size * beam_size, 1),
                         self.args.bos_index, dtype=torch.long)
        # [batch_size * beam_size]
        active = src_mask.new_ones(batch_size * beam_size)
        # [batch_size]
        batches = tgt.new_tensor(range(batch_size)) * beam_size
        # accumulated scores
        scores = x.new_full((batch_size, self.args.beam_size),
                            MIN).index_fill_(-1, tgt.new_tensor(0), 0).view(-1)

        def rank(scores, mask, k):
            scores = scores / \
                mask.sum(-1).unsqueeze(-1) ** self.args.length_penalty
            return scores.view(batch_size, -1).topk(k, -1)[1]

        if self.args.encoder != 'transformer':
            past_key_values = self.decoder(
                input_ids=torch.full_like(tgt[:, :1], self.args.eos_index),
                attention_mask=torch.ones_like(src_mask[:, :1]),
                encoder_hidden_states=x,
                encoder_attention_mask=src_mask,
                past_key_values=None,
                use_cache=True,
            )[1]

        # 解码结果一般不超过src的4.5倍
        for t in range(1, min(self.args.max_len, int(4.8 * x.shape[1]))):
            tgt_mask = tgt.ne(self.args.pad_index)
            if self.args.encoder == 'transformer':
                attn_mask = tgt_mask.new_ones(t, t).tril_()
                s_y = self.decoder(self.embed(
                    tgt[active]), x[active], tgt_mask[active], src_mask[active], attn_mask)
                # [n_active, n_words]
                s_y = self.classifier(s_y[:, -1]).log_softmax(-1)
                # only allow finished sequences to get <pad>
                # [batch_size * beam_size, n_words]
                s_y = x.new_full((batch_size * beam_size, n_words),
                                 MIN).masked_scatter_(active.unsqueeze(-1), s_y)
            else:
                input_ids = tgt[:, -1:]
                s_y, new_past_key_values = self.decoder(
                    input_ids=input_ids,
                    attention_mask=torch.cat(
                        (torch.ones_like(tgt_mask[:, :1]), tgt_mask), 1),
                    encoder_hidden_states=x,
                    encoder_attention_mask=src_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )[:2]
                del past_key_values
                past_key_values = new_past_key_values
                # [n_active, n_words]
                s_y = self.classifier(s_y[:, -1]).log_softmax(-1)
                # only allow finished sequences to get <pad>
                s_y[~active] = MIN

            s_y[~active, self.args.pad_index] = 0

            # [batch_size * beam_size, n_words]
            scores = scores.unsqueeze(-1) + s_y
            # [batch_size, beam_size]
            cands = rank(scores, tgt_mask, beam_size)
            # [batch_size * beam_size]
            scores = scores.view(batch_size, -1).gather(-1, cands).view(-1)
            # beams, tokens = cands // n_words, cands % n_words
            beams, tokens = cands.div(
                n_words, rounding_mode='floor'), (cands % n_words).view(-1, 1)
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            # [batch_size * beam_size, seq_len + 1]
            tgt = torch.cat((tgt[indices], tokens), 1)
            if self.args.encoder != 'transformer':
                past_key_values = self._reorder_cache(past_key_values, indices)
            active = tokens.ne(tokens.new_tensor(
                (self.args.eos_index, self.args.pad_index))).all(-1)

            if not active.any():
                break
        cands = rank(scores.view(-1, 1),
                     tgt.ne(self.args.pad_index), self.args.topk)
        return tgt[(batches.unsqueeze(-1) + cands).view(-1)].view(batch_size, self.args.topk, -1)
