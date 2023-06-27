import torch
from typing import Iterable, Union, Callable, Iterable, Optional, Union, List
from supar.utils.field import Field


class TruncatedField(Field):
    # 将超过max_len长度的数据进行截断
    def __init__(self, 
                 *args, 
                 max_len:int = 1024, 
                 decode: Optional[Callable] = None,
                 manual_bos_idx: int = None, 
                 manual_eos_idx: int = None, 
                 fix_pad_conflict: bool = False, 
                 **kargs) -> Field:
        super().__init__(*args, **kargs)
        self.max_len = max_len
        self.decode = decode
        self.manual_bos_idx = manual_bos_idx
        self.manual_eos_idx = manual_eos_idx
        self.fix_pad_conflict = fix_pad_conflict
    
    def transform(self, sequences: Iterable[List[str]]) -> Iterable[torch.Tensor]:
        r"""
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (Iterable[List[str]]):
                A list of sequences.

        Returns:
            A list of tensors transformed from the input sequences.
        """
        for seq in sequences:
            seq = self.preprocess(seq)
            if self.use_vocab:
                seq = [self.vocab[token] for token in seq]
            if self.fix_pad_conflict:
                if self.pad_index in seq:
                    seq = [i+1 for i in seq]
            if self.bos:
                if self.manual_bos_idx is not None:
                    seq = [self.manual_bos_idx] + seq
                else:
                    seq = [self.bos_index] + seq
            if self.eos:
                if self.manual_eos_idx is not None:
                    seq = seq + [self.manual_eos_idx]
                else:
                    seq = seq + [self.eos_index]
            yield torch.tensor(seq, dtype=torch.long)
    
    def preprocess(self, data: Union[str, Iterable]) -> Iterable:
        t_len = self.max_len
        data = super().preprocess(data)
        if self.bos:
            t_len -= 1
        if self.eos:
            t_len -= 1
        return data[:t_len]
    
    def postprocess(self, data: Iterable) -> Iterable:
        if self.decode is not None:
            data = self.decode(data)
        if self.bos is not None and data[0] == self.bos:
            data = data[1:]
        if self.eos is not None and data[-1] == self.eos:
            data = data[:-1]
        return data
    
    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if hasattr(self, 'vocab'):
            params.append(f"vocab_size={len(self.vocab)}")
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        params.append(f"max_len={self.max_len}")
        return s + ', '.join(params) + ')'
        