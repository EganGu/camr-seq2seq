# -*- coding: utf-8 -*-

import os

import torch
from torch.optim import AdamW, Optimizer
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from typing import Iterable, Union
from datetime import datetime, timedelta

from supar.parser import Parser
from supar.config import Config
from supar.utils.common import BOS, EOS, PAD, UNK
from supar.utils.logging import get_logger, logging
from supar.utils.tokenizer import BPETokenizer
from supar.utils.transform import Batch
from supar.utils import Dataset
from supar.utils.parallel import gather, is_dist, is_master
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.fn import set_rng_state
from supar.utils.metric import Metric

from .metric import SmatchMetric
from .model import Seq2SeqModel
from .transform import AMR
from .tokenization_bert import AMRBertTokenizer
from .field import TruncatedField

logger = get_logger(__name__)
# 静默penman的log
logging.getLogger("penman").setLevel(logging.CRITICAL)
logging.getLogger("penman.layout").setLevel(logging.CRITICAL)


class Seq2SeqParser(Parser):

    NAME = 'seq2seq'
    MODEL = Seq2SeqModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SRC = self.transform.SRC
        self.TGT = self.transform.TGT

    def init_optimizer(self) -> Optimizer:
        return AdamW(params=[{'params': p, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
                             for n, p in self.model.named_parameters()],
                     lr=self.args.lr,
                     betas=(self.args.get('mu', 0.9),
                            self.args.get('nu', 0.999)),
                     eps=self.args.get('eps', 1e-8),
                     weight_decay=self.args.get('weight_decay', 0))

    def train_step(self, batch: Batch) -> torch.Tensor:
        feats = None
        if self.args.use_syn:
            src, tgt, tags, arcs, rels = batch
            feats = {'pos': tags, 'arc': arcs, 'rel': rels}
        else:
            src, tgt = batch

        x = self.model(src, feats=feats)
        src_mask, tgt_mask = batch.mask, tgt.ne(self.args.pad_index)
        loss = self.model.loss(x, tgt, src_mask, tgt_mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SmatchMetric:
        feats = None
        if self.args.use_syn:
            src, tgt, tags, arcs, rels = batch
            feats = {'pos': tags, 'arc': arcs, 'rel': rels}
        else:
            src, tgt = batch
        x = self.model(src, feats=feats)
        src_mask, tgt_mask = batch.mask, tgt.ne(self.args.pad_index)
        loss = self.model.loss(x, tgt, src_mask, tgt_mask)
        preds = golds = states = None
        if self.args.eval_tgt:
            golds = [s.values[1] for s in batch.sentences]
            preds = self.model.decode(x, batch.mask)[:, 0]
            pred_mask = preds.ne(self.args.pad_index)
            preds = [i.tolist()
                     for i in preds[pred_mask].split(pred_mask.sum(-1).tolist())]
            preds_wstate = [self.transform.decode_graph(i) for i in preds]
            preds = [x[0] for x in preds_wstate]
            states = [x[1] for x in preds_wstate]

        return SmatchMetric(loss, preds, golds, tgt_mask, states)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        feats = None
        if self.args.use_syn:
            src, tags, arcs, rels = batch
            feats = {'pos': tags, 'arc': arcs, 'rel': rels}
        else:
            src, = batch

        x = self.model(src, feats=feats)
        tgt = self.model.decode(x, batch.mask)

        # preds_wstate = [[self.transform.decode_graph(cand) for cand in i] for i in tgt.tolist()]
        batch.tgt = [[self.transform.decode_graph(
            cand)[0] for cand in i] for i in tgt.tolist()]
        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            return cls.load(**args)

        logger.info("Building the fields")
        if args.encoder == 'transformer':
            t = AMRBertTokenizer.from_pretrained(args.bert)
            # t = BPETokenizer(path=os.path.join(os.path.dirname(path)),
            #                  files=args.vocab,
            #                  backend='huggingface',
            #                  pad=PAD,
            #                  unk=UNK,
            #                  bos=BOS,
            #                  eos=EOS)
        # 中文用bert分词器
        elif args.encoder == 'bart':
            t = AMRBertTokenizer.from_pretrained(args.bart)
        elif args.encoder == 'bert':
            t = AMRBertTokenizer.from_pretrained(args.bert)

        SRC = TruncatedField('src', max_len=args.get('src_max_len', args.max_len), pad=t.pad_token, unk=t.unk_token,
                             bos=t.bos_token, eos=t.eos_token, tokenize=t.tokenize)
        TGT = TruncatedField('tgt', max_len=args.max_len, pad=t.pad_token, unk=t.unk_token, bos=t.bos_token,
                             eos=t.eos_token, tokenize=t.tokenize_amr, decode=t.get_decoder())
        if args.use_syn:
            TAG = TruncatedField('tags', max_len=args.max_len,
                                 pad=t.pad_token, bos=t.bos_token, eos=t.eos_token)
            REL = TruncatedField('rels', max_len=args.max_len,
                                 pad=t.pad_token, bos=t.bos_token, eos=t.eos_token)
            ARC = TruncatedField('arcs', max_len=args.max_len, use_vocab=False, pad=t.pad_token, bos=t.bos_token, eos=t.eos_token,
                                 manual_bos_idx=args.max_len-2, manual_eos_idx=args.max_len-1, fix_pad_conflict=True)
            transform = AMR(SRC=SRC, TGT=TGT, POS=TAG, HEAD=ARC, DEPREL=REL)
        else:
            transform = AMR(SRC=SRC, TGT=TGT)

        # share the vocab
        SRC.vocab = TGT.vocab = t.get_vocab()
        args.update({'n_words': len(SRC.vocab),
                     'pad_index': SRC.pad_index,
                     'unk_index': SRC.unk_index,
                     'bos_index': SRC.bos_index,
                     'eos_index': SRC.eos_index})
        if args.use_syn:
            train = Dataset(transform, args.train, **args)
            TAG.build(train)
            REL.build(train)
            args.update({'n_tags': len(TAG.vocab),
                         'n_rels': len(REL.vocab)})
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int,
        patience: int,
        batch_size: int = 5000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        clip: float = 5.0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> None:
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            epochs (int):
                The number of training iterations.
            patience (int):
                The number of consecutive iterations after which the training process would be early stopped if no improvement.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            clip (float):
                Clips gradient of an iterable of parameters at specified value. Default: 5.0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        eval_batch_size = args.get('eval_batch_size', batch_size)
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
            eval_batch_size = eval_batch_size // dist.get_world_size()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        args.even = args.get('even', is_dist())
        train = Dataset(self.transform, args.train, **args).build(
            batch_size=batch_size,
            n_buckets=buckets,
            shuffle=True,
            distributed=is_dist(),
            even=args.even,
            seed=args.seed,
            n_workers=workers
        )
        dev = Dataset(self.transform, args.dev, **args).build(
            batch_size=eval_batch_size,
            n_buckets=buckets,
            shuffle=False,
            distributed=is_dist(),
            even=False,
            n_workers=workers
        )
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test, **args).build(
                batch_size=eval_batch_size,
                n_buckets=buckets,
                shuffle=False,
                distributed=is_dist(),
                even=False,
                n_workers=workers
            )
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")
        loader, sampler = train.loader, train.loader.batch_sampler
        args.steps = len(loader) * epochs // args.update_steps
        args.save(f"{args.path}.yaml")

        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(module=self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get(
                                 'find_unused_parameters', True),
                             static_graph=args.get('static_graph', False))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(
                    dist.group.WORLD, fp16_compress_hook)
        if args.wandb and is_master():
            import wandb
            # start a new wandb run to track this script
            wandb.init(config=args.primitive_config,
                       project=args.get('project', self.NAME),
                       name=args.get('name', args.path),
                       resume=self.args.checkpoint)
        self.step, self.epoch, self.best_e, self.patience = 1, 1, 1, patience
        # uneven batches are excluded
        self.n_batches = min(gather(len(loader))) if is_dist() else len(loader)
        self.best_metric, self.elapsed = Metric(), timedelta()
        if args.checkpoint:
            try:
                self.optimizer.load_state_dict(
                    self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(
                    self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(
                    self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                sampler.set_epoch(self.epoch)
            except AttributeError:
                logger.warning(
                    "No checkpoint found. Try re-launching the training procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            with self.join():
                # we should reset `step` as the number of batches in different processes is not necessarily equal
                self.step = 1
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device, enabled=args.amp):
                            loss = self.train_step(batch)
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(
                            self.model.parameters(), args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(
                        f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
                    # log metrics to wandb
                    if args.wandb and is_master():
                        wandb.log(
                            {'lr': self.scheduler.get_last_lr()[0], 'loss': loss})
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            if epoch >= self.args.start_eval:
                with self.join(), torch.autocast(self.device, enabled=args.amp):
                    metric = self.reduce(
                        sum([self.eval_step(i) for i in progress_bar(dev.loader)], Metric()))
                    logger.info(f"{'dev:':5} {metric}")
                    if args.wandb and is_master():
                        wandb.log({'dev': metric.values, 'epochs': epoch})
                    if args.test:
                        test_metric = self.reduce(
                            sum([self.eval_step(i) for i in progress_bar(test.loader)], Metric()))
                        logger.info(f"{'test:':5} {test_metric}")
                        if args.wandb and is_master():
                            wandb.log(
                                {'test': test_metric.values, 'epochs': epoch})

            t = datetime.now() - start
            self.epoch += 1
            if epoch > self.args.start_eval:
                self.patience -= 1
            self.elapsed += t

            if epoch > self.args.start_eval and metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if is_dist():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            best.model.eval()
            with best.join():
                test_metric = sum([best.eval_step(i)
                                  for i in progress_bar(test.loader)], Metric())
                logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")
        if args.wandb and is_master():
            wandb.finish()
