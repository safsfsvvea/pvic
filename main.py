"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from pvic import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory, DataFactory_CLIP
from configs import base_detector_args, advanced_detector_args
import ModifiedCLIP as clip
import time

warnings.filterwarnings("ignore")

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    if args.CLIP:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        args.clip_model, args.clip_preprocess = clip.load(args.CLIP_path, device=device)
        for param in args.clip_model.parameters():
            param.requires_grad = False
        args.clip_model.eval()
        trainset = DataFactory_CLIP(
            name=args.dataset, partition=args.partitions[0],
            data_root=args.data_root, args=args
        )
        testset = DataFactory_CLIP(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root, args=args
        )
    else:
        if args.CLIP_query: 
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            args.clip_model, args.clip_preprocess = clip.load(args.CLIP_path, device=device)
            for param in args.clip_model.parameters():
                param.requires_grad = False
            args.clip_model.eval()
        trainset = DataFactory(
            name=args.dataset, partition=args.partitions[0],
            data_root=args.data_root
        )
        testset = DataFactory(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root
        )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            testset, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_verbs = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_verbs = 24
    
    model = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: PViC loaded from saved checkpoint {args.resume}.")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # modified for load of clip
    else:
        print(f"=> Rank {rank}: PViC randomly initialised.")

    engine = CustomisedDLE(model, train_loader, test_loader, args)

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            """
            NOTE This evaluation results on V-COCO do not necessarily follow the 
            protocol as the official evaluation code, and so are only used for
            diagnostic purposes.
            """
            ap = engine.test_vcoco()
            if rank == 0:
                print(f"The mAP is {ap.mean():.4f}.")
            return
        else:
            ap, max_recall = engine.test_hico()
            if rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = trainset.dataset.rare
                non_rare = trainset.dataset.non_rare
                print(
                    f"The mAP is {ap.mean():.4f},"
                    f" rare: {ap[rare].mean():.4f},"
                    f" none-rare: {ap[non_rare].mean():.4f},"
                    f" mean max recall: {max_recall.mean():.4f}"
                )
            return

    model.freeze_detector()
    param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    engine(args.epochs)

@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.num_verbs = 117
    args.num_triplets = 600
    object_to_target = dataset.dataset.object_to_verb
    model = build_detector(args, object_to_target)
    if args.eval:
        model.eval()
    if os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        print(f"Loading checkpoints from {args.resume}.")
        model.load_state_dict(ckpt['model_state_dict'], strict=False) # modified for load of clip

    image, target = dataset[998]
    outputs = model([image], targets=[target])

if __name__ == '__main__':

    if "DETR" not in os.environ:
        raise KeyError(f"Specify the detector type with env. variable \"DETR\".")
    elif os.environ["DETR"] == "base":
        parser = argparse.ArgumentParser(parents=[base_detector_args(),])
        parser.add_argument('--detector', default='base', type=str)
        parser.add_argument('--raw-lambda', default=2.8, type=float)
    elif os.environ["DETR"] == "advanced":
        parser = argparse.ArgumentParser(parents=[advanced_detector_args(),])
        parser.add_argument('--detector', default='advanced', type=str)
        parser.add_argument('--raw-lambda', default=1.7, type=float)

    parser.add_argument('--CLIP', action='store_true', help='use CLIP feature')
    parser.add_argument('--CLIP_text', action='store_true', help='use CLIP text feature')
    parser.add_argument('--CLIP_encoder', action='store_true', help='use CLIP feature in encoder stage')
    parser.add_argument('--CLIP_decoder', action='store_true', help='use CLIP feature in decoder stage')
    parser.add_argument('--clip4hoi_decoder', action='store_true', help='use clip4hoi decoder')
    parser.add_argument('--CLIP_path', default='checkpoints/clip/ViT-B-32.pt', type=str)
    parser.add_argument('--CLIP_query', action='store_true', help='use CLIP bbox feature and fusion with detr queries')
    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--use-wandb', default=True, action='store_true')

    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=8, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--extract_feature', action='store_true', help='extract object feature')
    parser.add_argument('--object_feature_replace_prob', default=0, type=float, help='probability of replacing object query')
    parser.add_argument('--object_feature_replace_thresh', default=.9, type=float, help='score threshold of replacing object query')
    parser.add_argument('--same_object_verb', action='store_true', help='replace object query when object and verb category are exactly the same')
    parser.add_argument('--max-object-features', default=10, type=int)
    parser.add_argument('--object_feature_dir', default='/bd_byt4090i1/users/clin/pvic/object_features')
    parser.add_argument('--test_out_dir', default='')
    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    # os.environ["MASTER_PORT"] = "12345"
    start_time = time.time()
    mp.spawn(main, nprocs=args.world_size, args=(args,))
    end_time = time.time()
    print(f"Total main time: {end_time - start_time:.2f} seconds.")
