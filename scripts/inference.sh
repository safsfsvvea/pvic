#!/bin/bash
DETR=base python inference.py --resume checkpoints/pvic-detr-r50-hicodet.pth --index 4050 --action 111 --CLIP --CLIP_path /bd_targaryen/users/clin/pvic/checkpoints/clip/ViT-B-32.pt --CLIP_text