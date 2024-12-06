import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from pvic import build_detector
from utils import custom_collate, DataFactory
from configs import base_detector_args, advanced_detector_args
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")
def extract_and_save_features(args, outputs, output_dir, json_output):
    """
    从 outputs 提取特征并保存到指定的路径
    """
    for batch_idx, props in enumerate(outputs):
        labels = props["labels"]
        scores = props["scores"]
        hidden_states = props["hidden_states"]

        for i, (label, score) in enumerate(zip(labels, scores)):
            label = label.item()
            score = score.item()
            feature = hidden_states[i].cpu().numpy()

            if score > args.object_score_thresh and label != 0:
                if label not in json_output:
                    json_output[label] = {
                        "class_id": label,
                        "feature_file": f"{label}_features.npy",
                        "num_features": 0
                    }

                feature_file = os.path.join(output_dir, "features", f"{label}_features.npy")
                if os.path.exists(feature_file):
                    existing_features = np.load(feature_file)
                else:
                    existing_features = np.empty((0, hidden_states.shape[-1]))

                if json_output[label]["num_features"] < args.max_object_features:
                    all_features = np.vstack([existing_features, feature[np.newaxis, :]])
                    np.save(feature_file, all_features)
                    json_output[label]["num_features"] += 1


@torch.no_grad()
def main(args):
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )
    
    model = build_detector(args, None)

    model.freeze_detector()
    model.eval()
    output_dir = args.feature_output_dir
    os.makedirs(output_dir, exist_ok=True)
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    json_output = {}
    for images, targets in tqdm(test_loader, desc="Processing images", unit="batch"):
        with torch.no_grad():
            outputs = model(images)
            extract_and_save_features(args, outputs, output_dir, json_output)
    metadata_file = os.path.join(output_dir, "features_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(json_output, f, indent=4)

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

    parser.add_argument('--extract_feature', action='store_true', help='extract object feature')
    parser.add_argument('--feature_output_dir', default='./object_features')
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--object-score-thresh', default=0.9, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--max-object-features', default=100, type=int)

    args = parser.parse_args()
    print(args)
    main(args)