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
import ModifiedCLIP as clip
warnings.filterwarnings("ignore")
from PIL import ImageDraw
import time

def extract_and_save_features_test(args, outputs, output_dir, json_output, images_PIL, global_counter):
    """
    从 outputs 提取特征并保存到指定的路径，并将被选择的 object 及其 bbox 可视化到对应的图像上。
    使用全局计数器避免文件名冲突。
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for batch_idx, (props, image_PIL) in enumerate(zip(outputs, images_PIL)):
        labels = props["labels"]
        scores = props["scores"]
        bboxes = props["boxes"]  # 假设 bbox 是 (N, 4)，格式为 [x1, y1, x2, y2]
        hidden_states = props["hidden_states"]

        # 创建可编辑的 PIL 图片副本
        draw = ImageDraw.Draw(image_PIL)
        has_valid_object = False

        for i, (label, score, bbox) in enumerate(zip(labels, scores, bboxes)):
            label = label.item()
            score = score.item()
            feature = hidden_states[i].cpu().numpy()
            bbox = bbox.cpu().numpy()

            if score > args.object_score_thresh and label in {1, 2, 3, 4, 5}:
                
                # 如果当前类别尚未记录到 JSON，进行初始化
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

                # 如果尚未达到最大特征数，存储特征
                # if json_output[label]["num_features"] < args.max_object_features:
                has_valid_object = True
                all_features = np.vstack([existing_features, feature[np.newaxis, :]])
                np.save(feature_file, all_features)
                json_output[label]["num_features"] += 1

                # 可视化：绘制当前 bbox 和类别
                draw.rectangle(
                    [bbox[0], bbox[1], bbox[2], bbox[3]], 
                    outline="red", width=3
                )
                draw.text(
                    (bbox[0], bbox[1] - 10), 
                    f"Label: {label}, Score: {score:.2f}", 
                    fill="red"
                )
        if has_valid_object:
            # 保存标注后的图像，文件名包含全局计数器和时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_save_path = os.path.join(vis_dir, f"image_{global_counter:06d}_{timestamp}.jpg")
            image_PIL.save(image_save_path)

        # 更新全局计数器
        global_counter += 1

    return global_counter

def extract_and_save_features_verb(args, region_props, paired_inds, labels, output_dir, json_output):
    """
    从 region_props, paired_inds 和 labels 提取特征并保存到指定路径。

    json_output 的结构：
    {
        "object_id": {
            "verb_id": {
                "feature_file": str,
                "num_features": int
            }
        }
    }
    """
    feature_cache = {}  # cache features to reduce IO

    for i, props in enumerate(region_props):
        object_labels = props["labels"].cpu().numpy()
        object_scores = props["scores"].cpu().numpy()
        # if args.CLIP_query:
        #     object_features = props["clip_features"].cpu().numpy()
        # else:
        object_features = props["hidden_states"].cpu().numpy()
         
        pairs = paired_inds[i]  # (M, 2), pairing indices for human-object pairs
        batch_labels = labels[i]  # (M, 117)

        processed_instances = set()
        for pair_idx, (human_idx, object_idx) in enumerate(pairs):
            object_label = int(object_labels[object_idx])
            object_score = float(object_scores[object_idx])

            # Ignore objects below score threshold
            if object_score <= args.object_score_thresh:
                continue

            # Extract the feature for the current object
            object_feature = object_features[object_idx]

            # Get the corresponding verbs for this pair
            verb_indices = torch.nonzero(batch_labels[pair_idx], as_tuple=False).squeeze(1).tolist()
            if not isinstance(verb_indices, list):
                verb_indices = [verb_indices]

            for verb_id in verb_indices:
                key = (object_label, verb_id)
                instance_key = (object_idx, verb_id)
                if object_label not in json_output:
                    json_output[object_label] = {}

                if verb_id not in json_output[object_label]:
                    json_output[object_label][verb_id] = {
                        "feature_file": f"object_{object_label}_verb_{verb_id}_features.npy",
                        "num_features": 0
                    }
                if key not in feature_cache:
                    feature_cache[key] = []

                if instance_key not in processed_instances:
                    if json_output[object_label][verb_id]["num_features"] + len(feature_cache[key]) < args.max_object_features:
                        feature_cache[key].append(object_feature)
                        processed_instances.add(instance_key)

    # Save all features at once
    for (object_label, verb_id), features in feature_cache.items():
        # if object_label not in json_output:
        #     json_output[object_label] = {}

        # if verb_id not in json_output[object_label]:
        #     json_output[object_label][verb_id] = {
        #         "feature_file": f"object_{object_label}_verb_{verb_id}_features.npy",
        #         "num_features": 0
        #     }
        if len(features) > 0:
            feature_file = os.path.join(output_dir, "features", json_output[object_label][verb_id]["feature_file"])

            # Load existing features if the file exists
            if os.path.exists(feature_file):
                existing_features = np.load(feature_file)
            else:
                existing_features = np.empty((0, object_features.shape[-1]))

            # Append new features and save
            all_features = np.vstack([existing_features, np.array(features)])
            np.save(feature_file, all_features)

            # Update the count of features in the JSON
            json_output[object_label][verb_id]["num_features"] = all_features.shape[0]

# def extract_and_save_features(args, outputs, output_dir, json_output):
#     """
#     从 outputs 提取特征并保存到指定的路径。
#     """
#     feature_cache = {}  # cache features to reduce IO

#     for props in outputs:
#         labels = props["labels"].cpu().numpy()
#         scores = props["scores"].cpu().numpy()
#         hidden_states = props["hidden_states"].cpu().numpy()

#         for label, score, feature in zip(labels, scores, hidden_states):
#             label = int(label)
#             score = float(score)

#             # Skip low-score objects
#             if score <= args.object_score_thresh:
#                 continue

#             key = label
#             if key not in feature_cache:
#                 feature_cache[key] = []

#             if len(feature_cache[key]) < args.max_object_features:
#                 feature_cache[key].append(feature)

#     # Save all features at once
#     for label, features in feature_cache.items():
#         if label not in json_output:
#             json_output[label] = {
#                 "class_id": label,
#                 "feature_file": f"{label}_features.npy",
#                 "num_features": 0
#             }

#         feature_file = os.path.join(output_dir, "features", json_output[label]["feature_file"])

#         # Load existing features if the file exists
#         if os.path.exists(feature_file):
#             existing_features = np.load(feature_file)
#         else:
#             existing_features = np.empty((0, hidden_states.shape[-1]))

#         # Append new features and save
#         all_features = np.vstack([existing_features, np.array(features)])
#         np.save(feature_file, all_features)

#         # Update the count of features in the JSON
#         json_output[label]["num_features"] = all_features.shape[0]
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

            if score > args.object_score_thresh:
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
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if args.CLIP_query: 
        args.clip_model, args.clip_preprocess = clip.load(args.CLIP_path, device=device)
        for param in args.clip_model.parameters():
            param.requires_grad = False
        args.clip_model.eval()
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root, args=args
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )
    
    model = build_detector(args, None)
    model = model.to(device)
    model.freeze_detector()
    model.eval()
    output_dir = args.feature_output_dir
    os.makedirs(output_dir, exist_ok=True)
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    json_output = {}
    global_counter = 0
    for images, targets, images_PIL in tqdm(test_loader, desc="Processing images", unit="batch"):
        with torch.no_grad():
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print("targets: ", targets)
            region_props, paired_inds, labels = model(images, targets)
            if args.same_object_verb:
                extract_and_save_features_verb(args, region_props, paired_inds, labels, output_dir, json_output)
            else:
                # extract_and_save_features(args, region_props, output_dir, json_output)
                global_counter = extract_and_save_features_test(
                    args, region_props, output_dir, json_output, images_PIL, global_counter
                )
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
    parser.add_argument('--same_object_verb', action='store_true', help='replace object query when object and verb category are exactly the same')
    parser.add_argument('--feature_output_dir', default='./object_features')
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--object-score-thresh', default=0.9, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--max-object-features', default=100, type=int)
    parser.add_argument('--CLIP_path', default='checkpoints/clip/ViT-B-32.pt', type=str)
    parser.add_argument('--CLIP_query', action='store_true', help='use CLIP bbox feature and fusion with detr queries')

    args = parser.parse_args()
    print(args)
    main(args)