import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 定义文件路径和目标类
file_paths = [
    '/bd_byt4090i1/users/clin/pvic/outputs/clip_query_mbf_replace_prob05_thre05/test/hoi_class_results.json',
    '/bd_byt4090i1/users/clin/pvic/outputs/clip_query_mbf_noreplace/test/hoi_class_results.json',
    '/bd_byt4090i1/users/clin/pvic/outputs/clip_query_mbf_replace/test/hoi_class_results.json'
]
model_names = ['replace_prob05_thre05', 'noreplace', 'replace']  # 模型对应的标签
target_classes = [152, 241, 577, 428, 558]  # 目标类索引

# 存储结果的字典
results = {model: {'AP': [], 'Max Recall': []} for model in model_names}

# 提取数据
for file_path, model_name in zip(file_paths, model_names):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for target_class in target_classes:
        # 获取目标类的 AP 和 Max Recall
        index = data['Class'].index(f'Class {target_class}')
        ap = data['AP'][index]
        max_recall = data['Max Recall'][index]
        
        results[model_name]['AP'].append(ap)
        results[model_name]['Max Recall'].append(max_recall)

# 转换为 DataFrame
results_df = pd.DataFrame({
    'Model': [model for model in model_names for _ in target_classes],
    'Class': target_classes * len(model_names),
    'AP': [ap for model in model_names for ap in results[model]['AP']],
    'Max Recall': [rec for model in model_names for rec in results[model]['Max Recall']],
})

# 保存提取结果
output_csv = 'hoi_class_comparison.csv'
results_df.to_csv(output_csv, index=False)
print(f"提取结果已保存到 {output_csv}")

# 绘制比较图（散点图）
plt.figure(figsize=(12, 6))

# 绘制 AP 图（散点图）
plt.subplot(1, 2, 1)
for model in model_names:
    plt.scatter(
        target_classes,
        results[model]['AP'],
        label=f'{model} (AP)',
        s=100,  # 设置点的大小
        marker='o'  # 设置点的形状
    )
plt.title('AP Comparison')
plt.xlabel('Class Index')
plt.ylabel('AP')
plt.legend()

# 绘制 Max Recall 图（散点图）
plt.subplot(1, 2, 2)
for model in model_names:
    plt.scatter(
        target_classes,
        results[model]['Max Recall'],
        label=f'{model} (Max Recall)',
        s=100,  # 设置点的大小
        marker='s'  # 设置点的形状
    )
plt.title('Max Recall Comparison')
plt.xlabel('Class Index')
plt.ylabel('Max Recall')
plt.legend()

# 保存和显示图像
output_img = 'hoi_class_comparison_scatter.png'
plt.tight_layout()
plt.savefig(output_img)
print(f"比较图已保存到 {output_img}")
plt.show()
