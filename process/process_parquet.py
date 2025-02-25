import base64
import io
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def convert_json_to_parquet(input_json_path, output_parquet_path, images_dir=None):
    """
    将SP-DocVQA JSON数据转换为Huggingface兼容的Parquet文件
    
    参数:
    input_json_path (str): 输入JSON文件的路径
    output_parquet_path (str): 输出Parquet文件的路径
    images_dir (str, optional): 图像文件夹的根目录路径，如果为None，则假设使用JSON中的完整路径
    
    返回:
    None
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_parquet_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取原始JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备数据列表
    records = []
    
    for item in tqdm(data):
        # 构建图像路径
        image_path = item['image']
        if images_dir:
            image_path = os.path.join(images_dir, image_path)
        
        try:
            # 读取图像并转换为字节
            img = Image.open(image_path)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # 将答案格式化为所需格式
            answers = item['answers']
            if isinstance(answers, list):
                # 使用第一个答案或合并多个答案
                formatted_answer = f"<answer> {answers[0]} </answer>"
            else:
                formatted_answer = f"<answer> {answers} </answer>"
            
            # 创建记录
            record = {
                'image': img_bytes,
                'problem': item['question'],
                'solution': formatted_answer
            }
            
            records.append(record)
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    # 将DataFrame转换为Parquet文件
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_parquet_path)
    
    print(f"转换完成! 已将数据保存到: {output_parquet_path}")
    print(f"共处理 {len(records)} 条记录")

# 另一种实现方式: 使用datasets库直接创建Huggingface数据集
def create_hf_dataset(input_json_path, output_dir, images_dir=None):
    """
    创建Huggingface兼容的数据集
    
    参数:
    input_json_path (str): 输入JSON文件的路径
    output_dir (str): 输出数据集目录
    images_dir (str, optional): 图像文件夹的根目录路径
    
    返回:
    datasets.Dataset: Huggingface数据集对象
    """
    try:
        from datasets import Dataset, Features
        from datasets import Image as HFImage
        from datasets import Value
    except ImportError:
        print("请先安装datasets库: pip install datasets")
        return None
    
    # 读取原始JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备数据
    image_paths = []
    problems = []
    solutions = []
    
    for item in data['data']:
        # 构建图像路径
        image_path = item['image']
        if images_dir:
            image_path = os.path.join(images_dir, image_path)
        
        # 将答案格式化为所需格式
        answers = item['answers']
        if isinstance(answers, list):
            formatted_answer = f"<answer> {answers[0]} </answer>"
        else:
            formatted_answer = f"<answer> {answers} </answer>"
        
        image_paths.append(image_path)
        problems.append(item['question'])
        solutions.append(formatted_answer)
    
    # 创建初始数据集
    dataset_dict = {
        'image_path': image_paths,
        'problem': problems,
        'solution': solutions
    }
    
    # 创建数据集
    dataset = Dataset.from_dict(dataset_dict)
    
    # 加载图像
    features = Features({
        'image': HFImage(),
        'problem': Value('string'),
        'solution': Value('string')
    })
    
    # 将路径转换为图像output
    def load_image(example):
        try:
            example['image'] = example['image_path']
            return example
        except Exception as e:
            print(f"加载图像 {example['image_path']} 时出错: {str(e)}")
            return example
    
    dataset = dataset.cast_column('image_path', HFImage())
    dataset = dataset.rename_column('image_path', 'image')
    
    # 保存数据集
    dataset.save_to_disk(output_dir)
    
    print(f"已创建Huggingface数据集并保存到: {output_dir}")
    return dataset

# 使用示例
if __name__ == "__main__":
    input_file = "playground/DocVQA/simplified_train.json"
    output_file = "playground/DocVQA/sample.parquet"
    images_folder = "playground/DocVQA"  # 图像文件夹的根目录
    
    # 方法1: 生成Parquet文件
    convert_json_to_parquet(input_file, output_file, images_folder)
    
    # 方法2: 创建Huggingface数据集
    # create_hf_dataset(input_file, "output/docvqa_dataset", images_folder)