import json
import os


def process_docvqa_json(input_path, output_path):
    """
    从SP-DocVQA数据集JSON文件中提取简化数据并保存到指定路径
    
    参数:
    input_path (str): 输入JSON文件的路径
    output_path (str): 输出JSON文件的路径
    
    返回:
    list: 提取的简化数据列表
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所需字段
    simplified_data = []
    for item in data['data']:
        simplified_item = {
            "image": item['image'],
            "question": item['question'],
            "answers": item['answers']
        }
        simplified_data.append(simplified_item)
    
    # 保存为新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成! 已将简化数据保存到: {output_path}")
    print(f"共处理 {len(simplified_data)} 条记录")
    
    return simplified_data

# 使用示例
if __name__ == "__main__":
    input_file = "playground/DocVQA/train_v1.0_withQT.json"
    output_file = "playground/DocVQA/simplified_train.json"
    
    process_docvqa_json(input_file, output_file)