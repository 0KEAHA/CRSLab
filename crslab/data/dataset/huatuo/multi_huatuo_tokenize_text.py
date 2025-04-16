import json
import os
import time # 用于计时比较
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools # 用于传递额外参数给 map
import itertools # 用于简化扁平化操作

# 假设 resources 和 DATASET_PATH 已经定义好
# from resources import resources
# from crslab.config import DATASET_PATH
# Mock definitions for demonstration:
resources = {
    'pkuseg': {'folder_path': 'dummy_dataset'}
}
# 请确保这个路径是正确的，指向包含 source_*.json 文件的目录
DATASET_PATH = r'H:\ThesisCode\CRSLab\CRSLab\data\dataset\huatuo\pkuseg'


def _tokenize(tokenize_method ,string):
    """
    Tokenize the data using the provided tokenizer.
    :param tokenize_method: The name of the tokenizer ('pkuseg').
    :param string: The string to tokenize.
    :return: The tokenized data (list of words).
    """
    # 注意：pkuseg 会在每个被调用的进程中加载和初始化。
    # 对于 pkuseg 这种轻量级库，通常问题不大。
    # 如果是大型模型或初始化开销大的库，考虑使用 Pool(initializer=...)
    if tokenize_method == 'pkuseg':
        # 动态导入，确保每个进程都能找到
        try:
            import pkuseg
        except ImportError:
            print("错误：请先安装 pkuseg 库 (pip install pkuseg)")
            return string.split() # 简单回退

        # 可以在这里缓存 seg 实例以提高同一进程内的效率，但跨进程无效
        if 'seg_instance' not in globals():
            print(f"Initializing pkuseg in process {os.getpid()}")
            globals()['seg_instance'] = pkuseg.pkuseg()
        seg = globals()['seg_instance']

        # 更简单的方式：每次调用都创建实例，开销相对较小
        seg = pkuseg.pkuseg()
        tokenized_data = seg.cut(string)
        return tokenized_data
    else:
        # 可以添加对其他分词器的支持
        print(f"警告：不支持的分词器 '{tokenize_method}'，返回按空格分割的结果。")
        return string.split()

# --- 新的并行处理函数，粒度为 text ---
def process_data_parallel_text_level(tokenize_method, num_workers=None):
    """
    并行处理数据文件，并行粒度为每个对话中的 'text' 字段。
    :param tokenize_method: 使用的分词方法名称。
    :param num_workers: 使用的进程数，默认为 CPU 核心数。
    """
    if num_workers is None:
        num_workers = cpu_count()
        print(f"未指定工作进程数，将使用所有可用核心: {num_workers}")

    resource = resources.get(tokenize_method)
    if not resource:
        print(f"错误: 在 resources 中未找到分词器 '{tokenize_method}' 的配置。")
        return

    dpath = DATASET_PATH
    data_list = ['train_data.json', 'valid_data.json', 'test_data.json']

    print(f"\n开始使用 {num_workers} 个进程进行并行处理 (粒度: text)...")

    for data_file in data_list: # 外层文件处理保持串行
        source_data_path = os.path.join(dpath, "source_" + data_file)
        out_path = os.path.join(dpath, data_file)

        if not os.path.exists(source_data_path):
            print(f"警告：源文件不存在，跳过: {source_data_path}")
            continue

        print(f"\n正在处理文件: {source_data_path}")
        start_time = time.time()

        try:
            with open(source_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"错误：加载 JSON 文件失败 {source_data_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"错误：期望 JSON 顶层是列表，但得到 {type(data)}。跳过文件 {source_data_path}")
            continue
        if not data:
            print(f"警告：文件 {source_data_path} 为空列表。")
            # 创建空的输出文件
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            print(f"空的 {data_file} 已处理并保存到 {out_path}.")
            continue

        # --- 1. 提取所有文本并记录位置 ---
        texts_to_process = []
        text_locations = [] # 存储 (item_index, conv_index)
        for i, item in enumerate(data):
            conv_list = item.get('conv', [])
            if not isinstance(conv_list, list): # 添加健壮性检查
                 print(f"警告: item {i} 中的 'conv' 不是列表，跳过此 item 的对话处理。")
                 continue
            for j, conv in enumerate(conv_list):
                 # 确保 conv 是字典并且包含 'text' 字段，且 'text' 是字符串
                if isinstance(conv, dict) and 'text' in conv and isinstance(conv['text'], str):
                    texts_to_process.append(conv['text'])
                    text_locations.append((i, j))
                # else:
                    # 可以选择在这里打印警告，如果数据格式不符合预期
                    # print(f"警告: item {i}, conv {j} 格式不正确或缺少文本，已跳过。")


        if not texts_to_process:
            print(f"警告：在文件 {source_data_path} 中没有找到可处理的文本。")
            # 即使没有文本，也应保存原始（或处理后的）数据结构
            with open(out_path, 'w', encoding='utf-8') as f:
                # 因为没有文本被修改，可以直接保存原始 data
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"{data_file} (无文本处理) 已保存到 {out_path}.")
            continue


        # --- 2. 并行分词 ---
        # 使用 functools.partial 固定 tokenize_method 参数
        tokenize_func = functools.partial(_tokenize, tokenize_method)

        print(f"开始对 {len(texts_to_process)} 个文本片段进行并行分词...")
        tokenized_texts = []
        # 使用 with 语句确保进程池正确关闭
        try:
            with Pool(processes=num_workers) as pool:
                # 使用 imap 以便与 tqdm 结合显示进度
                # pool.imap 按顺序返回结果，这对于后续重构很重要
                tokenized_texts = list(tqdm(pool.imap(tokenize_func, texts_to_process, chunksize=100), # chunksize 可调整
                                            total=len(texts_to_process),
                                            desc=f"分词 ({data_file})"))
        except Exception as e:
            print(f"错误：并行分词过程中发生严重错误: {e}")
            # 可以在这里决定是退出、跳过文件还是尝试串行处理作为后备
            continue # 跳过当前文件

        if len(tokenized_texts) != len(text_locations):
             print(f"错误：分词结果数量 ({len(tokenized_texts)}) 与预期 ({len(text_locations)}) 不匹配！跳过文件 {data_file} 的保存。")
             continue

        # --- 3. 重构数据 ---
        print("分词完成，正在将结果写回原数据结构...")
        for k, tokenized_text in enumerate(tokenized_texts):
            item_index, conv_index = text_locations[k]
            # 定位到原始数据中的对应位置并更新 'text'
            try:
                # 再次检查以防万一
                if isinstance(data[item_index].get('conv'), list) and len(data[item_index]['conv']) > conv_index:
                     conv_dict = data[item_index]['conv'][conv_index]
                     if isinstance(conv_dict, dict) and 'text' in conv_dict:
                          conv_dict['text'] = tokenized_text
                     else:
                          print(f"警告：重构时发现位置 ({item_index}, {conv_index}) 的结构已更改或无效，跳过更新。")
                else:
                     print(f"警告：重构时发现位置 ({item_index}, {conv_index}) 无效，跳过更新。")

            except IndexError:
                print(f"严重错误：在重构数据时索引越界 ({item_index}, {conv_index})。数据可能已损坏。")
                # 可能需要更健壮的错误处理
                continue # 跳过这个更新
            except Exception as e:
                 print(f"错误：重构数据时发生未知错误在位置 ({item_index}, {conv_index}): {e}")
                 continue # 跳过这个更新


        # --- 保存结果 ---
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            end_time = time.time()
            print(f"{data_file} 已成功处理并保存到 {out_path}.")
            print(f"文件处理耗时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"错误：写入 JSON 文件失败 {out_path}: {e}")

# --- 运行新的并行版本 ---
# 使用 if __name__ == '__main__': 是 multiprocessing 的最佳实践
if __name__ == '__main__':
    print("\n" + "="*20 + " 运行多核并行版本 (粒度: text) " + "="*20)
    start_parallel = time.time()
    # 可以指定进程数，例如 process_data_parallel_text_level('pkuseg', num_workers=4)
    process_data_parallel_text_level('pkuseg', num_workers=8)
    end_parallel = time.time()
    print(f"\n多核并行版本 (粒度: text) 总耗时: {end_parallel - start_parallel:.2f} 秒")
    print("="*60)