import json
import os
import time # 用于计时比较
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools # 用于传递额外参数给 map
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM



resources = {
    'pkuseg': {'folder_path': 'dummy_dataset'}
}
DATASET_PATH = 'H:\ThesisCode\CRSLab\CRSLab\data\dataset\huatuo\qwen' 



model_path = "H:\ThesisCode\model\qwen"


# _tokenize 函数保持不变，但注意 pkuseg 在每个进程中可能需要重新初始化
# 如果初始化成本高，可以考虑使用 Pool 的 initializer
def _tokenize(tokenize_method ,string):
    """
    Tokenize the data using the provided tokenizer.
    :param tokenize_method: The name of the tokenizer ('pkuseg').
    :param string: The string to tokenize.
    :return: The tokenized data (list of words).
    """
   
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
    elif tokenize_method == 'qwen':
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if 'seg_instance' not in globals():
            print(f"Initializing pkuseg in process {os.getpid()}")
            globals()['seg_instance'] = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            pad_token='<|endoftext|>',
            config=config
        )
        tokenizer = globals()['seg_instance']
        tokens = tokenizer.encode(string, add_special_tokens=False)
        return tokens
    else:
        # 可以添加对其他分词器的支持
        print(f"警告：不支持的分词器 '{tokenize_method}'，返回按空格分割的结果。")
        return string.split()

# --- 并行处理的核心工作函数 ---
def _process_item(item, tokenize_method):
    """
    处理单个数据项（item），对其所有对话进行分词。
    设计为可以被 multiprocessing.Pool.map 或 starmap 调用。
    :param item: 一个包含 'conv' 列表的字典。
    :param tokenize_method: 使用的分词方法名称。
    :return: 处理（分词）后的 item 字典。
    """
    for conv in item.get('conv', []): # 使用 .get 提供鲁棒性
        if 'text' in conv and isinstance(conv['text'], str):
            conv['text'] = _tokenize(tokenize_method, conv['text'])
        # 如果 text 已经是 list 或其他非字符串类型，可以选择跳过或报错
    return item

# --- 并行版本的 process_data ---
def process_data_parallel(tokenize_method, num_workers=None):
    """
    并行处理数据文件。
    :param tokenize_method: 使用的分词方法名称。
    :param num_workers: 使用的进程数，默认为 CPU 核心数。
    """
    if num_workers is None:
        num_workers = cpu_count()
        print(f"未指定工作进程数，将使用所有可用核心: {num_workers}")

    # resource = resources.get(tokenize_method)
    # if not resource:
    #     print(f"错误: 在 resources 中未找到分词器 '{tokenize_method}' 的配置。")
    #     return

    #dpath = os.path.join(DATASET_PATH, resource['folder_path'])
    dpath = DATASET_PATH
    data_list = ['train_data.json', 'valid_data.json', 'test_data.json']

    print(f"\n开始使用 {num_workers} 个进程进行并行处理...")

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
            # 仍然创建一个空的输出文件
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            print(f"空的 {data_file} 已处理并保存到 {out_path}.")
            continue


        # --- 使用 multiprocessing.Pool 进行并行处理 ---
        # functools.partial 用来固定 _process_item 函数的 tokenize_method 参数
        # 这样 map 函数只需要传递 data 中的每个 item
        process_func = functools.partial(_process_item, tokenize_method=tokenize_method)

        # 使用 with 语句确保进程池正确关闭
        with Pool(processes=num_workers) as pool:
            # 使用 map 将 process_func 应用到 data 列表的每个元素上
            # map 会阻塞直到所有任务完成
            # 使用 tqdm 显示并行处理的进度
            results = list(tqdm(pool.imap(process_func, data), total=len(data), desc=f"分词 ({data_file})"))
            # pool.imap 是 map 的惰性版本，配合 list() 和 tqdm 可以显示进度

        # --- 并行处理结束 ---

        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            end_time = time.time()
            print(f"{data_file} 已成功处理并保存到 {out_path}.")
            print(f"文件处理耗时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"错误：写入 JSON 文件失败 {out_path}: {e}")




if __name__ == '__main__':
    # # 清理之前可能生成的输出文件，以便比较
    # for f_name in ['train_data.json', 'valid_data.json', 'test_data.json']:
    #     out_f = os.path.join(dummy_folder, f_name)
    #     if os.path.exists(out_f):
    #         os.remove(out_f)

    print("\n" + "="*20 + " 运行多核并行版本 " + "="*20)
    start_parallel = time.time()
    # 可以指定进程数，例如 process_data_parallel('pkuseg', num_workers=4)
    process_data_parallel('qwen', num_workers=8)
    end_parallel = time.time()
    print(f"\n多核并行版本总耗时: {end_parallel - start_parallel:.2f} 秒")
    print("="*50)

    # 可以在这里添加代码来验证输出文件是否与串行版本一致（如果运行了的话）