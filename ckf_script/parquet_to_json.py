import pandas as pd
import os



def parquet_to_json(parquet_file, json_file, json_format='records'):
    """
    将 Parquet 文件转换为 JSON。
    
    参数:
        parquet_file: Parquet 文件路径
        json_file: 输出 JSON 文件路径
        json_format: 'records' (普通列表) 或 'lines' (每行一个JSON对象)
    """
    try:
        # 1. 读取 Parquet 文件
        df = pd.read_parquet(parquet_file, engine='pyarrow')
        print(f"成功读取 Parquet 文件，包含 {len(df)} 行数据。")

        # 2. 转换为 JSON
        if json_format == 'records':
            # 输出为一个大的 JSON 数组: [{"col": val}, {"col": val}]
            # indent=4 让文件由人类可读的缩进（文件体积会变大，生产环境可去掉）
            df.to_json(json_file, orient='records', indent=4, force_ascii=False)
            
        elif json_format == 'lines':
            # 输出为 JSON Lines (NDJSON)，每一行是一个对象
            # 这种格式更适合通过网络流式传输或处理大数据
            df.to_json(json_file, orient='records', lines=True, force_ascii=False)

        print(f"成功转换并保存至: {json_file}")

    except Exception as e:
        print(f"转换过程中发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你实际的文件名
    input_dir = '/mnt/whuscs/ckf/Tool-MT/EnvTuningProject/data'
    output_dir = '/mnt/whuscs/ckf/Tool-MT/EnvTuningProject/data_json'
    
    os.makedirs(output_dir,exist_ok=True)
    for file_name in os.listdir(input_dir):
        input_filename = os.path.join(input_dir, file_name)
        output_filename = os.path.join(output_dir, file_name.split(".")[0]+".json")
        # 运行转换
        # 这里的 orient='records' 会生成最通用的 JSON 格式
        parquet_to_json(input_filename, output_filename, json_format='records')
    
    