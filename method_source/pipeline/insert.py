import os
import pandas as pd
import time
import re
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import sqlite3

# ================= 3. 数据库操作部分 =================

DB_FILE = "call_chain_mapping.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS call_chain_mapping (
            call_chain TEXT,
            id TEXT,
            generated_description TEXT,
            PRIMARY KEY (call_chain, id)
        )
    """)
    conn.commit()
    conn.close()

def insert_description_files(description_file_paths):
    """
    修复版：正确接收列表，批量插入
    """
    conn = sqlite3.connect(DB_FILE)
    # 开启 WAL 模式提高并发写入性能
    conn.execute("PRAGMA journal_mode=WAL;") 
    
    for file_path in description_file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"📥 Importing {file_path} into DB...")
        chunk_iter = pd.read_csv(file_path, chunksize=5000, dtype=str) # 分块读取防止内存爆
        
        for chunk in chunk_iter:
            data = chunk[['call_chain', 'id', 'generated_description']].values.tolist()
            conn.executemany("""
                INSERT OR REPLACE INTO call_chain_mapping (call_chain, id, generated_description)
                VALUES (?, ?, ?)
            """, data)
            conn.commit()
            
    print("✅ DB Import finished, description files inserted or updated successfully.")
    conn.close()

def merge_generated_description(main_file, output_file, batch_size=500):
    """
    读取大 CSV 文件，逐行补充 generated_description
    :param main_file: 输入的主文件路径
    :param output_file: 输出文件路径
    :param batch_size: 每批次处理的行数
    """
    conn = sqlite3.connect(DB_FILE)

    # 获取文件总行数，用于进度条
    total_rows = sum(1 for _ in open(main_file, 'r', encoding='utf-8')) - 1  # -1 去掉表头
    first_write = not os.path.exists(output_file)
        
    # 统计成功和未成功添加的行数
    successfully_added = 0
    not_added = 0

    with tqdm(total=total_rows, desc="Merging descriptions", unit="row") as pbar:
        for chunk in pd.read_csv(main_file, chunksize=batch_size, dtype=str):

            # 将 chunk 分批查询到数据库，防止一次性传入过多的参数
            # 每次将 call_chain 和 id 放到 IN 子句中分批处理
            for start in range(0, len(chunk), batch_size):
                end = min(start + batch_size, len(chunk))
                batch = chunk.iloc[start:end]
                keys = [(row["call_chain"], row["id"]) for _, row in batch.iterrows()]

                if not keys:  # 处理空数据块
                    successfully_added += len(chunk)
                    chunk.to_csv(output_file, mode="a", index=False, header=first_write, encoding="utf-8")
                    first_write = False
                    pbar.update(len(chunk))
                    continue
                # 修改查询条件：用OR连接多个AND条件，替代元组IN语法
                conditions = " OR ".join(["(call_chain = ? AND id = ?)"] * len(keys))
                query = f"""
                    SELECT call_chain, id, generated_description 
                    FROM call_chain_mapping
                    WHERE {conditions}
                """

                # 执行查询，并传递正确的参数
                result = pd.read_sql_query(query, conn, params=[item for sublist in keys for item in sublist])
                # 合并数据
                batch = batch.merge(result, on=["call_chain", "id"], how="left")
                # 在合并数据后添加：
                successfully_added += batch['generated_description'].notna().sum()
                not_added += batch['generated_description'].isna().sum()

                # 追加到 CSV 文件
                batch.to_csv(output_file, mode="a", index=False, header=first_write, encoding="utf-8")
                first_write = False  # 仅第一次写入表头

                pbar.update(len(batch))

    conn.close()
    # print(f"合并完成，已保存到 {output_file}")
    print("合并完成，已保存.")
    print(f"成功添加的行数：{successfully_added}")
    print(f"未成功添加的行数：{not_added}")

if __name__ == "__main__":
    # 1. 配置路径
    # output_dir = "./output_benign"
    # output_dir = "./output_test_benign_1"
    # output_dir = "./output_test_malware"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_train")

    input_csv = os.path.join(output_dir, "final_training_data.csv")
    # input_csv = os.path.join(output_dir, "deduplicated_gui_callchain.csv")
    desc_output_csv = os.path.join(output_dir, "result_with_descriptions.csv")
    final_merged_csv = os.path.join(output_dir, "final_training_data_with_desc.csv")

    # 2. 创建数据库
    init_db()

    # 3. 导入数据库
    # 注意：这里传入的是列表 [path]，可以传入多个文件
    insert_description_files([desc_output_csv]) 

    # 4. 合并回原始数据
    merge_generated_description(input_csv, final_merged_csv)

