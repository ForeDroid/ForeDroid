import os
import json
import pandas as pd
import itertools
import re
import time
import sqlite3
import argparse
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


# === fromDirReports2csv_1.py ===

def process_all_apk_folders(input_root_folder, output_csv_path):
    first_write = not os.path.exists(output_csv_path)
    apk_folders = [folder for folder in os.listdir(input_root_folder) if os.path.isdir(os.path.join(input_root_folder, folder))]

    for apk_folder in tqdm(apk_folders, desc="Processing APK folders", unit="folder"):
        apk_folder_path = os.path.join(input_root_folder, apk_folder)
        json_file_path = os.path.join(apk_folder_path, "CallGraphInfo", "rule_analysis_report.json")
        if not os.path.exists(json_file_path):
            continue
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        for item in data:
            findings = item.get('findings', [])
            if len(findings) < 2:
                continue
            sapcg_info_list = findings[0]
            if not sapcg_info_list:
                continue
            sapcg_info = sapcg_info_list[0]
            sapcg_file = sapcg_info.get('sapcg_file', '')
            sensitiveAPI = sapcg_info.get('sensitiveAPI', '')
            call_chain_list = findings[1]
            for call_entry in call_chain_list:
                call_chain = call_entry.get('call_chain', [])
                call_chain_str = " -> ".join(call_chain)
                entry_method = call_chain[0] if call_chain else ''
                entry_point_total = call_entry.get('entry_point', {})
                entry_point = entry_point_total.get('entry_point', '')
                actions = entry_point_total.get('actions', [])
                actions_str = ", ".join(actions)
                origin_entry_point = entry_point_total.get('origin_entry_point', '')
                origin_actions = entry_point_total.get('origin_actions', [])
                origin_actions_str = ", ".join(origin_actions)
                origin_call_chain = entry_point_total.get('origin_call_chain', [])
                origin_call_chain_str = " -> ".join(origin_call_chain)
                origin_entry_method = origin_call_chain[0] if origin_call_chain else ''
                record = {
                    'apk_name': apk_folder,
                    'sapcg_file': sapcg_file,
                    'sensitiveAPI': sensitiveAPI,
                    'call_chain': call_chain_str,
                    'entry_method': entry_method,
                    'entry_point': entry_point,
                    'entry_point_actions': actions_str,
                    'origin_entry_point': origin_entry_point,
                    'origin_entry_point_actions': origin_actions_str,
                    'origin_call_chain': origin_call_chain_str,
                    'origin_entry_method': origin_entry_method,
                }
                df = pd.DataFrame([record])
                df.to_csv(output_csv_path, mode='a', header=first_write, index=False, encoding='utf-8')
                first_write = False


# === preprocess_merge_entry_2.py ===

def merge_columns(input_csv_path, output_csv_path, chunksize=100000):
    first_write = not os.path.exists(output_csv_path)
    current_id = 1
    for chunk in pd.read_csv(input_csv_path, encoding='utf-8', chunksize=chunksize):
        if 'sapcg_file' in chunk.columns:
            chunk = chunk.drop(columns=['sapcg_file'])
        chunk['entry_point'] = chunk['origin_entry_point'].fillna(chunk['entry_point'])
        chunk['call_chain'] = chunk['origin_call_chain'].fillna('') + " -> " + chunk['call_chain']
        chunk['entry_point_actions'] = chunk['origin_entry_point_actions'].fillna(chunk['entry_point_actions'])
        chunk['entry_method'] = chunk['origin_entry_method'].fillna(chunk['entry_method'])
        chunk = chunk.drop(columns=['origin_entry_point', 'origin_call_chain', 'origin_entry_point_actions', 'origin_entry_method'])
        chunk = chunk[chunk['entry_point'].notnull() & (chunk['entry_point'] != '')]
        chunk['id'] = range(current_id, current_id + len(chunk))
        current_id += len(chunk)
        chunk.to_csv(output_csv_path, mode='a', header=first_write, index=False, encoding='utf-8')
        first_write = False


def filter_action(input_csv_path, output_csv_path, chunksize=100000):
    first_write = not os.path.exists(output_csv_path)
    for chunk in pd.read_csv(input_csv_path, encoding='utf-8', chunksize=chunksize):
        chunk = chunk[chunk['entry_point_actions'].notnull() & (chunk['entry_point_actions'] != '')]
        chunk.to_csv(output_csv_path, mode='a', header=first_write, index=False, encoding='utf-8')
        first_write = False


# === filter_backstage_api_3.py ===

def update_df1_with_gui_info(input_csv_path_1, input_csv_path_2, output_csv_path):
    df1 = pd.read_csv(input_csv_path_1, encoding='utf-8')
    df1['apk'] = df1['apk_name']
    df1['call_chain_split'] = df1['call_chain'].dropna().str.split(" -> ")
    call_chain_exploded = df1[['id', 'apk', 'call_chain_split']].explode('call_chain_split')
    call_chain_exploded.rename(columns={'call_chain_split': 'callback'}, inplace=True)
    call_chains = set(call_chain_exploded['callback'].dropna())
    df2 = pd.read_csv(input_csv_path_2, sep=';', encoding='utf-8')
    df2 = df2.drop(columns=['id'], errors='ignore')
    filtered_df = df2[df2['apk'].isin(df1['apk'])]
    filtered_df = filtered_df[(filtered_df['callback'].isin(call_chains)) & (filtered_df['apk'].isin(df1['apk']))]
    filtered_df = filtered_df.merge(call_chain_exploded[['id', 'callback']], on='callback', how='left')
    filtered_df = filtered_df[['id', 'context', 'label', 'rawtext']]
    filtered_df.drop_duplicates(inplace=True)
    filtered_df = filtered_df.groupby('id', as_index=False).agg({
        'context': lambda x: '. '.join(set(x.dropna())),
        'label': lambda x: '. '.join(set(x.dropna())),
        'rawtext': lambda x: '. '.join(set(x.dropna()))
    })
    df1 = df1.merge(filtered_df, on='id', how='left')
    df1.to_csv(output_csv_path, index=False, encoding='utf-8')



# === llama_des_4.py ===

def process_call_chains(input_csv_path, output_csv_path, max_retries=5, retry_delay=2, pause_duration=120):
    df = pd.read_csv(input_csv_path)
    df["generated_description"] = None
    llm = ChatOpenAI(model="llama3-70b-8192")
    MAX_LENGTH = 2500
    output_file_exists = os.path.isfile(output_csv_path)

    for index, row in df.iterrows():
        call_chain = row["call_chain"]
        apk_name = row["apk_name"]
        row_id = row["id"]

        prompt = (
            "Please generate a concise behavior description for the following API call chain.\n"
            "- Use consistent noun/verb phrasing.\n"
            "- Combine functions if necessary.\n\n"
            "API call chain:\n" + call_chain
        )

        if len(prompt) > MAX_LENGTH:
            continue

        try:
            response = llm.invoke(prompt).content.strip().replace("\n", " ")
            df.at[index, "generated_description"] = response
        except Exception as e:
            df.at[index, "generated_description"] = "Description generation failed"
            print(f"Failed for {apk_name} ID {row_id}: {e}")

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"LLM descriptions saved to {output_csv_path}")


def init_db(db_path="call_chain_mapping.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS call_chain_mapping (
            call_chain TEXT,
            id TEXT,
            generated_description TEXT,
            PRIMARY KEY (call_chain, id)
        )
    """)
    conn.commit()
    conn.close()

def insert_description_files(description_files, db_path="call_chain_mapping.db"):
    conn = sqlite3.connect(db_path)
    for file in description_files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR REPLACE INTO call_chain_mapping (call_chain, id, generated_description) VALUES (?, ?, ?)""", 
                         (row['call_chain'], row['id'], row['generated_description']))
    conn.commit()
    conn.close()

def merge_generated_description(main_file, output_file, db_path="call_chain_mapping.db", batch_size=500):
    conn = sqlite3.connect(db_path)
    total_rows = sum(1 for _ in open(main_file, 'r', encoding='utf-8')) - 1
    first_write = not os.path.exists(output_file)

    with tqdm(total=total_rows, desc="Merging descriptions", unit="row") as pbar:
        for chunk in pd.read_csv(main_file, chunksize=batch_size, dtype=str):
            keys = [(row["call_chain"], row["id"]) for _, row in chunk.iterrows()]
            query = f"""
                SELECT call_chain, id, generated_description 
                FROM call_chain_mapping
                WHERE (call_chain, id) IN ({','.join(['(?, ?)'] * len(keys))})
            """
            result = pd.read_sql_query(query, conn, params=[item for pair in keys for item in pair])
            chunk = chunk.merge(result, on=["call_chain", "id"], how="left")
            chunk.to_csv(output_file, mode="a", index=False, header=first_write, encoding="utf-8")
            first_write = False
            pbar.update(len(chunk))
    conn.close()
    print(f"Merged file written to {output_file}")
    
    
# === main execution with argparse ===

def main(args):
    process_all_apk_folders(args.input_dir, args.temp_csv1)
    merge_columns(args.temp_csv1, args.temp_csv2)
    filter_action(args.temp_csv2, args.filtered_csv)
    update_df1_with_gui_info(args.filtered_csv, args.backstage_api_csv, args.final_csv)
    
    process_call_chains(args.final_csv, args.llm_output_csv)
    init_db()
    insert_description_files([args.llm_output_csv])
    merge_generated_description(args.final_csv, args.merged_output_csv)

    print("ForeDroid static behavior pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForeDroid Static Behavior Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing ICCBot JSON outputs")
    parser.add_argument("--backstage_api_csv", type=str, required=True, help="CSV file containing Backstage API call traces")
    parser.add_argument("--temp_csv1", type=str, default="temp1.csv")
    parser.add_argument("--temp_csv2", type=str, default="temp2.csv")
    parser.add_argument("--filtered_csv", type=str, default="filtered.csv")
    parser.add_argument("--final_csv", type=str, default="final.csv")
    parser.add_argument("--llm_output_csv", type=str, default="llm_output.csv")
    parser.add_argument("--merged_output_csv", type=str, default="final_merged.csv")
    args = parser.parse_args()
    main(args)


