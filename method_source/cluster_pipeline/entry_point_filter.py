# 导入必要的库
import csv
import re
import numpy as np
import pandas as pd
import spacy
import webcolors
from gensim.models import KeyedVectors
from nltk.corpus import stopwords, wordnet, words
import sys

# 增大字段大小限制
csv.field_size_limit(sys.maxsize)

# 入口点过滤器类，用于处理和清洗Android应用的入口点数据
class EntryPointFilter:
    def __init__(self, model_path='../models/GoogleNews-vectors-negative300.bin', preserve_prepositions=False):
        """
        preserve_prepositions: 如果为 True，则保留介词和助动词（用于 GUI 管道），
                              因为"log in" ≠ "log out"。默认 False 用于 Background 管道。
        """
        # 初始化spacy的英文模型
        self.nlp = spacy.load("en_core_web_sm") 
        # 加载预训练的word2vec模型
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        # 定义一些Lorem ipsum示例词
        self.lorem = ['lorem', 'ipsum', 'dolor']
        # 额外的停用词
        self.extra_stopwords = ['null','button','menu','aaa','abc','adc','adn','etc']
        # 不应该被视为停用词的词
        self.non_stopwords = ["not","yes","no"]
        # GUI 管道中需要保留的介词和助动词
        self.gui_preserved_words = ["in", "out", "up", "down", "on", "off", "to", "for",
                                     "with", "by", "at", "from", "of", "into", "onto", "over"]
        # 获取英文停用词集合
        self.sw = set(stopwords.words('english'))
        # 添加CSS颜色名称到停用词集合
        self.sw.update(webcolors.CSS3_NAMES_TO_HEX)
        # 添加Lorem ipsum词和额外停用词
        self.sw.update(self.lorem)
        self.sw.update(self.extra_stopwords)
        # 从停用词集合中移除non_stopwords中的词
        for w in self.non_stopwords:
            self.sw.discard(w)
        # 如果启用 preserve_prepositions，从停用词集合中移除 GUI 场景关键词
        self.preserve_prepositions = preserve_prepositions
        if preserve_prepositions:
            for w in self.gui_preserved_words:
                self.sw.discard(w)
        # 获取word2vec模型中的所有词
        self.word_set = set(self.model.index_to_key)
        # 允许的词性标签
        self.allowed_tags = {'NN','NNS','NNP','NNPS','VB','VBD','VBN','VBP','VBZ','UH','JJ','VBG'}
        # 特征向量的维度
        self.num_features = 300
        # 用于正则表达式匹配换行符、引号等的编译对象
        self.rnorm = re.compile('[\n\r";]')
        # Android相关的术语集合
        self.android_terms = {
            "service", "activity", "android", "provider", "receiver",
            "broadcast", "intent", "application", "webview"
        }

    # 处理CSV文件，清洗入口点数据并生成特征向量
    # === 改动：新增 sep 参数，支持 ';' 分隔的 CSV（info_fusion 产物） ===这个改动去掉了
    def process_csv(self, in_file, out_file, vec_file):
        with open(in_file, 'r', encoding='utf-8') as csvfile, open(out_file, 'w', encoding='utf-8') as csvout:
            reader = csv.DictReader(csvfile, delimiter=',')
            writer = csv.DictWriter(csvout, fieldnames=['id', 'entry_point_cleaned'], delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            all_entries_dict = {}
            for row in reader:
                raw_id = row.get('id', '')
                # 提取方法名
                method_name = self.extract_method_name(row.get('entry_method', ''))
                # 清洗方法名
                cleaned_method = self.clean_entry_point(method_name)
                # 获取入口点并保留最后一个点之后的部分
                entry_point = self.keep_after_last_dot(row.get('entry_point', ''))
                # 清洗入口点
                cleaned_point = self.clean_entry_point(entry_point)
                # 获取入口点动作并清洗
                raw_entry_point_actions = row.get('entry_point_actions', '')
                entry_point_actions = self.clean_and_tokenize_action(raw_entry_point_actions).strip()
                # 合并所有清洗后的部分
                final_cleaned = " ".join(filter(None, [cleaned_point, cleaned_method, entry_point_actions]))
                writer.writerow({'id': raw_id, 'entry_point_cleaned': final_cleaned})
                # 生成特征向量
                avg_vec = self.make_feature_vec(final_cleaned)
                all_entries_dict[raw_id] = avg_vec
            # 将特征向量保存为DataFrame并写入CSV
            df = pd.DataFrame(np.array(list(all_entries_dict.values())), index=list(all_entries_dict.keys()), columns=range(self.num_features))
            df.index.name = 'id'
            df.to_csv(vec_file, sep=';')
            return df

    # 清洗入口点文本
    def clean_entry_point(self, text):
        # 移除非字母数字字符
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
        # 处理驼峰命名和下划线等
        text = self.better_camel_split(text).replace('-', ' ').replace('_',' ').replace('/', ' ')
        # 移除数字
        text = re.sub('[0-9]', '', text).strip()
        # 使用spacy处理文本
        doc = self.nlp(text)
        res_words = []
        for w in doc:
            token_lemma = w.lemma_.lower()
            # 如果是Android相关术语，直接保留
            if any(a_term in token_lemma for a_term in self.android_terms):
                res_words.append(token_lemma)
                continue
            # 检查词性标签是否在允许的范围内
            if w.tag_ in self.allowed_tags or token_lemma in self.non_stopwords:
                if self.is_valid_word(token_lemma):
                    res_words.append(token_lemma)
        return ' '.join(res_words)

    # 清洗和分词入口点动作
    def clean_and_tokenize_action(self, action):
        stop_words = {"android", "intent", "action", "com", "org"}
        action = action.lower()
        # 移除"android.intent.action."前缀
        action = re.sub(r"^android\.intent\.action\.", "", action)
        # 按点和下划线分词
        tokens = re.split(r"[._]", action)
        # 移除停用词
        tokens = [t for t in tokens if t and t not in stop_words]
        doc = self.nlp(" ".join(tokens))
        res_words = []
        for w in doc:
            token_lemma = w.lemma_.lower()
            # 如果是Android相关术语，直接保留
            if token_lemma in self.android_terms or any(token_lemma.startswith(t) or token_lemma.endswith(t) for t in self.android_terms):
                res_words.append(token_lemma)
                continue
            # 检查词性标签是否在允许的范围内
            if w.tag_ in self.allowed_tags or token_lemma in self.non_stopwords:
                if self.is_valid_word(token_lemma):
                    res_words.append(token_lemma)
        return ' '.join(res_words)

    # 检查单词是否有效（不在停用词中且在word2vec模型中）
    def is_valid_word(self, w):
        return w not in self.sw and w in self.word_set

    # 生成特征向量
    def make_feature_vec(self, cleaned_text):
        tokens = set(cleaned_text.split())
        featureVec = np.zeros((self.num_features,), dtype="float32")
        # 计算有效词的数量
        nwords = sum(1 for token in tokens if token in self.word_set)
        # 累加所有有效词的向量
        for token in tokens:
            if token in self.word_set:
                featureVec += self.model[token]
        # 返回平均向量
        return featureVec / nwords if nwords > 0 else featureVec

    # 保留字符串中最后一个点之后的部分
    def keep_after_last_dot(self, s: str) -> str:
        return s.rsplit('.', 1)[-1].strip() if isinstance(s, str) else ''

    # 从完整方法签名中提取方法名
    def extract_method_name(self, full_signature):
        if not isinstance(full_signature, str): return ''
        # 使用正则匹配方法名
        match = re.match(r'<[^:]+:\s*[^ ]+\s+([^(]+)\(', full_signature.strip())
        return match.group(1).strip() if match else full_signature

    # 更好的驼峰命名分割
    def better_camel_split(self, s: str) -> str:
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        return re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)