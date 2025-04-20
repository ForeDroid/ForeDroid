import csv
import re
import numpy as np
import pandas as pd
import spacy
import webcolors
from gensim.models import KeyedVectors
from nltk.corpus import stopwords, wordnet, words

class EntryPointFilter:
    def __init__(self, model_path='../data/GoogleNews-vectors-negative300.bin'):
        self.nlp = spacy.load("en_core_web_sm")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.lorem = ['lorem', 'ipsum', 'dolor']
        self.extra_stopwords = ['null','button','menu','aaa','abc','adc','adn','etc']
        self.non_stopwords = ["not","yes","no"]
        self.sw = set(stopwords.words('english'))
        self.sw.update(webcolors.CSS3_NAMES_TO_HEX)
        self.sw.update(self.lorem)
        self.sw.update(self.extra_stopwords)
        for w in self.non_stopwords:
            self.sw.discard(w)
        self.word_set = set(self.model.index_to_key)
        self.allowed_tags = {'NN','NNS','NNP','NNPS','VB','VBD','VBN','VBP','VBZ','UH','JJ','VBG'}
        self.num_features = 300
        self.rnorm = re.compile('[\n\r";]')
        self.android_terms = {
            "service", "activity", "android", "provider", "receiver",
            "broadcast", "intent", "application", "webview"
        }

    def process_csv(self, in_file, out_file, vec_file):
        with open(in_file, 'r', encoding='utf-8') as csvfile, open(out_file, 'w', encoding='utf-8') as csvout:
            reader = csv.DictReader(csvfile, delimiter=',')
            writer = csv.DictWriter(csvout, fieldnames=['id', 'entry_point_cleaned'], delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            all_entries_dict = {}
            for row in reader:
                raw_id = row.get('id', '')
                method_name = self.extract_method_name(row.get('entry_method', ''))
                cleaned_method = self.clean_entry_point(method_name)
                entry_point = self.keep_after_last_dot(row.get('entry_point', ''))
                cleaned_point = self.clean_entry_point(entry_point)
                raw_entry_point_actions = row.get('entry_point_actions', '')
                entry_point_actions = self.clean_and_tokenize_action(raw_entry_point_actions).strip()
                final_cleaned = " ".join(filter(None, [cleaned_point, cleaned_method, entry_point_actions]))
                writer.writerow({'id': raw_id, 'entry_point_cleaned': final_cleaned})
                avg_vec = self.make_feature_vec(final_cleaned)
                all_entries_dict[raw_id] = avg_vec
            df = pd.DataFrame(np.array(list(all_entries_dict.values())), index=list(all_entries_dict.keys()), columns=range(self.num_features))
            df.index.name = 'id'
            df.to_csv(vec_file, sep=';')
            return df

    def clean_entry_point(self, text):
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
        text = self.better_camel_split(text).replace('-', ' ').replace('_',' ').replace('/', ' ')
        text = re.sub('[0-9]', '', text).strip()
        doc = self.nlp(text)
        res_words = []
        for w in doc:
            token_lemma = w.lemma_.lower()
            if any(a_term in token_lemma for a_term in self.android_terms):
                res_words.append(token_lemma)
                continue
            if w.tag_ in self.allowed_tags or token_lemma in self.non_stopwords:
                if self.is_valid_word(token_lemma):
                    res_words.append(token_lemma)
        return ' '.join(res_words)

    def clean_and_tokenize_action(self, action):
        stop_words = {"android", "intent", "action", "com", "org"}
        action = action.lower()
        action = re.sub(r"^android\.intent\.action\.", "", action)
        tokens = re.split(r"[._]", action)
        tokens = [t for t in tokens if t and t not in stop_words]
        doc = self.nlp(" ".join(tokens))
        res_words = []
        for w in doc:
            token_lemma = w.lemma_.lower()
            if token_lemma in self.android_terms or any(token_lemma.startswith(t) or token_lemma.endswith(t) for t in self.android_terms):
                res_words.append(token_lemma)
                continue
            if w.tag_ in self.allowed_tags or token_lemma in self.non_stopwords:
                if self.is_valid_word(token_lemma):
                    res_words.append(token_lemma)
        return ' '.join(res_words)

    def is_valid_word(self, w):
        return w not in self.sw and w in self.word_set

    def make_feature_vec(self, cleaned_text):
        tokens = set(cleaned_text.split())
        featureVec = np.zeros((self.num_features,), dtype="float32")
        nwords = sum(1 for token in tokens if token in self.word_set)
        for token in tokens:
            if token in self.word_set:
                featureVec += self.model[token]
        return featureVec / nwords if nwords > 0 else featureVec

    def keep_after_last_dot(self, s: str) -> str:
        return s.rsplit('.', 1)[-1].strip() if isinstance(s, str) else ''

    def extract_method_name(self, full_signature):
        if not isinstance(full_signature, str): return ''
        match = re.match(r'<[^:]+:\s*[^ ]+\s+([^(]+)\(', full_signature.strip())
        return match.group(1).strip() if match else full_signature

    def better_camel_split(self, s: str) -> str:
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        return re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)