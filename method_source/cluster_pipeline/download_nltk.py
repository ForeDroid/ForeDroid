import nltk
import ssl

# download_nltk.py
import nltk

# 设置代理
# nltk.set_proxy('http://你的代理地址:端口')  # 例如 'http://127.0.0.1:7890'
nltk.download('stopwords')

# 如果遇到 SSL 证书问题（在某些环境下），可以临时取消证书验证
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 确保 stopwords 已下载
# nltk.download('stopwords', quiet=True)