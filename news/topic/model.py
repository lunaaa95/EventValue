from gensim import corpora
from gensim.models import LdaModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import jieba
import re
from tqdm import tqdm

from ..storyline.embedding import StorylineEmbedding


class Tokenizer:
    def __init__(self) -> None:
        jieba.load_userdict('extra/dict.txt')

        with open('extra/stop_words.txt') as f:
            stop_words = f.readlines()
        self.stop_words = set([w.strip() for w in stop_words])

        self.decimal_regex = re.compile(r'^(-?\d+)(\.\d+)?%?$')

    def __call__(self, text):
        words = []
        for w in jieba.cut(text):
            if w not in self.stop_words and self.decimal_regex.search(w) is None:
                words.append(w)
        return words


class TopicModel:
    def __init__(self, nr_topics=10) -> None:
        tokenizer = Tokenizer()
        embedding_model = StorylineEmbedding(max_length=1024)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(tokenizer=tokenizer)
        ctfidf_model = ClassTfidfTransformer()

        self.topic_model = BERTopic(
            embedding_model=embedding_model,               # Step 1 - Extract embeddings
            umap_model=umap_model,                         # Step 2 - Dimension Reduction
            hdbscan_model=hdbscan_model,                   # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,             # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,                     # Step 5 - Extract topic words
            nr_topics=nr_topics,
            )
    
    def fit(self, docs):
        topics, probs = self.topic_model.fit_transform(docs)
    
    def get_document_topic(self, docs):
        return self.topic_model.get_document_info(docs)

    @classmethod
    def load(cls, model_path):
        model = cls.__new__(cls)
        tokenizer = Tokenizer()
        model.topic_model = BERTopic.load(model_path)
        return model

    def save(self, model_path):
        self.topic_model.save(model_path)
