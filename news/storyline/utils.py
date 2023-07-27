from ..topic.model import TopicModel
from .model import Storyline

import pandas as pd


def build_stroylines_by_topic(data: pd.DataFrame, news_column, time_column, info_column, nr_topics=10):
    # 主题模型抽取主题
    topic_model = TopicModel(nr_topics)
    news_docs = data[news_column].to_list()
    topic_model.fit(news_docs)
    news_with_topic = topic_model.get_document_topic(news_docs)
    news_with_topic['time'] = data[time_column]
    news_with_topic['related'] = data[info_column]

    # 遍历每个主题
    topics = news_with_topic.groupby('Topic')
    storylines = []
    for t, d in topics:
        sl = Storyline(
            news=d['Document'],
            time=d['time'],
            related=d['related'],
            topic=d.iloc[0]['Representation']
        )
        storylines.append(sl)

    return storylines
