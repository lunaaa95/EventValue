from transformers import AutoModel, AutoTokenizer
import torch

from .model import Storyline


class NewsEmbedding:
    def __init__(self, max_length=1024) -> None:
        #Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.max_length = max_length

    @staticmethod
    def sentence_mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def __call__(self, news: list[str]):
        """embed news in list one by one

        news:
            type: list[str]
            shape: (n,)

        return:
            type: torch.Tensor
            shape: (n, dim_embedding)
        """

        #Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = self.sentence_mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings


class StorylineEmbedding:
    def __init__(self, max_length=1024) -> None:
        self.embedding_model = NewsEmbedding(max_length)
    
    @staticmethod
    def news_mean_pooling(sentence_embeddings):
        return sentence_embeddings.mean(dim=0)
    
    def __call__(self, storyline: Storyline):
        """embed storyline by mean pooling news embeddings

        storyline:
            type: Storyline
        
        return:
            type: torch.Tensor
            shape: (dim_embedding,)
        """
        news = storyline.news.to_list()
        news_embeddings = self.embedding_model(news)
        storyline_embedding = self.news_mean_pooling(news_embeddings)
        return storyline_embedding
