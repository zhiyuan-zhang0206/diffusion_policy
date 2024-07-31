from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

class DistilBERTWrapper:
    def __init__(self, device:str=None):
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device).eval()

    @torch.no_grad()
    def __call__(self, texts: List[str]):
        tokenized = self.tokenizer(texts, return_tensors='pt', padding=True)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        embedding = self.distilbert(input_ids, attention_mask).last_hidden_state
        embedding = embedding.mean(1)
        return embedding