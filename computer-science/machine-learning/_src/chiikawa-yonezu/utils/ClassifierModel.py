import torch
from transformers import BertModel


class ClassifierModel(torch.nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.linear = torch.nn.Linear(768, 2)  # BERTの隠れ層の次元数と出力クラス数

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        return self.linear(pooled_output)
