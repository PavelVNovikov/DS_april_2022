import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', )

text = "Who was Jim Henson? Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)  # tensor len = words number


bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased').eval()

with torch.no_grad():
    embeddings = bert_model(tokenized_text)  # tensor len = [words number, 768]

n_classes = 20
linear = nn.Linear(768, n_classes)

predict = linear(embeddings)


Who was Jim Henson?
    \/
 0 1 2 3
    \/
[0...] [1...] [2...] [3...]
    \/
query = [0...]
key = [1...] [2...] [3...]
values = [1...] [2...] [3...]

attn(key, q) -> [0.1, 0.45, 0.45]

