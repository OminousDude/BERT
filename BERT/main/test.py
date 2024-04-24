import util
import dataset
import attention
import embedding
import model
import trainer

import torch
from transformers import BertTokenizerFast, AutoTokenizer

tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

model = torch.load('saves/BERT_time: 11|04|2024 21:50:17|step: 4000.pt')

vocab = dict((v,k) for k,v in tokenizer.get_vocab().items())

trainer = trainer.BertTrainer(model, vocab=vocab, device="cuda")

text = "I [MASK] my horse!"

trainer.test(text, tokenizer, tokenizer[0])