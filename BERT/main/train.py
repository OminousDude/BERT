import torch
from torch.utils.data import DataLoader

# Import my own classes
import util
import dataset
import attention
import embedding
import model
import trainer

if __name__ == "__main__":
    # Set the calculation precision to high instead of extra high which is the base
    # By doing this the model will approximate the value 
    # For example if the correct value is 0.378613298417239 it would predict something like 0.3786132982131234
    # This will trade a slight amount of precision for speed
    torch.set_float32_matmul_precision('high')

    util.random_seed(42)

    train_args = util.TrainArguments(batch_size=128, max_seq_len=128, vocab_size=30000, num_heads=12, emb_dim=768, num_layers=12)

    # Create dataset and dataloader
    data = dataset.BertDataset('/media/maxim/DataSets/BERT/BERT-DATA/', vocab_size=train_args.vocab_size, max_seq_len=train_args.max_seq_len)
    tokenizer = data.tokenizer
    loader = DataLoader(data, batch_size=train_args.batch_size, shuffle=True)

    # Print dataset vocab size to make sure that my dataset initialized correctly
    util.print(len(data.vocab), "Dataset Vocab Length: ", "\n")

    # Print the first sentence in my dataset decoded
    util.print(tokenizer.decode(next(iter(loader))[0][0]), "First sentence data: ", "\n")
    # Print the first label sentence in my dataset
    util.print(next(iter(loader))[1][0], "First label sentence data: ", "\n")

    # Set vocab reverse so that I can input the id and get word
    vocab = dict((v,k) for k,v in tokenizer.get_vocab().items())

    # "max_seq_len" and "pos_enc_len" must be same
    # "feed_forward_dim" should be "emb_dim" * 4
    # "pos_enc_len" and "feed_forward_layer" are automatically following the previous directions and passing them is not encouraged
    

    bert = model.Bert(train_args, 0.1)

    model = model.BertLM(bert)
    # LM = torch.load('saves/BERT_time: 12|04|2024 11:54:13|step: 15000.pt')

    trainer = trainer.BertTrainer(model, vocab=vocab, device="cuda")
    trainer.train(loader, 5)