import torch
import sentencepiece as sp
import argparse

from torch.utils.data import DataLoader

from model import SVAE
from dataset import AutoencoderDataset
from loss import sequence_cross_entropy_with_logits, latent_loss

parser = argparse.ArgumentParser()

parser.add_argument('-hidden_dim',
                    type=int,
                    default=128,
                    help="model hidden dimension")

parser.add_argument('-latent_dim',
                    type=int,
                    default=128,
                    help="model latent dimension")

parser.add_argument('-max_len',
                    type=int,
                    default=64,
                    help="Limited length for text")

# Training option
parser.add_argument('-lr',
                    type=float,
                    default=1e-4,
                    help="Learning rate")

parser.add_argument('-label_smoothing',
                    type=float,
                    default=0.2,
                    help="label smoothing ratio")

parser.add_argument('-gpu',
                    type=bool,
                    default=True,
                    help="use cuda or not")

parser.add_argument('-batch_size',
                    type=int,
                    default=512,
                    help="Batch size")

parser.add_argument('-epoch',
                    type=int,
                    default=100,
                    help="Number of epoch")

# Path option
parser.add_argument('-restore', type=str, default=None, help="Restoring model path")
parser.add_argument('-save', type=str, default='', help="saving model path")
parser.add_argument('-tok', type=str, default='', help="sentencepiece tokenizer path")
parser.add_argument('-tr_data', type=str, default='', help="training path")
parser.add_argument('-dev_data', type=str, default='', help="dev data path")

args = parser.parse_args()


class TrainOperator:
    def __init__(self):
        # source
        self.tok = sp.SentencePieceProcessor()
        self.tok.load(args.tok)
        self.vocab = self.tok.GetPieceSize()
        self.pad_id = self.tok.piece_to_id('<pad>')
        self.cuda = args.gpu and torch.cuda.is_available()

        # for data parallel
        if self.cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
        self.n_gpu=1

        # load loader
        self.train_loader = self._construct_loader('train')
        self.dev_loader = self._construct_loader('dev')
        print('* Train Operator is loaded')

    def setup_train(self, model_path=None):
        self.model = SVAE(vocab=self.vocab,
                          hidden_dim=args.hidden_dim,
                          latent_dim=args.latent_dim)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))

        # Data Parallel
        if self.cuda:
            if self.n_gpu == 1:
                pass
            elif self.n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        print("* Model setup is finished")

    def train(self):
        printInterval = 100
        init_loss = 1e5

        for n in range(args.epoch):
            for batch_id, batch in enumerate(self.train_loader):
                loss_tr = self._calculate_loss(batch)

                if (batch_id + 1) % printInterval == 0 or batch_id == 0:
                    loss_eval = self._evaluate()
                    loss_tr = round(loss_tr, 4)
                    print("|Epoch: {} | batch: {}/{}| tr_loss: {} | val_loss: {} |".format(n + 1,
                                                                                           batch_id + 1,
                                                                                           len(self.train_loader),
                                                                                           loss_tr,
                                                                                           loss_eval
                                                                                           ))
                    if loss_eval < init_loss:
                        init_loss = loss_eval
                        if self.n_gpu <= 1:
                            torch.save(self.model.state_dict(), args.save)
                        elif self.n_gpu > 1:
                            torch.save(self.model.module.state_dict(), args.save)

    def _construct_loader(self, type):
        if type == 'train':
            with open(args.tr_data, 'r', encoding='utf-8') as file:
                data = file.read()
            data = data.split("\n")

        elif type == 'dev':
            with open(args.dev_data, 'r', encoding='utf-8') as file:
                data = file.read()
            data = data.split("\n")
        else:
            raise (ValueError('type should be "train" or "dev"'))

        data = data[:-1]
        dataset = AutoencoderDataset(data, self.tok, args.max_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, )
        return loader

    def _calculate_loss(self, batch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        src_id, len_src, tgt_enc_id, tgt_enc_len, tgt_dec_id, tgt_dec_len = batch
        if self.cuda:
            src_id = src_id.cuda()
            len_src = len_src.cuda()
            tgt_enc_id = tgt_enc_id.cuda()
            tgt_enc_len = tgt_enc_len.cuda()
            tgt_dec_id = tgt_dec_id.cuda()
            tgt_dec_len = tgt_dec_len.cuda()

        logits, dec_max_len, mu, sigma = self.model(src_id, len_src, tgt_enc_id, tgt_enc_len)

        targets = tgt_dec_id[:, :dec_max_len]
        seq_mask = (targets != self.pad_id).float()

        CE = sequence_cross_entropy_with_logits(logits=logits,
                                                targets=targets,
                                                weights=seq_mask,
                                                average='token',
                                                label_smoothing=args.label_smoothing)
        KLD = latent_loss(mu, sigma)
        loss = CE + KLD

        if is_train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.tolist()

    def _evaluate(self):
        loss_eval = 0
        batch_id = 0
        for batch_id, batch in enumerate(self.dev_loader):
            loss = self._calculate_loss(batch, is_train=False)
            loss_eval += loss
        loss_eval = (loss_eval / (batch_id + 1))
        return round(loss_eval, 4)


if __name__ == '__main__':
    trainer = TrainOperator()
    trainer.setup_train(model_path=args.restore)
    trainer.train()