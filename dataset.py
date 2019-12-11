import numpy as np

from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    def __init__(self, data, tok, max_len):
        self.tok = tok
        self.max_len = max_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # src
        src = self.data[idx]
        src_id, len_src = self._tokenize(self.tok,
                                         self.max_len,
                                         src,
                                         is_src=True)

        # tgt
        tgt_enc_id, tgt_enc_len = self._tokenize(self.tok,
                                                     self.max_len,
                                                     src, is_src=False,
                                                     is_input=True)

        tgt_dec_id, tgt_dec_len = self._tokenize(self.tok,
                                                 self.max_len,
                                                 src, is_src=False,
                                                 is_input=False)

        return np.array(src_id), np.array(len_src), \
               np.array(tgt_enc_id), np.array(tgt_enc_len),\
               np.array(tgt_dec_id), np.array(tgt_dec_len)

    def _tokenize(self, tokenizer, max_len, sent, is_src=True, is_input=True):
        tokens = tokenizer.encode_as_ids(sent)
        token_len = len(tokens)

        if is_src:
            if len(tokens) < max_len:
                tokens = tokens + [tokenizer.piece_to_id('<pad>')] * (max_len - len(tokens))
                token_len = token_len
            elif len(tokens) == 0:
                tokens += [0]
                token_len = 1
            else:
                tokens = tokens[:max_len]
                token_len = max_len

        elif not is_src and is_input:
            tokens = [tokenizer.piece_to_id('<s>')] + tokens
            if len(tokens) < max_len+1:
                tokens = tokens + [tokenizer.piece_to_id('<pad>')] * (max_len+1 - len(tokens))
                token_len = token_len + 1
            elif len(tokens) == 0:
                tokens += [0]
                token_len = 1
            else:
                tokens = tokens[:max_len+1]
                token_len = token_len + 1

        elif not is_src and not is_input:
            tokens = tokens + [tokenizer.piece_to_id('</s>')]
            if len(tokens) < max_len+1:
                tokens = tokens + [tokenizer.piece_to_id('<pad>')] * (max_len+1 - len(tokens))
                token_len = token_len + 1
            elif len(tokens) == 0:
                tokens += [0]
                token_len = 1
            else:
                tokens = tokens[:max_len+1]
                token_len = token_len + 1
        return tokens, token_len
