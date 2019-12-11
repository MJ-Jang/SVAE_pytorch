import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable


class SVAE(nn.Module):
    def __init__(self, vocab: int, hidden_dim: int, latent_dim: int):
        super(SVAE, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.word_embeddings = nn.Embedding(vocab, hidden_dim)
        self.enc_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1)
        self.dec_gru = nn.GRU(hidden_dim, latent_dim, num_layers=1)

        self.softmax = nn.Softmax(dim=-1)

        self.mu_enc = nn.Linear(hidden_dim * 2, latent_dim)
        self.log_sigma_enc = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc = nn.Linear(latent_dim, vocab)

    def forward(self,
               enc_input_ids: torch.LongTensor,
               enc_length: torch.LongTensor,
               dec_input_ids: torch.LongTensor,
               dec_length: torch.LongTensor,
               batch_size: int = None):

        h_enc = self.encode(enc_input_ids, enc_length)
        z, mu, sigma = self._sample_latent(h_enc)
        logits, dec_max_len = self.decode(z, dec_input_ids, dec_length)
        return logits, dec_max_len, mu, sigma

    def encode(self,
               input_ids: torch.LongTensor,
               length: torch.LongTensor):
        input_vectors = self.word_embeddings(input_ids)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_vectors,
                                                               length.tolist(),
                                                               batch_first=True,
                                                               enforce_sorted=False)

        h_0 = Variable(torch.zeros(1, input_vectors.size(0), self.hidden_dim)).to(input_vectors.device)

        output, h = self.enc_gru(packed_input, h_0)
        output, self.output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_state = torch.stack([o[l-1] for o,l in zip(output, self.output_lengths)])

        weights = [self._return_weight(o, l) for o, l in zip(output, self.output_lengths)]
        weights = torch.stack(weights)

        semantic_vector = (weights.unsqueeze(2) * output).sum(dim=1)
        final_state = torch.cat([last_state, semantic_vector], dim=1)
        return final_state

    def decode(self,
               z,
               input_ids: torch.LongTensor,
               length: torch.LongTensor):

        input_vectors = self.word_embeddings(input_ids)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_vectors,
                                                               length.tolist(),
                                                               batch_first=True,
                                                               enforce_sorted=False)
        output, _ = self.dec_gru(packed_input, z.unsqueeze(0))
        output, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output, int(max(out_lengths))

    def _return_weight(self, out, length):
        alpha_fw = torch.cat([self.softmax(torch.mul(out[:length], out[length - 1]).sum(1)),
                              torch.FloatTensor([0] * int((max(self.output_lengths) - length)))])

        alpha_bw = torch.cat([self.softmax(torch.mul(out[:length], out[0]).sum(1)),
                              torch.FloatTensor([0] * int((max(self.output_lengths) - length)))])
        alpha = (alpha_fw + alpha_bw) / 2
        return alpha

    def _sample_latent(self, h_enc):
        mu = self.mu_enc(h_enc)
        log_sigma = self.log_sigma_enc(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        latent = mu + sigma * Variable(std_z, requires_grad=False)
        return latent, mu, sigma


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
