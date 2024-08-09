import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
import math

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)



def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_model,n_layers,n_heads,d_k,d_v,d_ff):
        ##  [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##[batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


##  MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_layers,n_heads,d_k,d_v,d_ff):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, device='cuda')
        self.W_K = nn.Linear(d_model, d_k * n_heads, device='cuda')
        self.W_V = nn.Linear(d_model, d_v * n_heads,device='cuda')
        self.linear = nn.Linear(n_heads * d_v, d_model,device='cuda')
        self.layer_norm = nn.LayerNorm(d_model,device='cuda')
        # self.W_Q = nn.Linear(d_model, d_k * n_heads)
        # self.W_K = nn.Linear(d_model, d_k * n_heads)
        # self.W_V = nn.Linear(d_model, d_v * n_heads)
        # self.linear = nn.Linear(n_heads * d_v, d_model)
        # self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask,d_model,n_layers,n_heads,d_k,d_v,d_ff):


        ## Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]


        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)



        ##context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask,d_model,n_layers,n_heads,d_k,d_v,d_ff)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


##PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,n_layers,n_heads,d_k,d_v,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1,device='cuda')
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1,device='cuda')
        self.layer_norm = nn.LayerNorm(d_model,device='cuda')
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


## PositionalEncoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        device = torch.device('cuda:0')
        data = (self.pe[:x.size(0), :]).to(device)
        x = x + data
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_layers,n_heads,d_k,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,n_layers,n_heads,d_k,d_v,d_ff)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,n_layers,n_heads,d_k,d_v,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask,d_model,n_layers,n_heads,d_k,d_v,d_ff):

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask,d_model,n_layers,n_heads,d_k,d_v,d_ff) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self,d_model,n_layers,n_heads,d_k,d_v,d_ff):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model) 
        self.pos_emb = PositionalEncoding(d_model) 
        self.layers = nn.ModuleList([EncoderLayer(d_model,n_layers,n_heads,d_k,d_v,d_ff) for _ in range(n_layers)]) 

    def forward(self, enc_inputs,word_seq_tensor,d_model,n_layers,n_heads,d_k,d_v,d_ff):



        # enc_outputs = self.src_emb(enc_inputs)

        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)

        enc_self_attn_mask = get_attn_pad_mask(word_seq_tensor, word_seq_tensor)
        enc_self_attns = []  

        enc_outputs_list = [] 

        for layer in self.layers:
            ## EncoderLayer
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask,d_model,n_layers,n_heads,d_k,d_v,d_ff)
            enc_self_attns.append(enc_self_attn)
            enc_outputs_list.append(enc_outputs)

        enc_self_attns = torch.stack(enc_self_attns[-3:], dim=1)
        enc_self_attn = enc_self_attns.mean(dim=1)

        enc_outputs_list = torch.stack(enc_outputs_list[-3:], dim=1)
        enc_outputs_list = enc_outputs_list.mean(dim=1)
        return enc_outputs, enc_self_attn, enc_outputs_list


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]

        ## get_attn_pad_mask 
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        ## get_attn_subsequent_mask 
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns




class Transformer(nn.Module):

    def __init__(self,d_model):
        super(Transformer, self).__init__()
        self.d_model = d_model  # Embedding Size
        self.d_ff = 2048  # FeedForward dimension
        self.d_k = self.d_v = 64  # dimension of K(=Q), V
        self.n_layers = 3  # number of Encoder of Decoder Layer
        self.n_heads = 12  # number of heads in Multi-Head Attention
        self.encoder = Encoder(self.d_model,self.n_layers,self.n_heads,self.d_k,self.d_v,self.d_ff)

    def forward(self,inputs,word_seq_tensor):
        enc_outputs, enc_self_attns, enc_outputs_list = self.encoder(inputs,word_seq_tensor,self.d_model,self.n_layers,self.n_heads,self.d_k,self.d_v,self.d_ff)
        # print(enc_outputs.size())
        # enc_outputs_list 
        return enc_outputs, enc_self_attns, enc_outputs_list



if __name__ == '__main__':


    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)
    src_len = 5 # length of source
    tgt_len = 5 # length of target


    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    print(enc_inputs)
    print(enc_inputs.size())
    for epoch in range(1):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()



