import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        # Fill the even indices with sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # Fill the odd indices with cos
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)

        # Register the buffer instead of a model parameter
        # so that it is not updated during backpropagation
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor
        x = x + (self.pe[:, : x.shape(1), :]).requires_grad_(
            False
        )  # (batch_size, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        # X shape: (batch, seq_len, hidden_size)
        # Keep the dimensions for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, h: int, dropout: float
    ) -> None:  # h is the number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) x (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = nn.Softmax(dim=-1)(
            attention_scores
        )  # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_probs = dropout(attention_probs)

        # (batch_size, h, seq_len, seq_len) x (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, d_k)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

    def forward(self, q, k, v, mask=None):
        query = self.w_q(
            q
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(
            k
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(
            v
        )  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        query = query.view

        # Split the query, key, and value into h different parts
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_probs = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], self.h * self.d_k)
        )

        # (batch_size, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        droupout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_1 = ResidualConnection(droupout)
        self.residual_connection_2 = ResidualConnection(droupout)

    def forward(self, x, mask):
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, mask)
        )
        x = self.residual_connection_2(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
