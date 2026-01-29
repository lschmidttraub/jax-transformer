import jax.numpy as jnp
from jax.nn import one_hot, softmax, gelu, one_hot
from flax import linen as nn
from typing import Any, Sequence


class Attention(nn.Module):
    n_heads:int
    head_dim:int 
    dim:int

    @nn.compact 
    def __call__(self, x):
        q = nn.Dense(self.head_dim*self.n_heads, use_bias=False)(x)
        k = nn.Dense(self.head_dim*self.n_heads, use_bias=False)(x)
        v = nn.Dense(self.head_dim*self.n_heads, use_bias=False)(x)
        q, k, v = [y.reshape((*y.shape[:-1], self.n_heads, self.head_dim)) for y in [q,k,v]]
        dot = jnp.einsum("bihj bkhj -> bhik", q, k)
        w = softmax(dot/jnp.sqrt(self.head_dim))
        a = jnp.einsum("bhij bjhk -> bihk", w, )
        a = a.reshape((*a.shape[:-2], -1))
        return nn.Dense(x.shape[-1])(a)

class FFN(nn.Module):
    hidden_dim: int 

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = gelu(x)
        x = nn.Dense(x.shape[-1])(x)
        return x


class Transformer(nn.Module):
    layers: int 
    emb_dim: int 
    n_heads: int 
    head_dim: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.emb_dim)(x)
        for _ in range(self.layers):
            xnorm = nn.LayerNorm()(x)
            x = x + Attention(self.n_heads, self.head_dim)(xnorm)
            x = x+FFN(self.ffn_dim)(x)
        return x
