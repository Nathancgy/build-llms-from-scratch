import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Model Configuration Parameters
# =============================================================================

# Default values for the GPT model
DEFAULT_VOCAB_SIZE = 5000    # Vocabulary size
DEFAULT_D_MODEL = 256        # Embedding dimension
DEFAULT_NUM_HEADS = 4        # Number of attention heads
DEFAULT_D_FF = 512           # Feed-forward network dimension
DEFAULT_NUM_LAYERS = 4       # Number of transformer layers
DEFAULT_DROPOUT = 0.1        # Dropout rate
DEFAULT_MAX_LEN = 1024       # Maximum sequence length
DEFAULT_PAD_IDX = 0          # Padding token index

# Special tokens by default
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']

# =============================================================================
# Model Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module that allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        # (batch_size, num_heads, query_len, d_k) @ (batch_size, num_heads, d_k, key_len) = (batch_size, num_heads, query_len, key_len)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        # Apply mask for padding or causal attention
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        # (batch_size, num_heads, query_len, key_len) @ (batch_size, num_heads, value_len, d_k) = (batch_size, num_heads, query_len, d_k)
        x = torch.matmul(attention, V)
        
        # Reshape back to original dimensions
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_linear(x)
        
        return output, attention


class PositionwiseFeedforward(nn.Module):
    """
    Position-wise feed-forward network, applied to each position separately and identically.
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # First linear layer with ReLU activation
        x = self.dropout(F.relu(self.fc1(x)))
        # Second linear layer
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization module that normalizes its inputs across the features dimension.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        # Calculate mean and standard deviation
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize and apply scale and shift
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class DecoderLayer(nn.Module):
    """
    Decoder layer for language modeling that consists of multi-head self-attention
    and a position-wise feed-forward network with residual connections and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block with residual connection and layer norm
        _x, attention = self.self_attention(x, x, x, mask)
        x = x + self.dropout(_x)
        x = self.norm1(x)
        
        # Feed-forward block with residual connection and layer norm
        _x = self.feed_forward(x)
        x = x + self.dropout(_x)
        x = self.norm2(x)
        
        return x, attention


class PositionalEncoding(nn.Module):
    """
    Positional encoding that adds position information to the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

# =============================================================================
# Main GPT Model
# =============================================================================

class GPT(nn.Module):
    """
    A simplified GPT-style transformer for next token prediction.
    """
    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE, 
                 d_model=DEFAULT_D_MODEL, 
                 num_heads=DEFAULT_NUM_HEADS, 
                 d_ff=DEFAULT_D_FF, 
                 num_layers=DEFAULT_NUM_LAYERS, 
                 dropout=DEFAULT_DROPOUT, 
                 max_len=DEFAULT_MAX_LEN, 
                 pad_idx=DEFAULT_PAD_IDX):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        
    def make_causal_mask(self, x):
        """
        Create a causal mask to prevent attention to future tokens.
        """
        # Get sequence length
        seq_len = x.shape[1]
        
        # Create a lower triangular matrix (including the diagonal)
        # This ensures each position can only attend to previous positions
        mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
        
        # Create padding mask where pad tokens cannot be attended to
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Combine causal and padding masks
        mask = mask.to(x.device)
        mask = mask & pad_mask
        
        return mask
    
    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor of shape [batch_size, seq_len]
        Returns:
            logits: Predicted next token logits of shape [batch_size, seq_len, vocab_size]
            attentions: List of attention weights from each decoder layer
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Create causal mask
        mask = self.make_causal_mask(x)
        
        # Create embedding and add positional encoding
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Apply decoder layers
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, mask)
            attentions.append(attention)
        
        # Final linear layer
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, vocab_size]
        logits = self.fc_out(x)
        
        return logits, attentions
    
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input tensor of shape [batch_size, prompt_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: If set, only sample from the top k most likely tokens
            
        Returns:
            generated: Generated sequence including the prompt
        """
        self.eval()
        with torch.no_grad():
            # Start with the prompt
            x = prompt.clone()
            
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                logits, _ = self.forward(x)
                
                # Get the next token predictions (last token in sequence)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    # Get the top-k values and indices
                    values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                    # Create a mask of the top-k positions
                    mask = torch.zeros_like(next_token_logits).scatter_(1, indices, 1.0)
                    # Apply the mask (set non-top-k values to -inf)
                    next_token_logits = torch.where(mask > 0, next_token_logits, torch.tensor(-float('inf')).to(next_token_logits.device))
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the new token to the sequence
                x = torch.cat([x, next_token], dim=1)
                
            return x 