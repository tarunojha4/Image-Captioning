"""PyTorch Image Captioning Model with Bahdanau Spatial Attention.

Architecture:
- Encoder: Pretrained CNN (ResNet-50) extracting spatial feature map (B, 49, 2048)
- Attention: Bahdanau Additive Attention over 7x7 spatial grid locations
- Decoder: LSTM recurrent network with context concatenation and Teacher Forcing
- Decoding: Greedy search & Beam search with attention weight extraction for OpenCV heatmaps
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

import config
from src.vocab import Vocabulary


class EncoderCNN(nn.Module):
    """Convolutional Neural Network feature extractor with spatial feature preservation."""

    def __init__(
        self,
        encoded_image_size: int = 7,
        fine_tune: bool = False,
        feature_dim: int = config.ENCODER_DIM,
    ):
        super().__init__()
        self.encoded_image_size = encoded_image_size

        # Use ResNet-50 architecture
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            # Fallback if offline/sandboxed without pre-downloaded weights
            resnet = models.resnet50(weights=None)

        # Remove the classification head and final pooling layer
        # Keeps conv layers up to layer4 -> shape (B, 2048, H/32, W/32)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pool to ensure fixed spatial grid size (7x7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial grid features from images.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            Tensor of shape (B, 49, 2048).
        """
        features = self.resnet(images)  # (B, 2048, 7, 7)
        features = self.adaptive_pool(features)
        # Reshape to (B, 2048, 49) then permute to (B, 49, 2048)
        batch_size = features.size(0)
        channels = features.size(1)
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)
        return features

    def fine_tune(self, fine_tune: bool = True) -> None:
        """Enable or freeze convolutional backbone parameters."""
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only unfreeze upper conv layers (layer4)
        if fine_tune:
            for c in list(self.resnet.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = True


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) Spatial Attention mechanism."""

    def __init__(
        self,
        encoder_dim: int = config.ENCODER_DIM,
        decoder_dim: int = config.DECODER_DIM,
        attention_dim: int = config.ATTENTION_DIM,
    ):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, encoder_features: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Args:
            encoder_features: (B, num_pixels=49, encoder_dim)
            decoder_hidden: (B, decoder_dim)

        Returns:
            context: (B, encoder_dim)
            alpha: (B, num_pixels=49)
        """
        att1 = self.encoder_att(encoder_features)       # (B, 49, att_dim)
        att2 = self.decoder_att(decoder_hidden)         # (B, att_dim)
        att2 = att2.unsqueeze(1)                        # (B, 1, att_dim)
        scores = self.full_att(self.relu(att1 + att2))  # (B, 49, 1)
        scores = scores.squeeze(2)                      # (B, 49)
        alpha = self.softmax(scores)                    # (B, 49)

        # Context vector is weighted sum of encoder features
        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


class DecoderRNN(nn.Module):
    """LSTM Decoder with Bahdanau Attention and Teacher Forcing."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        decoder_dim: int = config.DECODER_DIM,
        encoder_dim: int = config.ENCODER_DIM,
        attention_dim: int = config.ATTENTION_DIM,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        # Initialize hidden and cell states from mean encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # LSTMCell receives embedded word concatenated with attention context
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden and cell state from average spatial features."""
        mean_features = encoder_features.mean(dim=1)  # (B, encoder_dim)
        h = torch.tanh(self.init_h(mean_features))     # (B, decoder_dim)
        c = torch.tanh(self.init_c(mean_features))     # (B, decoder_dim)
        return h, c

    def forward(
        self,
        encoder_features: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with Teacher Forcing for training.

        Args:
            encoder_features: (B, 49, encoder_dim)
            captions: (B, max_len)
            lengths: (B,) sequence lengths

        Returns:
            predictions: (B, max_decode_len, vocab_size)
            alphas: (B, max_decode_len, 49)
        """
        batch_size = encoder_features.size(0)
        # Decode length is max caption length - 1 (predicting t+1 given t)
        decode_len = captions.size(1) - 1

        embeddings = self.embedding(captions)  # (B, max_len, embed_dim)
        h, c = self.init_hidden_state(encoder_features)

        predictions = torch.zeros(batch_size, decode_len, self.vocab_size, device=encoder_features.device)
        alphas = torch.zeros(batch_size, decode_len, encoder_features.size(1), device=encoder_features.device)

        for t in range(decode_len):
            context, alpha = self.attention(encoder_features, h)
            # Input to LSTMCell: [word_embed_t, context_t]
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fc_out(self.dropout(h))

            predictions[:, t, :] = output
            alphas[:, t, :] = alpha

        return predictions, alphas

    def generate_caption_greedy(
        self,
        encoder_features: torch.Tensor,
        vocab: Vocabulary,
        max_len: int = config.MAX_CAPTION_LEN,
    ) -> Tuple[List[int], torch.Tensor]:
        """Greedy decoding step-by-step.

        Returns:
            token_ids: List of generated token IDs.
            attention_weights: Tensor of shape (T, 49).
        """
        h, c = self.init_hidden_state(encoder_features)
        current_token = torch.tensor([config.START_IDX], device=encoder_features.device)

        token_ids: List[int] = []
        alphas_list: List[torch.Tensor] = []

        for _ in range(max_len):
            embed = self.embedding(current_token)  # (1, embed_dim)
            context, alpha = self.attention(encoder_features, h)
            lstm_input = torch.cat([embed, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.fc_out(h)  # (1, vocab_size)

            predicted_id = logits.argmax(dim=1).item()
            token_ids.append(predicted_id)
            alphas_list.append(alpha.squeeze(0).cpu())

            if predicted_id == config.END_IDX:
                break

            current_token = torch.tensor([predicted_id], device=encoder_features.device)

        attention_weights = torch.stack(alphas_list, dim=0) if alphas_list else torch.empty(0)
        return token_ids, attention_weights

    def generate_caption_beam_search(
        self,
        encoder_features: torch.Tensor,
        vocab: Vocabulary,
        beam_width: int = 3,
        max_len: int = config.MAX_CAPTION_LEN,
    ) -> Tuple[List[int], torch.Tensor, float]:
        """Beam search decoding for higher caption quality and fluency."""
        k = beam_width
        device = encoder_features.device

        # Expand encoder features for beam width
        encoder_features = encoder_features.expand(k, -1, -1)  # (k, 49, enc_dim)
        h, c = self.init_hidden_state(encoder_features)

        # Initial beam: [([tokens], score, h, c, [alphas])]
        beams = [([config.START_IDX], 0.0, h[0:1], c[0:1], [])]
        completed_beams = []

        for _ in range(max_len):
            candidates = []
            for tokens, score, curr_h, curr_c, alphas in beams:
                last_token = torch.tensor([tokens[-1]], device=device)
                embed = self.embedding(last_token)
                context, alpha = self.attention(encoder_features[0:1], curr_h)
                lstm_input = torch.cat([embed, context], dim=1)
                next_h, next_c = self.lstm_cell(lstm_input, (curr_h, curr_c))
                log_probs = torch.log_softmax(self.fc_out(next_h), dim=1).squeeze(0)

                topk_log_probs, topk_indices = log_probs.topk(k)
                for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_tokens = tokens + [idx]
                    new_score = score + log_prob
                    new_alphas = alphas + [alpha.squeeze(0).cpu()]
                    candidates.append((new_tokens, new_score, next_h, next_c, new_alphas))

            # Select top-k candidates
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)
            beams = []
            for cand in candidates[:k]:
                if cand[0][-1] == config.END_IDX:
                    completed_beams.append(cand)
                else:
                    beams.append(cand)

            if len(completed_beams) >= k:
                break

        if not completed_beams:
            completed_beams = beams

        # Best beam sorted by length-normalized score
        completed_beams.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)
        best_tokens, best_score, _, _, best_alphas = completed_beams[0]

        # Exclude <start> token
        token_ids = best_tokens[1:]
        attn_tensor = torch.stack(best_alphas, dim=0) if best_alphas else torch.empty(0)
        return token_ids, attn_tensor, best_score


class PyTorchCaptioner(nn.Module):
    """Full End-to-End Image Captioning Model (Encoder + Decoder + Attention)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        decoder_dim: int = config.DECODER_DIM,
        encoder_dim: int = config.ENCODER_DIM,
        attention_dim: int = config.ATTENTION_DIM,
        fine_tune_encoder: bool = config.FINE_TUNE_ENCODER,
    ):
        super().__init__()
        self.encoder = EncoderCNN(fine_tune=fine_tune_encoder, feature_dim=encoder_dim)
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            encoder_dim=encoder_dim,
            attention_dim=attention_dim,
        )

    def forward(
        self, images: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        predictions, alphas = self.decoder(features, captions, lengths)
        return predictions, alphas

    def generate(
        self,
        image_tensor: torch.Tensor,
        vocab: Vocabulary,
        method: str = "beam",
        beam_width: int = 3,
        max_len: int = config.MAX_CAPTION_LEN,
    ) -> Tuple[str, torch.Tensor, Optional[float]]:
        """Generate caption from an image tensor (1, 3, H, W)."""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image_tensor)
            if method == "beam":
                token_ids, alphas, score = self.decoder.generate_caption_beam_search(
                    features, vocab, beam_width=beam_width, max_len=max_len
                )
            else:
                token_ids, alphas = self.decoder.generate_caption_greedy(
                    features, vocab, max_len=max_len
                )
                score = None

        caption = vocab.decode(token_ids, remove_special_tokens=True)
        return caption, alphas, score
