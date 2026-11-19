"""TensorFlow / Keras Image Captioning Model with Additive Attention.

Architecture:
- CNN Encoder: Pretrained CNN (ResNet-50 / MobileNetV2) with projection layer
- Bahdanau Attention: Additive alignment over spatial grid features
- RNN Decoder: GRU recurrent decoder with word embeddings and attention context
- Loss: Masked Sparse Categorical Crossentropy (ignores <pad> tokens)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

os.environ.setdefault("KERAS_HOME", os.path.expanduser("/tmp/.keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import config
from src.vocab import Vocabulary


class CNN_Encoder(tf.keras.Model):
    """Extracts spatial feature maps and projects to embedding dimension."""

    def __init__(self, embedding_dim: int = config.EMBED_DIM, fine_tune: bool = config.FINE_TUNE_ENCODER):
        super().__init__()
        # MobileNetV2 is lightweight and fast on CPU / Metal / GPU
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights=None, input_shape=(224, 224, 3)
        )
        base_model.trainable = fine_tune
        self.feature_extractor = base_model
        self.fc = tf.keras.layers.Dense(embedding_dim, activation="relu")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x shape: (B, 224, 224, 3)
        features = self.feature_extractor(x)  # (B, 7, 7, 1280)
        # Reshape to (B, 49, 1280)
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
        # Project to (B, 49, embedding_dim)
        projected = self.fc(features)
        return projected


class BahdanauAttentionTF(tf.keras.layers.Layer):
    """TensorFlow implementation of Bahdanau Additive Attention."""

    def __init__(self, units: int = config.ATTENTION_DIM):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features: tf.Tensor, hidden: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # features shape: (B, 49, embedding_dim)
        # hidden shape: (B, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (B, 1, hidden_size)

        # score shape: (B, 49, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape: (B, 49, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape: (B, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    """GRU Decoder with Attention and Word Embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        units: int,
        vocab_size: int,
    ):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttentionTF(units)
        self.dropout = tf.keras.layers.Dropout(config.DROPOUT)

    def call(
        self, x: tf.Tensor, features: tf.Tensor, hidden: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Context vector from attention
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding: (B, 1, embedding_dim)
        x = self.embedding(x)

        # Concatenate context vector with word embedding: (B, 1, embedding_dim + feature_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passing concatenated vector to the GRU
        output, state = self.gru(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.dropout(x)
        x = self.fc2(x)  # (B, vocab_size)

        return x, state, attention_weights

    def reset_state(self, batch_size: int) -> tf.Tensor:
        return tf.zeros((batch_size, self.units))


class TFCaptioner(tf.keras.Model):
    """End-to-End TensorFlow Captioning Framework Wrapper."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBED_DIM,
        units: int = config.DECODER_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units

        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        """Compute crossentropy loss masking out <pad> tokens."""
        mask = tf.math.logical_not(tf.math.equal(real, config.PAD_IDX))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(
        self,
        img_tensor: tf.Tensor,
        target: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> tf.Tensor:
        """Single training step with Teacher Forcing."""
        loss = 0.0
        batch_size = tf.shape(img_tensor)[0]
        hidden = self.decoder.reset_state(batch_size=batch_size)
        dec_input = tf.fill([batch_size, 1], config.START_IDX)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                # Teacher forcing: feed real target word
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss

    def predict_caption(
        self,
        image_rgb: np.ndarray,
        vocab: Vocabulary,
        max_len: int = config.MAX_CAPTION_LEN,
    ) -> Tuple[str, np.ndarray]:
        """Generate a caption for a single normalized RGB image (224, 224, 3)."""
        img_tensor = tf.convert_to_tensor(image_rgb[np.newaxis, ...], dtype=tf.float32)
        features = self.encoder(img_tensor)

        hidden = self.decoder.reset_state(batch_size=1)
        dec_input = tf.constant([[config.START_IDX]], dtype=tf.int32)

        result_tokens = []
        attention_plot = []

        for _ in range(max_len):
            predictions, hidden, attention_weights = self.decoder(
                dec_input, features, hidden
            )
            # attention_weights shape: (1, 49, 1) -> (49,)
            attention_plot.append(tf.squeeze(attention_weights).numpy())

            predicted_id = int(tf.argmax(predictions[0]).numpy())
            result_tokens.append(predicted_id)

            if predicted_id == config.END_IDX:
                break

            dec_input = tf.constant([[predicted_id]], dtype=tf.int32)

        caption = vocab.decode(result_tokens, remove_special_tokens=True)
        attn_matrix = np.array(attention_plot) if attention_plot else np.empty((0, 49))
        return caption, attn_matrix
