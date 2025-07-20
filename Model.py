# class PositionalEncoding(layers.Layer):
#     def __init__(self, num_patches, dim):
#         super().__init__()
#         self.pos_emb = self.add_weight("pos_embedding", shape=[1, num_patches, dim], initializer='random_normal')

#     def call(self, x):
#         return x + self.pos_emb
# or
class PositionalEncoding(layers.Layer):
    """
    Adds learnable or sinusoidal positional encodings to patch tokens.
    """
    def __init__(self, num_patches, dim, use_sinusoidal=False, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.dim = dim
        self.use_sinusoidal = use_sinusoidal

        if self.use_sinusoidal:
            self.pos_encoding = self._get_sinusoidal_encoding(num_patches, dim)
        else:
            self.pos_emb = self.add_weight(
                name="learnable_pos_embedding",
                shape=(1, num_patches, dim),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True
            )

        self.gamma = self.add_weight(
            name="positional_scale",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

    def _get_sinusoidal_encoding(self, num_positions, dim):
        position = np.arange(num_positions)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))

        pe = np.zeros((num_positions, dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        pe = pe[np.newaxis, ...]  # Shape: (1, num_positions, dim)
        return tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        if self.use_sinusoidal:
            pos_embedding = self.pos_encoding
        else:
            pos_embedding = self.pos_emb

        return x + self.gamma * pos_embedding

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'dim': self.dim,
            'use_sinusoidal': self.use_sinusoidal
        })
        return config
#####################################################
def conv_stem(x):
    x = layers.Conv2D(64, 7, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

#####################################################

# def patch_embedding(x, patch_size=2, embed_dim=192):
#     x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(x)
#     B, H, W, C = x.shape
#     x = tf.reshape(x, [-1, H * W, C])
#     return x, H, W
# or
class PatchEmbedding(layers.Layer):
    """
    Advanced Patch Embedding Layer: Converts image/feature map to sequence of patch tokens using Conv2D.
    """
    def __init__(self, patch_size=16, embed_dim=768, flatten=True, norm_layer=True, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.norm_layer = norm_layer

        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            kernel_initializer='he_normal'
        )

        if self.norm_layer:
            self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)  # (B, H/P, W/P, C)

        if self.flatten:
            B = tf.shape(x)[0]
            H = tf.shape(x)[1]
            W = tf.shape(x)[2]
            C = tf.shape(x)[3]
            x = tf.reshape(x, [B, H * W, C])  # → (B, num_patches, embed_dim)

        if self.norm_layer:
            x = self.norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "flatten": self.flatten,
            "norm_layer": self.norm_layer
        })
        return config

###############################################
def transformer_block(x, num_heads, mlp_dim, dropout=0.1):
    # Self Attention
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x1, x1)
    x = layers.Add()([x, attn_output])

    # MLP Block
    x2 = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x2)
    mlp_output = layers.Dropout(dropout)(mlp_output)
    mlp_output = layers.Dense(x.shape[-1])(mlp_output)
    x = layers.Add()([x, mlp_output])
    return x
############################################################
def build_KDDS(input_shape=(224, 224, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape)

    # -------- Stage 1: CNN Stem --------
    x = conv_stem(inputs)  # (56x56x128)

    # -------- Stage 2: Patch Embedding --------
    patch_embed = PatchEmbedding(patch_size=2, embed_dim=192, flatten=True, norm_layer=True)
    x = patch_embed(x)  # (B, 784, 192)

    # -------- Positional Encoding --------
    pos_enc = PositionalEncoding(num_patches=784, dim=192)
    x = pos_enc(x)

    # -------- Transformer Blocks --------
    for _ in range(3):
        x = transformer_block(x, num_heads=4, mlp_dim=384)

    # -------- Stage 3: Downsample --------
    x = layers.Reshape((28, 28, 192))(x)
    x = layers.Conv2D(384, kernel_size=3, strides=2, padding='same')(x)  # → (14x14x384)
    x = layers.Reshape((-1, 384))(x)  # (B, 196, 384)

    # -------- Positional Encoding 2 --------
    x = PositionalEncoding(num_patches=196, dim=384)(x)

    # -------- Deeper Transformer Blocks --------
    for _ in range(2):
        x = transformer_block(x, num_heads=6, mlp_dim=768)

    # -------- Classification Head --------
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)
