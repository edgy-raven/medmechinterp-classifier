import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

OUTPUT_FILES = [
    "feature_nn_metrics.json",
    "feature_nn.keras",
    "feature_xgb_predictions.csv",
]
SEED = 7
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 2e-4
DROPOUT_RATE = 0.4
L2_REG = 1e-4
MAX_SEQ_LEN = 256
SEQ_EMBED_DIM = 128
SEQ_PROJ_DIMS = (128,)
ATTN_HEADS = 8
ATTN_KEY_DIM = 16
ATTN_FF_DIM = 256
ATTN_LAYERS = 2
ATTN_DROPOUT = 0.2
HIDDEN_DIMS = (1024, 1024, 512, 512, 256, 128)
LABEL_SMOOTHING = 0.05
SEQ_SPATIAL_DROPOUT = 0.2
TABULAR_NOISE = 0.02
STOCHASTIC_DEPTH_RATE = 0.0


def inputs(context):
    feature_path = context.get("combined_features_path")
    if not feature_path:
        raise FileNotFoundError("features output missing; run with --jobs features first.")
    annotator_path = context.get("annotator_path")
    if not annotator_path or not annotator_path.exists():
        raise FileNotFoundError("annotator output missing; run with --jobs annotator first.")
    return {
        "feature_path": feature_path,
        "annotator_path": annotator_path,
        "output_dir": context["model_root"],
    }


def outputs(context):
    return {
        "feature_nn_metrics_path": context["model_root"] / OUTPUT_FILES[0],
        "feature_nn_model_path": context["model_root"] / OUTPUT_FILES[1],
        "feature_xgb_predictions_path": context["model_root"] / OUTPUT_FILES[2],
    }


def build_sequence(trace):
    label_seq = trace.get("label_json") or []
    return [entry["taxonomy_tag"] for entry in label_seq]


def vectorize_sequences(sequences, pad_idx, max_len):
    batch_size = len(sequences)
    arr = np.full((batch_size, max_len), pad_idx, dtype=np.int32)
    for i, seq in enumerate(sequences):
        shifted = [tag + 1 for tag in seq[:max_len]]
        arr[i, : len(shifted)] = shifted
    return arr


def residual_block(x, dim):
    shortcut = x
    if x.shape[-1] != dim:
        shortcut = tf.keras.layers.Dense(
            dim, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
        )(shortcut)
    out = tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(x)
    out = tf.keras.layers.LayerNormalization()(out)
    out = tf.keras.layers.Activation("gelu")(out)
    out = tf.keras.layers.Dropout(DROPOUT_RATE)(out)
    out = tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(out)
    out = tf.keras.layers.LayerNormalization()(out)
    out = tf.keras.layers.Activation("gelu")(out)
    out = tf.keras.layers.Dropout(DROPOUT_RATE)(out)
    if STOCHASTIC_DEPTH_RATE > 0:
        out = tf.keras.layers.Dropout(
            STOCHASTIC_DEPTH_RATE, noise_shape=(None, 1)
        )(out)
    out = tf.keras.layers.Add()([shortcut, out])
    out = tf.keras.layers.Activation("gelu")(out)
    return out


def transformer_block(x, attn_mask):
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=ATTN_HEADS,
        key_dim=ATTN_KEY_DIM,
        dropout=ATTN_DROPOUT,
    )(x, x, attention_mask=attn_mask)
    attn_out = tf.keras.layers.Dropout(DROPOUT_RATE)(attn_out)
    x = tf.keras.layers.LayerNormalization()(x + attn_out)
    ff = tf.keras.layers.Dense(
        ATTN_FF_DIM, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(x)
    ff = tf.keras.layers.Activation("gelu")(ff)
    ff = tf.keras.layers.Dropout(DROPOUT_RATE)(ff)
    ff = tf.keras.layers.Dense(
        SEQ_EMBED_DIM, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    return x


def build_model(num_features, normalizer, vocab_size, seq_len):
    tabular_inputs = tf.keras.Input(shape=(num_features,), dtype=tf.float32, name="tabular_inputs")
    seq_inputs = tf.keras.Input(shape=(seq_len,), dtype="int32", name="seq_inputs")

    tabular = normalizer(tabular_inputs)
    if TABULAR_NOISE > 0:
        tabular = tf.keras.layers.GaussianNoise(TABULAR_NOISE)(tabular)
    token_emb = tf.keras.layers.Embedding(
        vocab_size, SEQ_EMBED_DIM, mask_zero=True
    )(seq_inputs)
    if SEQ_SPATIAL_DROPOUT > 0:
        token_emb = tf.keras.layers.SpatialDropout1D(SEQ_SPATIAL_DROPOUT)(token_emb)
    pos_emb = tf.keras.layers.Embedding(seq_len, SEQ_EMBED_DIM)(tf.range(seq_len))
    seq = token_emb + tf.expand_dims(pos_emb, axis=0)
    mask = tf.keras.layers.Lambda(
        lambda x: tf.not_equal(x, 0), output_shape=lambda s: s
    )(seq_inputs)
    attn_mask = tf.keras.layers.Lambda(
        lambda x: x[:, tf.newaxis, :], output_shape=lambda s: (s[0], 1, s[1])
    )(mask)
    for _ in range(ATTN_LAYERS):
        seq = transformer_block(seq, attn_mask)
    mask_f = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32), output_shape=lambda s: s
    )(mask)
    seq = tf.keras.layers.Lambda(
        lambda inputs: tf.reduce_sum(inputs[0] * inputs[1][..., None], axis=1),
        output_shape=lambda s: (s[0][0], s[0][2]),
    )([seq, mask_f])
    denom = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True),
        output_shape=lambda s: (s[0], 1),
    )(mask_f)
    seq = tf.keras.layers.Lambda(
        lambda inputs: inputs[0] / tf.maximum(inputs[1], 1.0),
        output_shape=lambda s: s[0],
    )([seq, denom])
    for dim in SEQ_PROJ_DIMS:
        seq = tf.keras.layers.Dense(
            dim, kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
        )(seq)
        seq = tf.keras.layers.LayerNormalization()(seq)
        seq = tf.keras.layers.Activation("gelu")(seq)
        seq = tf.keras.layers.Dropout(DROPOUT_RATE)(seq)

    x = tf.keras.layers.Concatenate()([tabular, seq])
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    for dim in HIDDEN_DIMS:
        x = residual_block(x, dim)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([tabular_inputs, seq_inputs], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def compute_split_metrics(y_true, y_prob, mask):
    if not mask.any():
        return None
    y_true_slice = y_true[mask]
    y_pred_slice = (y_prob[mask] >= 0.5).astype(np.int32)
    return {
        "rows": int(mask.sum()),
        "accuracy": float((y_pred_slice == y_true_slice).mean()),
    }


def run(feature_path, annotator_path, output_dir):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = pd.read_csv(feature_path, index_col="pmcid")
    feature_cols = [
        c
        for c in df.columns
        if c not in {"is_correct", "is_second_correct", "source", "model"}
        and not c.startswith("lstm_emb_")
    ]
    X = df[feature_cols].copy()
    X = X.fillna(0.0).to_numpy(dtype=np.float32)
    y = df["is_correct"].to_numpy(dtype=np.int32)

    annotator_data = json.load(annotator_path.open())
    num_states = max(annotator_data["state_to_idx"].values()) + 1
    vocab_size = num_states + 1
    seq_by_key = {}
    for trace in annotator_data["traces"]:
        key = (trace["pmcid"], str(trace.get("model")))
        seq_by_key[key] = build_sequence(trace)

    keys = list(zip(df.index.tolist(), df["model"].astype(str).tolist()))
    sequences = []
    for key in keys:
        if key not in seq_by_key:
            raise ValueError(f"Missing sequence for pmcid={key[0]} model={key[1]}")
        sequences.append(seq_by_key[key])
    lengths = [len(seq) for seq in sequences]
    max_len = int(np.percentile(lengths, 95))
    max_len = min(max_len, MAX_SEQ_LEN)
    seq_inputs = vectorize_sequences(sequences, pad_idx=0, max_len=max_len)

    train_mask = df["source"] == "train"
    val_mask = df["source"] == "val"
    nejm_mask = df["source"] == "nejm"
    test_mask = df["source"] == "test"

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X[train_mask])

    model = build_model(X.shape[1], normalizer, vocab_size, max_len)
    train_weights = np.ones(train_mask.sum(), dtype=np.float32)
    pos = np.sum(y[train_mask] == 1)
    neg = np.sum(y[train_mask] == 0)
    if pos > 0 and neg > 0:
        train_weights = np.where(y[train_mask] == 1, neg / pos, 1.0).astype(np.float32)
        train_weights = train_weights / np.mean(train_weights)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        [X[train_mask], seq_inputs[train_mask]],
        y[train_mask],
        sample_weight=train_weights,
        validation_data=([X[val_mask], seq_inputs[val_mask]], y[val_mask]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
    )

    model_path = output_dir / OUTPUT_FILES[1]
    model.save(model_path, include_optimizer=False)

    probs = model.predict(
        [X, seq_inputs], batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    pred_labels = (probs >= 0.5).astype(np.int32)
    available_sources = set(df["source"].dropna().unique())
    eval_sources = ["test"]
    if "nejm" in available_sources:
        eval_sources.append("nejm")
    eval_mask = df["source"].isin(eval_sources)

    pred_df = df[eval_mask].reset_index()[["pmcid", "model"]].copy()
    pred_df["model"] = pred_df["model"].astype(str)
    pred_df["predicted_judgement"] = pred_labels[eval_mask.to_numpy()]
    predictions_path = output_dir / OUTPUT_FILES[2]
    pred_df.to_csv(predictions_path, index=False)

    metrics = {
        "feature_count": len(feature_cols),
        "train": compute_split_metrics(y, probs, train_mask),
        "val": compute_split_metrics(y, probs, val_mask),
        "nejm": compute_split_metrics(y, probs, nejm_mask),
        "test": compute_split_metrics(y, probs, test_mask),
        "config": {
            "hidden_dims": list(HIDDEN_DIMS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
            "l2_reg": L2_REG,
            "seq_embed_dim": SEQ_EMBED_DIM,
            "seq_proj_dims": list(SEQ_PROJ_DIMS),
            "attn_heads": ATTN_HEADS,
            "attn_key_dim": ATTN_KEY_DIM,
            "attn_ff_dim": ATTN_FF_DIM,
            "attn_layers": ATTN_LAYERS,
            "attn_dropout": ATTN_DROPOUT,
            "max_seq_len": int(max_len),
            "seq_spatial_dropout": SEQ_SPATIAL_DROPOUT,
            "tabular_noise": TABULAR_NOISE,
            "label_smoothing": LABEL_SMOOTHING,
            "stochastic_depth_rate": STOCHASTIC_DEPTH_RATE,
        },
    }
    metrics_path = output_dir / OUTPUT_FILES[0]
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
