import json

import numpy as np
import pandas as pd
import tensorflow as tf

OUTPUT_FILENAME = "lstm_autoencoder_features.csv"
MODEL_FILENAME = "lstm_autoencoder.keras"
PREDICTION_FILENAME = "feature_xgb_predictions.csv"
EPOCHS = 2000
BATCH_SIZE = 32*3
LATENT_DIMS = 32
OUTPUT_EMBED_DIMS = 32
EMBED_DIM = 128
ENCODER_LSTM_DIMS = (32, 64, 128)
DECODER_LSTM_DIMS = (128, 64, 32)
DENSE_DIMS = (128, 64, LATENT_DIMS)
DROPOUT_RATE = 0.1
L2_REG = 1e-4
LEARNING_RATE = 3e-4
KL_WEIGHT = 0.1
LM_LABEL_SMOOTHING = 0.05
CLS_LOSS_WEIGHT = 0.2
CLS_LABEL_SMOOTHING = 0.05


def inputs(context):
    annotator_path = context.get("annotator_path")
    if not annotator_path or not annotator_path.exists():
        raise FileNotFoundError("annotator output missing; run with --jobs annotator first.")
    metadata_path = context.get("metadata_path")
    if not metadata_path or not metadata_path.exists():
        raise FileNotFoundError("metadata output missing; run with --jobs metadata first.")

    return {
        "annotator_path": annotator_path,
        "metadata_path": metadata_path,
        "output_dir": context["model_root"],
    }


def outputs(context):
    return {"lstm_autoencoder_features_path": context["model_root"] / OUTPUT_FILENAME}


def build_sequence(trace):
    label_seq = trace.get("label_json") or []
    return [entry["taxonomy_tag"] for entry in label_seq]


def vectorize_sequences(sequences, pad_idx, bos_idx):
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)

    encoder_inputs = np.full((batch_size, max_len), pad_idx, dtype=np.int32)
    decoder_inputs = np.full((batch_size, max_len), pad_idx, dtype=np.int32)
    targets = np.full((batch_size, max_len), pad_idx, dtype=np.int32)
    for i, seq in enumerate(sequences):
        shifted = [tag + 1 for tag in seq]
        seq_len = len(shifted)
        encoder_inputs[i, :seq_len] = shifted
        targets[i, :seq_len] = shifted
        decoder_inputs[i, 0] = bos_idx
        if seq_len > 1:
            decoder_inputs[i, 1:seq_len] = shifted[:-1]

    return encoder_inputs, decoder_inputs, targets


class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        z_mean, z_logvar = inputs
        kl = -0.5 * tf.reduce_sum(
            1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1
        )
        self.add_loss(self.weight * tf.reduce_mean(kl))
        return inputs


def build_model(vocab_size):
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32")
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32")

    embedding = tf.keras.layers.Embedding(
        vocab_size, EMBED_DIM, mask_zero=True, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG)
    )
    enc_emb = embedding(encoder_inputs)
    enc_out = enc_emb
    if len(ENCODER_LSTM_DIMS) > 1:
        for dim in ENCODER_LSTM_DIMS[:-1]:
            enc_out = tf.keras.layers.LSTM(dim, return_sequences=True)(enc_out)
    _, state_h, _ = tf.keras.layers.LSTM(
        ENCODER_LSTM_DIMS[-1], return_state=True
    )(enc_out)

    base = state_h
    for dim in DENSE_DIMS:
        base = tf.keras.layers.Dense(dim, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(base)
        base = tf.keras.layers.LayerNormalization()(base)
        base = tf.keras.layers.LeakyReLU()(base)

    z_mean = tf.keras.layers.Dense(
        LATENT_DIMS, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(base)
    z_logvar = tf.keras.layers.Dense(
        LATENT_DIMS, name="z_logvar", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(base)
    z_mean, z_logvar = KLDivergenceLayer(weight=KL_WEIGHT)([z_mean, z_logvar])
    z_mean_unit = tf.keras.layers.LayerNormalization(
        center=False, scale=False, name="z_mean_norm"
    )(z_mean)
    z_mean_unit = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=-1), name="z_mean_unit"
    )(z_mean_unit)
    eps = tf.keras.layers.Lambda(
        lambda x: tf.random.normal(tf.shape(x)),
        output_shape=lambda s: s,
    )(z_mean)
    z = tf.keras.layers.Lambda(
        lambda inputs: inputs[0] + tf.exp(0.5 * inputs[1]) * inputs[2],
        output_shape=lambda s: s[0],
    )([z_mean, z_logvar, eps])

    dec_emb = embedding(decoder_inputs)
    init_h = tf.keras.layers.Dense(
        DECODER_LSTM_DIMS[0], activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )
    init_c = tf.keras.layers.Dense(
        DECODER_LSTM_DIMS[0], activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )
    decoder_lstm_layers = [
        tf.keras.layers.LSTM(DECODER_LSTM_DIMS[0], return_sequences=True)
    ] + [
        tf.keras.layers.LSTM(dim, return_sequences=True)
        for dim in DECODER_LSTM_DIMS[1:]
    ]
    decoder_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)
    decoder_output = tf.keras.layers.Dense(
        vocab_size, name="decoder_logits", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )

    def decode(latent):
        h0 = init_h(latent)
        c0 = init_c(latent)
        dec_out = decoder_lstm_layers[0](dec_emb, initial_state=[h0, c0])
        for layer in decoder_lstm_layers[1:]:
            dec_out = layer(dec_out)
        dec_out = decoder_dropout(dec_out)
        return decoder_output(dec_out)

    logits = decode(z)
    recon_logits = decode(z_mean)
    correctness_logit = tf.keras.layers.Dense(
        1, name="correctness_logit", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(z_mean_unit)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], [logits, correctness_logit])
    encoder_model = tf.keras.Model(encoder_inputs, z_mean_unit)
    recon_model = tf.keras.Model([encoder_inputs, decoder_inputs], recon_logits)
    classifier_model = tf.keras.Model(encoder_inputs, correctness_logit)

    def smoothed_sparse_cce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        log_probs = tf.nn.log_softmax(y_pred)
        uniform = -tf.reduce_mean(log_probs, axis=-1)
        return (1.0 - LM_LABEL_SMOOTHING) * nll + LM_LABEL_SMOOTHING * uniform

    def smoothed_binary_ce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        if CLS_LABEL_SMOOTHING > 0:
            y_true = y_true * (1.0 - CLS_LABEL_SMOOTHING) + 0.5 * CLS_LABEL_SMOOTHING
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss=[smoothed_sparse_cce, smoothed_binary_ce],
        loss_weights=[1.0, CLS_LOSS_WEIGHT],
    )
    return model, encoder_model, recon_model, classifier_model


def train_vae(
    model,
    encoder_inputs,
    decoder_inputs,
    targets,
    labels,
    pad_idx,
    val_encoder_inputs,
    val_decoder_inputs,
    val_targets,
    val_labels,
    batch_size,
):
    mask = (targets != pad_idx).astype(np.float32)
    val_mask = (val_targets != pad_idx).astype(np.float32)
    seq_lengths = mask.sum(axis=1)
    seq_weights = 1.0 / np.sqrt(np.maximum(seq_lengths, 1.0))
    sample_weight = mask * seq_weights[:, None]
    cls_weight = seq_weights
    val_lengths = val_mask.sum(axis=1)
    val_seq_weights = 1.0 / np.sqrt(np.maximum(val_lengths, 1.0))
    val_sample_weight = val_mask * val_seq_weights[:, None]
    val_cls_weight = val_seq_weights
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]
    model.fit(
        [encoder_inputs, decoder_inputs],
        [targets, labels],
        sample_weight=[sample_weight, cls_weight],
        validation_data=(
            [val_encoder_inputs, val_decoder_inputs],
            [val_targets, val_labels],
            [val_sample_weight, val_cls_weight],
        ),
        epochs=EPOCHS,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
    )


def compute_reconstruction_losses(model, encoder_inputs, decoder_inputs, targets, pad_idx, batch_size):
    logits = model.predict([encoder_inputs, decoder_inputs], batch_size=batch_size, verbose=0)
    token_loss = tf.keras.losses.sparse_categorical_crossentropy(
        targets, logits, from_logits=True
    ).numpy()
    mask = (targets != pad_idx).astype(np.float32)
    masked_loss = token_loss * mask
    lengths = mask.sum(axis=1)
    lengths = np.maximum(lengths, 1.0)
    return (masked_loss.sum(axis=1) / lengths).tolist()


def run(annotator_path, metadata_path, output_dir):
    data = json.load(annotator_path.open())
    metadata = pd.read_csv(metadata_path)
    metadata["pmcid"] = metadata["pmcid"].astype(str)
    metadata["model"] = metadata["model"].astype(str)
    source_by_key = {
        (row["pmcid"], row["model"]): row["source"]
        for _, row in metadata.iterrows()
    }
    label_by_key = {
        (row["pmcid"], row["model"]): int(row["is_correct"])
        for _, row in metadata.iterrows()
    }
    num_states = max(data["state_to_idx"].values()) + 1
    pad_idx = 0
    bos_idx = num_states + 1
    vocab_size = num_states + 2

    traces = data["traces"]
    raw_sequences = []
    lengths = []
    for trace in traces:
        seq = build_sequence(trace)
        raw_sequences.append(seq)
        lengths.append(len(seq))
    max_len = int(np.percentile(lengths, 95))

    all_sequences = []
    train_sequences = []
    val_sequences = []
    train_labels = []
    val_labels = []
    missing_source = 0
    missing_label = 0

    for trace, seq in zip(traces, raw_sequences):
        truncated = seq[:max_len]
        all_sequences.append(truncated)
        key = (str(trace["pmcid"]), str(trace.get("model")))
        source = source_by_key.get(key)
        if source is None:
            missing_source += 1
            continue
        label = label_by_key.get(key)
        if label is None:
            missing_label += 1
            continue
        if source == "train":
            train_sequences.append(truncated)
            train_labels.append(label)
        elif source == "val":
            val_sequences.append(truncated)
            val_labels.append(label)
    if missing_source:
        print(f"Skipping {missing_source} trace(s) without split assignment for LSTM training.")
    if missing_label:
        print(f"Skipping {missing_label} trace(s) without metadata labels for LSTM training.")

    gpus = tf.config.list_physical_devices("GPU")
    num_replicas = len(gpus)
    if num_replicas > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model, encoder_model, recon_model, classifier_model = build_model(vocab_size)

    train_inputs, train_dec_inputs, train_targets = vectorize_sequences(train_sequences, pad_idx, bos_idx)
    val_inputs, val_dec_inputs, val_targets = vectorize_sequences(val_sequences, pad_idx, bos_idx)
    train_vae(
        model,
        train_inputs,
        train_dec_inputs,
        train_targets,
        np.array(train_labels, dtype=np.float32),
        pad_idx,
        val_inputs,
        val_dec_inputs,
        val_targets,
        np.array(val_labels, dtype=np.float32),
        batch_size=BATCH_SIZE,
    )
    model_path = output_dir / MODEL_FILENAME
    model.save(model_path, include_optimizer=False)

    infer_model, infer_encoder_model, infer_recon_model, infer_classifier_model = build_model(vocab_size)
    infer_model.set_weights(model.get_weights())
    infer_encoder_model.set_weights(encoder_model.get_weights())
    infer_recon_model.set_weights(recon_model.get_weights())
    infer_classifier_model.set_weights(classifier_model.get_weights())

    all_inputs, all_dec_inputs, all_targets = vectorize_sequences(all_sequences, pad_idx, bos_idx)
    losses = compute_reconstruction_losses(
        infer_recon_model, all_inputs, all_dec_inputs, all_targets, pad_idx, batch_size=BATCH_SIZE
    )
    embeddings = infer_encoder_model.predict(
        all_inputs, batch_size=BATCH_SIZE, verbose=0
    )
    correctness_logits = infer_classifier_model.predict(
        all_inputs, batch_size=BATCH_SIZE, verbose=0
    )
    correctness_probs = tf.sigmoid(correctness_logits).numpy().reshape(-1)

    embedding_cols = [f"lstm_emb_{i}" for i in range(OUTPUT_EMBED_DIMS)]
    rows = []
    for trace, loss, emb, prob in zip(traces, losses, embeddings, correctness_probs):
        row = {
            "pmcid": trace["pmcid"],
            "model": trace.get("model"),
            "lstm_reconstruction_loss": loss,
            "lstm_correctness_prob": float(prob),
        }
        row.update({col: float(val) for col, val in zip(embedding_cols, emb)})
        rows.append(row)

    output_path = output_dir / OUTPUT_FILENAME
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved LSTM VAE features to {output_path} ({len(rows)} rows)")
    print(f"Saved LSTM VAE model to {model_path}")

    pred_labels = (correctness_probs >= 0.5).astype(np.int32)
    available_sources = set(source_by_key.values())
    eval_sources = ["test"]
    if "nejm" in available_sources:
        eval_sources.append("nejm")
    eval_mask = np.array(
        [
            source_by_key.get((str(trace["pmcid"]), str(trace.get("model")))) in eval_sources
            for trace in traces
        ]
    )

    pred_df = df.loc[eval_mask, ["pmcid", "model"]].copy()
    pred_df["model"] = pred_df["model"].astype(str)
    pred_df["predicted_judgement"] = pred_labels[eval_mask]
    predictions_path = output_dir / PREDICTION_FILENAME
    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved LSTM correctness predictions to {predictions_path}")
