"""
Permutation importance for the Temporal Transformer model.

This script trains a Transformer-based soil moisture model with temporal
train/validation/test splitting and computes permutation importance on the
held-out test set. Dynamic features are permuted across samples while preserving
the within-window temporal order of each feature sequence.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("/home/miladkb/projects/Station_based_hourly/5_cm_daily/")
    result_dir: Path = Path("/home/miladkb/projects/Paper/Results/Permutation/Temporal_Transformer_5cm_daily_withGW")

    temporal_resolution: str = "daily"  # options: hourly, daily, weekly
    target_col: str = "Soil_TP5_VMC"

    d_model: int = 128
    num_heads: int = 8
    ff_dim: int = 512
    num_blocks: int = 4
    dropout: float = 0.15
    l2_lambda: float = 1e-5

    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    huber_delta: float = 0.15

    n_repeats: int = 3
    max_test_samples: int = 20_000
    random_seed: int = 42


CFG = Config()

SEQ_SIZE_BY_RESOLUTION = {
    "hourly": 96,
    "daily": 45,
    "weekly": 10,
}

DYNAMIC_COLS = [
    "Pluvio_Rain",
    "AvgAir_T",
    "AvgRH",
    "Groundwater",
    "Day_sin",
    "Day_cos",
]

STATIC_COLS = [
    "LatDD",
    "LongDD",
    "Elevation",
    "AQUIFER_NUM",
    "WELL_DEPTH",
]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_sequence_length(temporal_resolution: str) -> int:
    try:
        return SEQ_SIZE_BY_RESOLUTION[temporal_resolution.lower()]
    except KeyError as exc:
        valid = ", ".join(SEQ_SIZE_BY_RESOLUTION)
        raise ValueError(f"temporal_resolution must be one of: {valid}") from exc


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def require_columns(df: pd.DataFrame, required_cols: Iterable[str], file_path: Path) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {file_path.name}: {missing}")


# -----------------------------------------------------------------------------
# Data loading and sequence preparation
# -----------------------------------------------------------------------------

def load_station_file(
    file_path: Path,
    dynamic_cols: list[str],
    static_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    require_columns(df, ["Date", target_col, *dynamic_cols, *static_cols], file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Day"] = df["Date"].dt.dayofyear
    df["Day_sin"] = np.sin(2.0 * np.pi * df["Day"] / 365.0)
    df["Day_cos"] = np.cos(2.0 * np.pi * df["Day"] / 365.0)
    df["Station"] = file_path.stem

    target = df[[target_col]].to_numpy(dtype="float32")
    dynamic = df[dynamic_cols].to_numpy(dtype="float32")
    static = df[static_cols].to_numpy(dtype="float32")

    return df, target, dynamic, static


def make_sequences(
    target: np.ndarray,
    dynamic: np.ndarray,
    static: np.ndarray,
    seq_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sequences = len(target) - seq_size
    if n_sequences <= 0:
        return (
            np.empty((0, seq_size, dynamic.shape[1]), dtype="float32"),
            np.empty((0, static.shape[1]), dtype="float32"),
            np.empty((0,), dtype="float32"),
        )

    x_dyn = np.stack(
        [dynamic[i : i + seq_size] for i in range(n_sequences)],
        axis=0,
    ).astype("float32")

    x_static = static[seq_size:].astype("float32")
    y = target[seq_size:, 0].astype("float32")

    return x_dyn, x_static, y


def build_sequences_from_station_list(
    station_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    seq_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_dyn_all, x_static_all, y_all = [], [], []

    for target, dynamic, static in station_list:
        x_dyn, x_static, y = make_sequences(target, dynamic, static, seq_size)
        if len(y) > 0:
            x_dyn_all.append(x_dyn)
            x_static_all.append(x_static)
            y_all.append(y)

    if not x_dyn_all:
        raise ValueError("No valid sequences were created. Check data length and seq_size.")

    return np.vstack(x_dyn_all), np.vstack(x_static_all), np.concatenate(y_all)


def load_all_stations(cfg: Config, seq_size: int) -> list[dict]:
    station_data = []

    for file_path in sorted(cfg.data_dir.glob("*.csv")):
        df, target, dynamic, static = load_station_file(
            file_path=file_path,
            dynamic_cols=DYNAMIC_COLS,
            static_cols=STATIC_COLS,
            target_col=cfg.target_col,
        )

        if len(target) <= seq_size:
            print(f"Skipping {file_path.name}: not enough rows for seq_size={seq_size}")
            continue

        station_data.append(
            {
                "station": file_path.stem,
                "df": df,
                "target": target,
                "dynamic": dynamic,
                "static": static,
            }
        )

    if not station_data:
        raise ValueError(f"No valid station CSV files found in {cfg.data_dir}")

    return station_data


def temporal_split_by_station(
    station_data: list[dict],
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    train_list, val_list, test_list = [], [], []

    for item in station_data:
        target = item["target"]
        dynamic = item["dynamic"]
        static = item["static"]

        n_total = len(target)
        train_end = int(0.60 * n_total)
        val_end = int(0.80 * n_total)

        train_list.append((target[:train_end], dynamic[:train_end], static[:train_end]))
        val_list.append((target[train_end:val_end], dynamic[train_end:val_end], static[train_end:val_end]))
        test_list.append((target[val_end:], dynamic[val_end:], static[val_end:]))

    return train_list, val_list, test_list


# -----------------------------------------------------------------------------
# Scaling
# -----------------------------------------------------------------------------

def fit_scalers(
    train_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    result_dir: Path,
) -> tuple[MinMaxScaler, MinMaxScaler, MinMaxScaler]:
    target_scaler = MinMaxScaler()
    dyn_scaler = MinMaxScaler()
    stat_scaler = MinMaxScaler()

    train_target = np.vstack([target for target, _, _ in train_list])
    train_dynamic = np.vstack([dynamic for _, dynamic, _ in train_list])
    train_static = np.vstack([static for _, _, static in train_list])

    target_scaler.fit(train_target)
    dyn_scaler.fit(train_dynamic)
    stat_scaler.fit(train_static)

    joblib.dump(target_scaler, result_dir / "target_scaler.pkl")
    joblib.dump(dyn_scaler, result_dir / "dynamic_scaler.pkl")
    joblib.dump(stat_scaler, result_dir / "static_scaler.pkl")

    return target_scaler, dyn_scaler, stat_scaler


def scale_station_list(
    station_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_scaler: MinMaxScaler,
    dyn_scaler: MinMaxScaler,
    stat_scaler: MinMaxScaler,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    scaled = []

    for target, dynamic, static in station_list:
        scaled.append(
            (
                target_scaler.transform(target),
                dyn_scaler.transform(dynamic),
                stat_scaler.transform(static),
            )
        )

    return scaled


# -----------------------------------------------------------------------------
# Temporal Transformer model
# -----------------------------------------------------------------------------

def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    pe = np.zeros((seq_len, d_model), dtype="float32")

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** ((i // 2) / d_model))
            pe[pos, i] = math.sin(angle)
            if i + 1 < d_model:
                pe[pos, i + 1] = math.cos(angle)

    return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)


def transformer_encoder_block(
    x: tf.Tensor,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
) -> tf.Tensor:
    attention_input = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(attention_input, attention_input)
    x = Add()([x, attention_output])

    ffn_input = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = Dense(ff_dim, activation="gelu")(ffn_input)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)
    x = Add()([x, ffn_output])

    return x


def build_temporal_transformer_model(
    seq_size: int,
    n_dynamic_features: int,
    n_static_features: int,
    cfg: Config,
) -> Model:
    dynamic_input = Input(shape=(seq_size, n_dynamic_features), name="dynamic_sequence")
    static_input = Input(shape=(n_static_features,), name="static_attributes")

    x = Dense(cfg.d_model, name="linear_projection")(dynamic_input)
    x = x + positional_encoding(seq_size, cfg.d_model)

    for block_id in range(cfg.num_blocks):
        x = transformer_encoder_block(
            x=x,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )

    temporal_repr = GlobalAveragePooling1D(name="global_average_pooling")(x)
    temporal_repr = Dense(cfg.d_model, activation="relu", name="temporal_dense")(temporal_repr)

    static_repr = Dense(32, activation="relu", name="static_dense_1")(static_input)
    static_repr = Dropout(cfg.dropout, name="static_dropout")(static_repr)
    static_repr = Dense(32, activation="relu", name="static_dense_2")(static_repr)

    z = Concatenate(name="feature_concatenation")([temporal_repr, static_repr])
    z = Dense(64, activation="relu", kernel_regularizer=l2(cfg.l2_lambda), name="dense_64")(z)
    z = Dropout(cfg.dropout, name="head_dropout")(z)
    z = Dense(32, activation="relu", kernel_regularizer=l2(cfg.l2_lambda), name="dense_32")(z)

    output = Dense(1, name="soil_moisture")(z)

    return Model(inputs=[dynamic_input, static_input], outputs=output, name="Temporal_Transformer_Model")


# -----------------------------------------------------------------------------
# Permutation importance
# -----------------------------------------------------------------------------

def inverse_target(
    values: np.ndarray,
    target_scaler: MinMaxScaler,
) -> np.ndarray:
    return target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def permutation_importance(
    model: Model,
    x_dyn: np.ndarray,
    x_static: np.ndarray,
    y_true: np.ndarray,
    dynamic_cols: list[str],
    static_cols: list[str],
    target_scaler: MinMaxScaler,
    n_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    base_pred_scaled = model.predict([x_dyn, x_static], verbose=0).reshape(-1)
    y_true_inv = inverse_target(y_true, target_scaler)
    base_pred_inv = inverse_target(base_pred_scaled, target_scaler)
    base_rmse = rmse(y_true_inv, base_pred_inv)

    rows = []
    n_samples = x_dyn.shape[0]

    for feature_index, feature_name in enumerate(dynamic_cols):
        repeat_scores = []

        for _ in range(n_repeats):
            x_dyn_perm = x_dyn.copy()
            shuffle_index = rng.permutation(n_samples)
            x_dyn_perm[:, :, feature_index] = x_dyn_perm[shuffle_index, :, feature_index]

            pred_scaled = model.predict([x_dyn_perm, x_static], verbose=0).reshape(-1)
            pred_inv = inverse_target(pred_scaled, target_scaler)
            repeat_scores.append(rmse(y_true_inv, pred_inv))

        mean_rmse = float(np.mean(repeat_scores))
        std_rmse = float(np.std(repeat_scores, ddof=1)) if n_repeats > 1 else 0.0

        rows.append(
            {
                "Feature": feature_name,
                "Feature_Type": "Dynamic",
                "Baseline_RMSE": base_rmse,
                "Permuted_RMSE_Mean": mean_rmse,
                "Permuted_RMSE_Std": std_rmse,
                "RMSE_Increase": mean_rmse - base_rmse,
                "Relative_RMSE_Increase_Percent": 100.0 * (mean_rmse - base_rmse) / base_rmse,
            }
        )

    for feature_index, feature_name in enumerate(static_cols):
        repeat_scores = []

        for _ in range(n_repeats):
            x_static_perm = x_static.copy()
            shuffle_index = rng.permutation(n_samples)
            x_static_perm[:, feature_index] = x_static_perm[shuffle_index, feature_index]

            pred_scaled = model.predict([x_dyn, x_static_perm], verbose=0).reshape(-1)
            pred_inv = inverse_target(pred_scaled, target_scaler)
            repeat_scores.append(rmse(y_true_inv, pred_inv))

        mean_rmse = float(np.mean(repeat_scores))
        std_rmse = float(np.std(repeat_scores, ddof=1)) if n_repeats > 1 else 0.0

        rows.append(
            {
                "Feature": feature_name,
                "Feature_Type": "Static",
                "Baseline_RMSE": base_rmse,
                "Permuted_RMSE_Mean": mean_rmse,
                "Permuted_RMSE_Std": std_rmse,
                "RMSE_Increase": mean_rmse - base_rmse,
                "Relative_RMSE_Increase_Percent": 100.0 * (mean_rmse - base_rmse) / base_rmse,
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE_Increase", ascending=False)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    set_global_seed(CFG.random_seed)
    CFG.result_dir.mkdir(parents=True, exist_ok=True)

    seq_size = get_sequence_length(CFG.temporal_resolution)

    with open(CFG.result_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(CFG) | {"seq_size": seq_size, "dynamic_cols": DYNAMIC_COLS, "static_cols": STATIC_COLS}, f, indent=2, default=str)

    station_data = load_all_stations(CFG, seq_size)
    train_raw, val_raw, test_raw = temporal_split_by_station(station_data)

    target_scaler, dyn_scaler, stat_scaler = fit_scalers(train_raw, CFG.result_dir)

    train_scaled = scale_station_list(train_raw, target_scaler, dyn_scaler, stat_scaler)
    val_scaled = scale_station_list(val_raw, target_scaler, dyn_scaler, stat_scaler)
    test_scaled = scale_station_list(test_raw, target_scaler, dyn_scaler, stat_scaler)

    train_x_dyn, train_x_static, train_y = build_sequences_from_station_list(train_scaled, seq_size)
    val_x_dyn, val_x_static, val_y = build_sequences_from_station_list(val_scaled, seq_size)
    test_x_dyn, test_x_static, test_y = build_sequences_from_station_list(test_scaled, seq_size)

    model = build_temporal_transformer_model(
        seq_size=seq_size,
        n_dynamic_features=len(DYNAMIC_COLS),
        n_static_features=len(STATIC_COLS),
        cfg=CFG,
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(delta=CFG.huber_delta),
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=CFG.patience, restore_best_weights=True),
        ModelCheckpoint(
            filepath=CFG.result_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        [train_x_dyn, train_x_static],
        train_y,
        validation_data=([val_x_dyn, val_x_static], val_y),
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    training_time_sec = time.time() - start_time

    pd.DataFrame(history.history).to_csv(CFG.result_dir / "training_history.csv", index=False)

    training_metadata = {
        "temporal_resolution": CFG.temporal_resolution,
        "seq_size": seq_size,
        "d_model": CFG.d_model,
        "num_heads": CFG.num_heads,
        "ff_dim": CFG.ff_dim,
        "num_blocks": CFG.num_blocks,
        "dropout": CFG.dropout,
        "epochs_completed": len(history.history["loss"]),
        "training_time_sec": training_time_sec,
        "n_train_sequences": int(len(train_y)),
        "n_val_sequences": int(len(val_y)),
        "n_test_sequences": int(len(test_y)),
    }
    pd.Series(training_metadata).to_csv(CFG.result_dir / "training_metadata.csv")

    model_path = CFG.result_dir / "temporal_transformer_model.keras"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    if len(test_y) > CFG.max_test_samples:
        rng = np.random.default_rng(CFG.random_seed)
        sample_idx = rng.choice(len(test_y), size=CFG.max_test_samples, replace=False)
        test_x_dyn_perm = test_x_dyn[sample_idx]
        test_x_static_perm = test_x_static[sample_idx]
        test_y_perm = test_y[sample_idx]
    else:
        test_x_dyn_perm = test_x_dyn
        test_x_static_perm = test_x_static
        test_y_perm = test_y

    importance_df = permutation_importance(
        model=model,
        x_dyn=test_x_dyn_perm,
        x_static=test_x_static_perm,
        y_true=test_y_perm,
        dynamic_cols=DYNAMIC_COLS,
        static_cols=STATIC_COLS,
        target_scaler=target_scaler,
        n_repeats=CFG.n_repeats,
        random_seed=CFG.random_seed,
    )

    importance_path = CFG.result_dir / "permutation_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(importance_df)
    print(f"Permutation importance saved to: {importance_path}")


if __name__ == "__main__":
    main()
