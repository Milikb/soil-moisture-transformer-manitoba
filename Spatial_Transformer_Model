"""
Spatial Transformer Model for soil moisture prediction.

This script trains a Transformer-based sequence model using a spatial
station split. Entire stations are assigned to train, validation, or test
sets, so the test set evaluates prediction at withheld stations.

Temporal resolution controls the look-back window:
    hourly: 96 time steps
    daily:  45 time steps
    weekly: 10 time steps
"""

from pathlib import Path
import math
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
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


# ============================================================
# Configuration
# ============================================================
TEMPORAL_RESOLUTION = "daily"  # options: "hourly", "daily", "weekly"
SEQ_SIZE_BY_RESOLUTION = {
    "hourly": 96,
    "daily": 45,
    "weekly": 10,
}

DATA_DIR = Path("/home/miladkb/projects/Station_based_hourly/5_cm_daily/")
RESULT_DIR = Path("/home/miladkb/projects/Paper/Results/Spatial_Temporal/Tran_spa_daily_5cm")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Soil_TP5_VMC"
RANDOM_SEED = 42

SEQ_SIZE = SEQ_SIZE_BY_RESOLUTION[TEMPORAL_RESOLUTION]
D_MODEL = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_BLOCKS = 4
DROPOUT = 0.15
L2_WEIGHT = 1e-5

EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
HUBER_DELTA = 0.15

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
]


# ============================================================
# Data loading and sequence preparation
# ============================================================
def load_station_file(file_path: Path) -> dict:
    """Load one station file and create seasonal predictors."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Day"] = df["Date"].dt.dayofyear
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365.0)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365.0)
    df["Station"] = file_path.stem

    required_cols = ["Date", TARGET_COL] + DYNAMIC_COLS + STATIC_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path.name} is missing columns: {missing_cols}")

    df = df.dropna(subset=[TARGET_COL] + DYNAMIC_COLS + STATIC_COLS).reset_index(drop=True)

    return {
        "station": file_path.stem,
        "file_name": file_path.name,
        "df": df,
        "target": df[[TARGET_COL]].to_numpy(dtype="float32"),
        "dynamic": df[DYNAMIC_COLS].to_numpy(dtype="float32"),
        "static": df[STATIC_COLS].to_numpy(dtype="float32"),
    }


def make_sequences(target: np.ndarray, dynamic: np.ndarray, static: np.ndarray, seq_size: int):
    """Create one-step-ahead input sequences and targets."""
    x_dynamic, x_static, y = [], [], []

    for i in range(len(target) - seq_size):
        forecast_idx = i + seq_size
        x_dynamic.append(dynamic[i:forecast_idx])
        x_static.append(static[forecast_idx])
        y.append(target[forecast_idx, 0])

    return (
        np.asarray(x_dynamic, dtype="float32"),
        np.asarray(x_static, dtype="float32"),
        np.asarray(y, dtype="float32"),
    )


def build_sequences(station_records: list, seq_size: int):
    """Build sequences from a list of station records."""
    x_dynamic_all, x_static_all, y_all = [], [], []

    for record in station_records:
        x_dynamic, x_static, y = make_sequences(
            record["target"], record["dynamic"], record["static"], seq_size
        )
        if len(x_dynamic) > 0:
            x_dynamic_all.append(x_dynamic)
            x_static_all.append(x_static)
            y_all.append(y)

    if not x_dynamic_all:
        raise ValueError("No valid sequences were created. Check station lengths and sequence size.")

    return (
        np.vstack(x_dynamic_all),
        np.vstack(x_static_all),
        np.concatenate(y_all),
    )


# ============================================================
# Transformer model
# ============================================================
def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """Sinusoidal positional encoding."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates

    pe = np.zeros((seq_len, d_model), dtype="float32")
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)


def transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    """Pre-normalized Transformer encoder block."""
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(x_norm, x_norm)
    x = Add()([x, attention_output])

    x_norm = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = Dense(ff_dim, activation="gelu")(x_norm)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)
    x = Add()([x, ffn_output])

    return x


def build_spatial_transformer_model(
    seq_size: int,
    n_dynamic_features: int,
    n_static_features: int,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    num_blocks: int,
    dropout: float,
    l2_weight: float,
) -> Model:
    """Build the spatial Transformer model with dynamic and static inputs."""
    dynamic_input = Input(shape=(seq_size, n_dynamic_features), name="dynamic_input")
    static_input = Input(shape=(n_static_features,), name="static_input")

    x = Dense(d_model, name="linear_projection")(dynamic_input)
    x = x + positional_encoding(seq_size, d_model)

    for block_id in range(num_blocks):
        x = transformer_encoder_block(
            x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    temporal_features = GlobalAveragePooling1D(name="global_average_pooling")(x)
    temporal_features = Dense(d_model, activation="relu", name="temporal_dense")(
        temporal_features
    )

    static_features = Dense(32, activation="relu", name="static_dense_1")(static_input)
    static_features = Dropout(dropout, name="static_dropout")(static_features)
    static_features = Dense(32, activation="relu", name="static_dense_2")(static_features)

    z = Concatenate(name="feature_concatenation")([temporal_features, static_features])
    z = Dense(64, activation="relu", kernel_regularizer=l2(l2_weight), name="dense_64")(z)
    z = Dense(32, activation="relu", kernel_regularizer=l2(l2_weight), name="dense_32")(z)
    output = Dense(1, name="soil_moisture")(z)

    return Model(inputs=[dynamic_input, static_input], outputs=output)


# ============================================================
# Main workflow
# ============================================================
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    station_files = sorted(DATA_DIR.glob("*.csv"))
    if not station_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    station_records = []
    for file_path in station_files:
        try:
            record = load_station_file(file_path)
        except ValueError as err:
            print(f"Skipping {file_path.name}: {err}")
            continue

        if len(record["target"]) <= SEQ_SIZE:
            print(f"Skipping {file_path.name}: not enough rows for seq_size={SEQ_SIZE}")
            continue

        station_records.append(record)

    if not station_records:
        raise ValueError("No valid stations found after filtering.")

    n_stations = len(station_records)
    shuffled_indices = np.random.permutation(n_stations)

    train_end = int(0.60 * n_stations)
    val_end = int(0.80 * n_stations)

    train_indices = shuffled_indices[:train_end]
    val_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]

    train_records = [station_records[i] for i in train_indices]
    val_records = [station_records[i] for i in val_indices]
    test_records = [station_records[i] for i in test_indices]

    split_rows = []
    split_map = {}
    for split_name, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        for record in records:
            split_map[record["station"]] = split_name
            split_rows.append({"Station": record["station"], "Split": split_name})

    pd.DataFrame(split_rows).to_csv(RESULT_DIR / "station_split_assignments.csv", index=False)

    print(f"Total stations: {n_stations}")
    print(f"Train stations: {len(train_records)}")
    print(f"Validation stations: {len(val_records)}")
    print(f"Test stations: {len(test_records)}")

    target_scaler = MinMaxScaler()
    dynamic_scaler = MinMaxScaler()
    static_scaler = MinMaxScaler()

    target_scaler.fit(np.vstack([record["target"] for record in train_records]))
    dynamic_scaler.fit(np.vstack([record["dynamic"] for record in train_records]))
    static_scaler.fit(np.vstack([record["static"] for record in train_records]))

    joblib.dump(target_scaler, RESULT_DIR / "target_scaler.pkl")
    joblib.dump(dynamic_scaler, RESULT_DIR / "dynamic_scaler.pkl")
    joblib.dump(static_scaler, RESULT_DIR / "static_scaler.pkl")

    def scale_records(records: list) -> list:
        scaled_records = []
        for record in records:
            scaled_record = record.copy()
            scaled_record["target"] = target_scaler.transform(record["target"])
            scaled_record["dynamic"] = dynamic_scaler.transform(record["dynamic"])
            scaled_record["static"] = static_scaler.transform(record["static"])
            scaled_records.append(scaled_record)
        return scaled_records

    train_scaled = scale_records(train_records)
    val_scaled = scale_records(val_records)
    test_scaled = scale_records(test_records)

    train_x_dynamic, train_x_static, train_y = build_sequences(train_scaled, SEQ_SIZE)
    val_x_dynamic, val_x_static, val_y = build_sequences(val_scaled, SEQ_SIZE)
    test_x_dynamic, test_x_static, test_y = build_sequences(test_scaled, SEQ_SIZE)

    model = build_spatial_transformer_model(
        seq_size=SEQ_SIZE,
        n_dynamic_features=len(DYNAMIC_COLS),
        n_static_features=len(STATIC_COLS),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
        l2_weight=L2_WEIGHT,
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(
            RESULT_DIR / "best_spatial_transformer_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        [train_x_dynamic, train_x_static],
        train_y,
        validation_data=([val_x_dynamic, val_x_static], val_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
    )
    training_time_sec = time.time() - start_time

    pd.DataFrame(history.history).to_csv(RESULT_DIR / "training_history.csv", index=False)

    metadata = {
        "model_name": "Spatial Transformer Model",
        "temporal_resolution": TEMPORAL_RESOLUTION,
        "seq_size": SEQ_SIZE,
        "target_col": TARGET_COL,
        "dynamic_cols": ", ".join(DYNAMIC_COLS),
        "static_cols": ", ".join(STATIC_COLS),
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "ff_dim": FF_DIM,
        "num_blocks": NUM_BLOCKS,
        "dropout": DROPOUT,
        "l2_weight": L2_WEIGHT,
        "loss": f"Huber(delta={HUBER_DELTA})",
        "epochs_completed": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "training_time_sec": training_time_sec,
        "n_train_stations": len(train_records),
        "n_val_stations": len(val_records),
        "n_test_stations": len(test_records),
        "random_seed": RANDOM_SEED,
    }
    pd.Series(metadata).to_csv(RESULT_DIR / "training_metadata.csv")

    final_model_path = RESULT_DIR / "spatial_transformer_model.keras"
    model.save(final_model_path)
    print(f"Spatial Transformer model saved to {final_model_path}")

    all_scaled_records = scale_records(station_records)
    for record in all_scaled_records:
        station_name = record["station"]
        original_df = next(item["df"] for item in station_records if item["station"] == station_name)

        x_dynamic, x_static, y_seq = make_sequences(
            record["target"], record["dynamic"], record["static"], SEQ_SIZE
        )

        if len(x_dynamic) == 0:
            print(f"Skipping {station_name}: no valid prediction sequences.")
            continue

        y_pred_scaled = model.predict([x_dynamic, x_static], verbose=0).reshape(-1, 1)
        y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = target_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()

        prediction_df = pd.DataFrame(
            {
                "Date": original_df["Date"].iloc[SEQ_SIZE:].reset_index(drop=True),
                "Observed": y_true,
                "Predicted": y_pred,
                "Pluvio_Rain": original_df["Pluvio_Rain"].iloc[SEQ_SIZE:].to_numpy(),
                "AvgAir_T": original_df["AvgAir_T"].iloc[SEQ_SIZE:].to_numpy(),
                "AvgRH": original_df["AvgRH"].iloc[SEQ_SIZE:].to_numpy(),
                "LatDD": original_df["LatDD"].iloc[SEQ_SIZE:].to_numpy(),
                "LongDD": original_df["LongDD"].iloc[SEQ_SIZE:].to_numpy(),
                "Elevation": original_df["Elevation"].iloc[SEQ_SIZE:].to_numpy(),
                "Split": split_map[station_name],
            }
        )

        if "Groundwater" in original_df.columns:
            prediction_df["Groundwater"] = original_df["Groundwater"].iloc[SEQ_SIZE:].to_numpy()

        output_path = RESULT_DIR / f"Results_{station_name}.csv"
        prediction_df.to_csv(output_path, index=False)
        print(f"Saved station predictions: {output_path}")


if __name__ == "__main__":
    main()
