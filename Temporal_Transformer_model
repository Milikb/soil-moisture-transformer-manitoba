"""
Temporal Transformer model for soil moisture prediction.

This script trains a Transformer model using a temporal 60/20/20 split
within each station. It is configured for the daily model by default.
Change TEMPORAL_RESOLUTION, DATA_DIR, RESULT_DIR, and TARGET_COL as needed.

Look-back windows:
    hourly: 96 time steps
    daily : 45 time steps
    weekly: 10 time steps
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/home/miladkb/projects/Station_based_hourly/5_cm_daily/")
RESULT_DIR = Path("/home/miladkb/projects/Paper/Results/Temporal_Transformer/daily_5cm")

TEMPORAL_RESOLUTION = "daily"  # options: "hourly", "daily", "weekly"
SEQUENCE_LENGTHS: Dict[str, int] = {
    "hourly": 96,
    "daily": 45,
    "weekly": 10,
}
SEQ_SIZE = SEQUENCE_LENGTHS[TEMPORAL_RESOLUTION]

TARGET_COL = "Soil_TP5_VMC"

DYNAMIC_COLS = [
    "Pluvio_Rain",
    "AvgAir_T",
    "AvgRH",
    "Day",
    "Day_sin",
    "Day_cos",
]

METADATA_COLS = [
    "Pluvio_Rain",
    "AvgAir_T",
    "AvgRH",
    "LatDD",
    "LongDD",
    "Elevation",
    "AQUIFER_NUM",
]

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20

D_MODEL = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_BLOCKS = 4
DROPOUT_RATE = 0.10
L2_REG = 1e-5

LEARNING_RATE = 1e-3
HUBER_DELTA = 0.15
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10

RANDOM_SEED = 42


# =============================================================================
# Reproducibility and output directory
# =============================================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year and cyclic seasonal features."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.dayofyear
    df["Day_sin"] = np.sin(2.0 * np.pi * df["Day"] / 365.0)
    df["Day_cos"] = np.cos(2.0 * np.pi * df["Day"] / 365.0)
    return df


def validate_columns(df: pd.DataFrame, required_cols: List[str], file_path: Path) -> None:
    """Stop execution if a station file is missing required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path.name}: {', '.join(missing)}"
        )


def load_station_file(
    file_path: Path,
    dynamic_cols: List[str],
    target_col: str,
) -> Dict[str, object]:
    """Load one station CSV file and return target and dynamic predictors."""
    df = pd.read_csv(file_path)
    df = add_time_features(df)
    df = df.sort_values("Date").reset_index(drop=True)
    df["Station"] = file_path.stem

    required_cols = ["Date", target_col] + dynamic_cols
    validate_columns(df, required_cols, file_path)

    target = df[[target_col]].to_numpy(dtype=np.float32)
    dynamic = df[dynamic_cols].to_numpy(dtype=np.float32)

    return {
        "station": file_path.stem,
        "df": df,
        "target": target,
        "dynamic": dynamic,
    }


def load_all_stations(data_dir: Path) -> List[Dict[str, object]]:
    """Load all station CSV files from the input directory."""
    station_data: List[Dict[str, object]] = []

    for file_path in sorted(data_dir.glob("*.csv")):
        station = load_station_file(
            file_path=file_path,
            dynamic_cols=DYNAMIC_COLS,
            target_col=TARGET_COL,
        )

        if len(station["target"]) <= SEQ_SIZE:
            print(f"Skipping {file_path.name}: fewer than {SEQ_SIZE + 1} rows.")
            continue

        station_data.append(station)

    if not station_data:
        raise ValueError(f"No valid station files found in {data_dir}")

    return station_data


# =============================================================================
# Temporal split and scaling
# =============================================================================

def split_station_temporally(
    station_data: List[Dict[str, object]],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
           List[Tuple[np.ndarray, np.ndarray]],
           List[Tuple[np.ndarray, np.ndarray]]]:
    """Apply a 60/20/20 temporal split within each station."""
    train_list = []
    val_list = []
    test_list = []

    for station in station_data:
        target = station["target"]
        dynamic = station["dynamic"]

        n_total = len(target)
        train_end = int(TRAIN_FRACTION * n_total)
        val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n_total)

        train_list.append((target[:train_end], dynamic[:train_end]))
        val_list.append((target[train_end:val_end], dynamic[train_end:val_end]))
        test_list.append((target[val_end:], dynamic[val_end:]))

    return train_list, val_list, test_list


def fit_scalers(
    train_list: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit target and dynamic-feature scalers using training data only."""
    target_scaler = MinMaxScaler()
    dynamic_scaler = MinMaxScaler()

    train_target = np.vstack([target for target, _ in train_list])
    train_dynamic = np.vstack([dynamic for _, dynamic in train_list])

    target_scaler.fit(train_target)
    dynamic_scaler.fit(train_dynamic)

    joblib.dump(target_scaler, RESULT_DIR / "target_scaler.pkl")
    joblib.dump(dynamic_scaler, RESULT_DIR / "dynamic_scaler.pkl")

    return target_scaler, dynamic_scaler


def scale_station_list(
    station_list: List[Tuple[np.ndarray, np.ndarray]],
    target_scaler: MinMaxScaler,
    dynamic_scaler: MinMaxScaler,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Scale target and dynamic predictors for each station split."""
    scaled = []

    for target, dynamic in station_list:
        target_scaled = target_scaler.transform(target)
        dynamic_scaled = dynamic_scaler.transform(dynamic)
        scaled.append((target_scaled, dynamic_scaled))

    return scaled


# =============================================================================
# Sequence construction
# =============================================================================

def to_sequences(
    target: np.ndarray,
    dynamic: np.ndarray,
    seq_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create look-back sequences and one-step-ahead targets."""
    x_dynamic = []
    y = []
    target_indices = []

    for i in range(len(target) - seq_size):
        target_index = i + seq_size
        x_dynamic.append(dynamic[i:target_index])
        y.append(target[target_index, 0])
        target_indices.append(target_index)

    return (
        np.asarray(x_dynamic, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int32),
    )


def build_sequences_from_station_list(
    station_list: List[Tuple[np.ndarray, np.ndarray]],
    seq_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack sequences from all stations into one training/validation/test set."""
    x_list = []
    y_list = []

    for target, dynamic in station_list:
        x_dynamic, y, _ = to_sequences(target, dynamic, seq_size)
        if len(x_dynamic) > 0:
            x_list.append(x_dynamic)
            y_list.append(y)

    if not x_list:
        raise ValueError("No valid sequences were created.")

    return np.vstack(x_list), np.concatenate(y_list)


# =============================================================================
# Transformer model
# =============================================================================

def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """Create sinusoidal positional encoding."""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)

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
    dropout_rate: float,
) -> tf.Tensor:
    """Transformer encoder block with pre-normalization."""
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
    )(x_norm, x_norm)
    x = Add()([x, attention_output])

    x_norm = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = Dense(ff_dim, activation="gelu")(x_norm)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)
    x = Add()([x, ffn_output])

    return x


def build_temporal_transformer_model(
    seq_size: int,
    n_dynamic_features: int,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    num_blocks: int,
    dropout_rate: float,
) -> Model:
    """Build a temporal Transformer model for one-step-ahead prediction."""
    dynamic_input = Input(
        shape=(seq_size, n_dynamic_features),
        name="dynamic_input_sequence",
    )

    x = Dense(d_model, name="linear_projection")(dynamic_input)
    x = x + positional_encoding(seq_size, d_model)

    for _ in range(num_blocks):
        x = transformer_encoder_block(
            x=x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
        )

    x = GlobalAveragePooling1D(name="global_average_pooling")(x)
    x = Dense(128, activation="relu", name="temporal_dense")(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(L2_REG), name="dense_64")(x)
    x = Dropout(dropout_rate, name="dropout")(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(L2_REG), name="dense_32")(x)

    output = Dense(1, name="soil_moisture_prediction")(x)

    return Model(inputs=dynamic_input, outputs=output, name="Temporal_Transformer_Model")


# =============================================================================
# Training and prediction
# =============================================================================

def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> Tuple[Model, tf.keras.callbacks.History, float]:
    """Train the temporal Transformer model."""
    model = build_temporal_transformer_model(
        seq_size=SEQ_SIZE,
        n_dynamic_features=len(DYNAMIC_COLS),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        dropout_rate=DROPOUT_RATE,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=RESULT_DIR / "best_temporal_transformer_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start_time = time.time()

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
    )

    training_time_sec = time.time() - start_time

    return model, history, training_time_sec


def save_training_outputs(
    model: Model,
    history: tf.keras.callbacks.History,
    training_time_sec: float,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> None:
    """Save the model, training history, and training metadata."""
    final_model_path = RESULT_DIR / "temporal_transformer_model.keras"
    model.save(final_model_path)

    pd.DataFrame(history.history).to_csv(RESULT_DIR / "training_history.csv", index=False)

    test_loss, test_rmse = model.evaluate(test_x, test_y, verbose=0)

    training_metadata = {
        "model_name": "Temporal Transformer Model",
        "temporal_resolution": TEMPORAL_RESOLUTION,
        "seq_size": SEQ_SIZE,
        "target_col": TARGET_COL,
        "dynamic_cols": ", ".join(DYNAMIC_COLS),
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "ff_dim": FF_DIM,
        "num_blocks": NUM_BLOCKS,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "loss": f"Huber(delta={HUBER_DELTA})",
        "epochs_completed": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "training_time_sec": training_time_sec,
        "training_time_min": training_time_sec / 60.0,
        "test_loss_scaled": test_loss,
        "test_rmse_scaled": test_rmse,
    }

    pd.Series(training_metadata).to_csv(RESULT_DIR / "training_metadata.csv")
    print(f"Model saved to: {final_model_path}")


def predict_full_station_series(
    model: Model,
    station_data: List[Dict[str, object]],
    target_scaler: MinMaxScaler,
    dynamic_scaler: MinMaxScaler,
) -> None:
    """Generate full time-series predictions for each station."""
    prediction_dir = RESULT_DIR / "station_predictions"
    prediction_dir.mkdir(exist_ok=True)

    for station in station_data:
        station_name = station["station"]
        df = station["df"].copy()
        target = station["target"]
        dynamic = station["dynamic"]

        target_scaled = target_scaler.transform(target)
        dynamic_scaled = dynamic_scaler.transform(dynamic)

        x_dynamic, y_scaled, target_indices = to_sequences(
            target=target_scaled,
            dynamic=dynamic_scaled,
            seq_size=SEQ_SIZE,
        )

        if len(x_dynamic) == 0:
            print(f"Skipping {station_name}: no valid sequences.")
            continue

        pred_scaled = model.predict(x_dynamic, verbose=0).reshape(-1, 1)

        observed = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        predicted = target_scaler.inverse_transform(pred_scaled).flatten()

        n_total = len(target)
        train_end = int(TRAIN_FRACTION * n_total)
        val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n_total)

        split_labels = np.where(
            target_indices < train_end,
            "train",
            np.where(target_indices < val_end, "validation", "test"),
        )

        out_df = pd.DataFrame({
            "Station": station_name,
            "Date": df["Date"].iloc[target_indices].to_numpy(),
            "Split": split_labels,
            "Observed": observed,
            "Predicted": predicted,
        })

        available_metadata = [col for col in METADATA_COLS if col in df.columns]
        for col in available_metadata:
            out_df[col] = df[col].iloc[target_indices].to_numpy()

        out_path = prediction_dir / f"Results_{station_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved station predictions: {out_path}")


def main() -> None:
    """Run the temporal Transformer workflow."""
    print(f"Temporal resolution: {TEMPORAL_RESOLUTION}")
    print(f"Sequence length: {SEQ_SIZE}")
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {RESULT_DIR}")

    station_data = load_all_stations(DATA_DIR)

    train_raw, val_raw, test_raw = split_station_temporally(station_data)

    target_scaler, dynamic_scaler = fit_scalers(train_raw)

    train_scaled = scale_station_list(train_raw, target_scaler, dynamic_scaler)
    val_scaled = scale_station_list(val_raw, target_scaler, dynamic_scaler)
    test_scaled = scale_station_list(test_raw, target_scaler, dynamic_scaler)

    train_x, train_y = build_sequences_from_station_list(train_scaled, SEQ_SIZE)
    val_x, val_y = build_sequences_from_station_list(val_scaled, SEQ_SIZE)
    test_x, test_y = build_sequences_from_station_list(test_scaled, SEQ_SIZE)

    print(f"Training sequences: {train_x.shape}")
    print(f"Validation sequences: {val_x.shape}")
    print(f"Testing sequences: {test_x.shape}")

    model, history, training_time_sec = train_model(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
    )

    save_training_outputs(
        model=model,
        history=history,
        training_time_sec=training_time_sec,
        test_x=test_x,
        test_y=test_y,
    )

    predict_full_station_series(
        model=model,
        station_data=station_data,
        target_scaler=target_scaler,
        dynamic_scaler=dynamic_scaler,
    )


if __name__ == "__main__":
    main()
