# Recognizing Start and Stop Positions

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# Paths
DATA_DIR    = Path(os.getcwd()) / 'classification_data'
RESULTS_DIR = Path(os.getcwd()) / 'A12_results'
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / 'not_cut_classification_data.csv'


# Joint and feature definitions

JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

# Problem A — Kinect 3D: x,y,z 39 features
KINECT_COLS = []
for j in JOINTS:
    KINECT_COLS += [f'{j}_x', f'{j}_y', f'{j}_z']

# Problem B — PoseNet 2D: x,y  26 features
POSENET_COLS = []
for j in JOINTS:
    POSENET_COLS += [f'{j}_x', f'{j}_y']


EXTRA_COLS = [
    'left_hand_to_left_shoulder', 'right_hand_to_right_shoulder',
    'left_hand_to_left_hip', 'right_hand_to_right_hip',
    'left_elbow_to_left_shoulder', 'right_elbow_to_right_shoulder',
    'head_to_hip',
    'head_vx', 'head_vy', 'head_vz', 'head_speed',
    'left_hand_vx', 'left_hand_vy', 'left_hand_vz', 'left_hand_speed',
    'right_hand_vx', 'right_hand_vy', 'right_hand_vz', 'right_hand_speed',
    'head_ax', 'head_ay', 'head_az', 'head_accel',
    'left_hand_ax', 'left_hand_ay', 'left_hand_az', 'left_hand_accel',
    'right_hand_ax', 'right_hand_ay', 'right_hand_az', 'right_hand_accel',
]

# Columns NOT used as features
META_COLS = ['FrameNo', 'file_id', 'is_not_cut', 'label']

# Label mapping
LABEL_MAP = {'neutral': 0, 'start': 1, 'stop': 2}
LABEL_NAMES = ['neutral', 'start', 'stop']


# Load and examine data

print("\nLoading data")
raw_df = pd.read_csv(str(CSV_PATH))
raw_df.columns = raw_df.columns.str.strip()

print(f"Total frames: {len(raw_df)}")
print(f"\nLabel distribution:")
label_counts = raw_df['label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(raw_df) * 100
    print(f"  {label:10s}: {count:6d} ({pct:.2f}%)")

# Convert string labels to integers
raw_df['label_int'] = raw_df['label'].map(LABEL_MAP)

print("Transforming to Binary: Exercise (1) vs Non-Exercise (0)...")
raw_df['label_binary'] = 0
for file_id in raw_df['file_id'].unique():
    mask = raw_df['file_id'] == file_id
    # Find the rows where this specific video starts and stops
    start_idx = raw_df.index[mask & (raw_df['label'] == 'start')]
    stop_idx = raw_df.index[mask & (raw_df['label'] == 'stop')]
    
    if len(start_idx) > 0 and len(stop_idx) > 0:
        # Mark every frame BETWEEN start and stop as 1
        raw_df.loc[start_idx[0] : stop_idx[-1], 'label_binary'] = 1

# Overwrite the target variable to use our new binary labels
raw_df['label_int'] = raw_df['label_binary']
LABEL_NAMES = ['non-exercise', 'exercise']

# Prepare features for Problem A and B

def prepare_features(df, problem='A', use_extra=True):
    if problem == 'A':
        base_cols = KINECT_COLS      # 39 features
    else:
        base_cols = POSENET_COLS     

    feat_cols = [c for c in base_cols if c in df.columns]

    if use_extra:
        extra = [c for c in EXTRA_COLS if c in df.columns]
        feat_cols = feat_cols + extra

    X = df[feat_cols].values.astype(np.float32)
    y = df['label_int'].values.astype(np.int32)

    print(f"\nProblem {problem}:")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Samples : {len(X)}")

    return X, y, feat_cols


# Prepare both problems
X_A, y_A, cols_A = prepare_features(raw_df, problem='A', use_extra=True)
X_B, y_B, cols_B = prepare_features(raw_df, problem='B', use_extra=True)

# Split train/test 

def initial_split(X, y, test_size=0.1, random_state=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   
    )
    print(f"Train+Val: {len(X_trainval)} frames")
    print(f"Test     : {len(X_test)} frames")
    return X_trainval, X_test, y_trainval, y_test

print("\nSplitting Problem A:")
X_A_tv, X_A_test, y_A_tv, y_A_test = initial_split(X_A, y_A)

print("\nSplitting Problem B:")
X_B_tv, X_B_test, y_B_tv, y_B_test = initial_split(X_B, y_B)


# Resampling 

def create_sequences(X, y, window_size=30):
    Xs, ys = [], []
    for i in range(len(X) - window_size + 1):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size - 1])
    
    return np.array(Xs), np.array(ys)
    
    
def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    print(f"\nClass weights (higher = rarer = more important):")
    for cls, w in weight_dict.items():
        print(f"  {LABEL_NAMES[cls]:10s} (class {cls}): {w:.2f}x")
    return weight_dict


def oversample_minority(X, y, random_state=42):
    rng = np.random.default_rng(random_state)

    n_neutral = (y == 0).sum()
    n_start   = (y == 1).sum()
    n_stop    = (y == 2).sum()

    print(f"\nBefore oversampling: neutral={n_neutral} start={n_start} stop={n_stop}")

    # Oversample start frames to match neutral count
    start_idx = np.where(y == 1)[0]
    stop_idx  = np.where(y == 2)[0]

    n_copies_start = n_neutral // n_start
    n_copies_stop  = n_neutral // n_stop

    # Repeat indices
    start_oversampled = np.tile(start_idx, n_copies_start)
    stop_oversampled  = np.tile(stop_idx,  n_copies_stop)

    # Combine with original data
    all_idx = np.concatenate([
        np.where(y == 0)[0],   # all neutral frames
        start_oversampled,      # repeated start frames
        stop_oversampled        # repeated stop frames
    ])

    # Shuffle
    rng.shuffle(all_idx)

    X_resampled = X[all_idx]
    y_resampled = y[all_idx]

    n_neutral_new = (y_resampled == 0).sum()
    n_start_new   = (y_resampled == 1).sum()
    n_stop_new    = (y_resampled == 2).sum()

    print(f"After oversampling:  neutral={n_neutral_new} "
          f"start={n_start_new} stop={n_stop_new}")

    return X_resampled, y_resampled


# Compute class weights for training
class_weights_A = compute_class_weights(y_A_tv)
class_weights_B = compute_class_weights(y_B_tv)


# Define network architectures

def build_dense(input_dim, hidden_units=(128, 64),
                activation='relu', dropout_rate=0.2,
                l2_reg=1e-4, n_classes=2, name='Dense'):
    
    inputs = keras.Input(shape=(input_dim,), name='input')
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'drop_{i+1}')(x)

    # softmax = probability per class (must sum to 1)
    outputs = layers.Dense(n_classes, activation='softmax',
                           name='output')(x)
    return keras.Model(inputs, outputs, name=name)


def build_conv1d(input_dim, window_size=30,
                 filters=(64, 128), kernel_size=3,
                 dense_units=(64,), dropout_rate=0.2,
                 n_classes=2, name='Conv1D'):
    inputs = keras.Input(shape=(window_size, input_dim), name='input')
    x = inputs

    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size, activation='relu',
                          padding='same', name=f'conv_{i+1}')(x)
        x = layers.MaxPooling1D(2, padding='same', name=f'pool_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_conv_{i+1}')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'fc_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)

    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    return keras.Model(inputs, outputs, name=name)


def build_lstm(input_dim, window_size=30,
               lstm_units=(64, 32), dense_units=(32,),
               dropout_rate=0.2, n_classes=2, name='LSTM'):
    inputs = keras.Input(shape=(window_size, input_dim), name='input')
    x = inputs
    for i, u in enumerate(lstm_units):
        rs = (i < len(lstm_units) - 1)
        x = layers.LSTM(u, return_sequences=rs,
                        dropout=dropout_rate, name=f'lstm_{i+1}')(x)
    for i, u in enumerate(dense_units):
        x = layers.Dense(u, activation='relu', name=f'fc_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_{i+1}')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    return keras.Model(inputs, outputs, name=name)


def build_gru(input_dim, window_size=30,
              gru_units=(64, 32), dense_units=(32,),
              dropout_rate=0.2, n_classes=2, name='GRU'):
    inputs = keras.Input(shape=(window_size, input_dim), name='input')
    x = inputs
    for i, u in enumerate(gru_units):
        rs = (i < len(gru_units) - 1)
        x = layers.GRU(u, return_sequences=rs,
                       dropout=dropout_rate, name=f'gru_{i+1}')(x)
    for i, u in enumerate(dense_units):
        x = layers.Dense(u, activation='relu', name=f'fc_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_{i+1}')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    return keras.Model(inputs, outputs, name=name)


# Compile the model

def compile_model(model, optimizer='adam', lr=1e-3):
    opt_map = {
        'adam':    keras.optimizers.Adam(learning_rate=lr),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=lr),
        'sgd':     keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    }
    opt = opt_map.get(optimizer, keras.optimizers.Adam(lr))

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    return model


# Cross-Validation

def run_10fold_cv(X_trainval, y_trainval, X_test, y_test,
                  build_fn, input_dim, optimizer, batch_size,
                  class_weights, problem_name, arch_name,
                  use_oversampling=True, n_folds=3):
    run_name = f"{problem_name}_{arch_name}_{optimizer}_bs{batch_size}"
    print(f"  {run_name}  ({n_folds}-fold CV)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    best_val_f1  = -1
    best_model   = None
    best_scaler  = None

    for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(X_trainval, y_trainval)):

        print(f"\n  Fold {fold_idx+1}/{n_folds}", end='  ')

        X_tr, X_val = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[tr_idx], y_trainval[val_idx]

        if use_oversampling:
            X_tr, y_tr = oversample_minority(X_tr, y_tr,
                                             random_state=42 + fold_idx)

        is_seq = any(n in arch_name for n in ['Conv1D', 'LSTM', 'GRU'])
        
        if is_seq:
            X_tr, y_tr = create_sequences(X_tr, y_tr, window_size=30)
            X_val, y_val = create_sequences(X_val, y_val, window_size=30)

            scaler = StandardScaler()
            N, W, F = X_tr.shape
            X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(N, W, F)
            X_val_sc = scaler.transform(X_val.reshape(-1, F)).reshape(-1, W, F)
        else:
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)

        model = build_fn(input_dim)
        model = compile_model(model, optimizer=optimizer, lr=1e-3)

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_tr_sc, y_tr,
            validation_data=(X_val_sc, y_val),
            epochs=100,
            batch_size=batch_size,
            class_weight=class_weights,  # handle imbalance
            callbacks=[early_stop],
            verbose=0
        )

        y_pred = np.argmax(model.predict(X_val_sc, verbose=0), axis=1)

        # Calculate metrics for the 'Exercise' class (1)
        f1_ex = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        p_ex = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        r_ex = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

        print(f"EXERCISE F1={f1_ex:.3f} P={p_ex:.3f} R={r_ex:.3f} epochs={len(history.history['loss'])}")

        fold_metrics.append({
            'fold': fold_idx + 1,
            'f1': f1_ex, 'p': p_ex, 'r': r_ex,
            'epochs': len(history.history['loss']),
        })

        if f1_ex > best_val_f1:
            best_val_f1 = f1_ex
            best_model  = model
            best_scaler = scaler

    fold_df = pd.DataFrame(fold_metrics)
    avg = fold_df.mean(numeric_only=True)

    print(f"\n  {n_folds}-FOLD AVERAGE:")
    print(f"  EXERCISE -> P:{avg['p']:.3f} R:{avg['r']:.3f} F1:{avg['f1']:.3f}")
    print(f"  Avg epochs: {avg['epochs']:.1f}")


    print(f"\n  FINAL TEST SET EVALUATION (held-out 10%):")
    # 1. Prepare test data 
    X_test_final = X_test
    y_test_final = y_test
    
    # Check if the BEST model was a sequence model
    is_seq = any(n in arch_name for n in ['Conv1D', 'LSTM', 'GRU'])

    if is_seq:
        # Turn test data into sequences
        X_test_final, y_test_final = create_sequences(X_test, y_test, window_size=30)
        # Scale 3D data (Flatten -> Transform -> Reshape)
        Nt, Wt, Ft = X_test_final.shape
        X_test_sc = best_scaler.transform(X_test_final.reshape(-1, Ft)).reshape(Nt, Wt, Ft)
    else:
        # Standard 2D scaling
        X_test_sc = best_scaler.transform(X_test_final).astype(np.float32)

    # 2. Predict using the best model
    y_pred_test = np.argmax(best_model.predict(X_test_sc, verbose=0), axis=1)

    # 3. Report
    print(classification_report(
        y_test_final, y_pred_test,
        target_names=LABEL_NAMES, digits=3
    ))

    assess_frame_offset(y_test_final, y_pred_test, problem_name)

    # 4. Save best model
    best_model.save_weights(str(RESULTS_DIR / f'{run_name}.weights.h5'))
    
    import joblib
    joblib.dump(best_scaler, str(RESULTS_DIR / f'{run_name}_scaler.pkl'))

    return {
        'run':          run_name,
        'problem':      problem_name,
        'architecture': arch_name,
        'optimizer':    optimizer,
        'batch_size':   batch_size,
        'avg_f1':       avg['f1'],
        'avg_p':        avg['p'],
        'avg_r':        avg['r'],
        'avg_epochs':   avg['epochs'],
        'test_f1':      f1_score(y_test_final, y_pred_test, pos_label=1, zero_division=0)
    }


#  Frame offset analysis

def assess_frame_offset(y_true, y_pred, problem_name):
    true_starts = [i for i in range(1, len(y_true)) if y_true[i-1]==0 and y_true[i]==1]
    pred_starts = [i for i in range(1, len(y_pred)) if y_pred[i-1]==0 and y_pred[i]==1]
    true_stops  = [i for i in range(1, len(y_true)) if y_true[i-1]==1 and y_true[i]==0]
    pred_stops  = [i for i in range(1, len(y_pred)) if y_pred[i-1]==1 and y_pred[i]==0]

    print(f"\n  Frame Offset ({problem_name}):")
    for label, tp, pp in [('START', true_starts, pred_starts),
                           ('STOP',  true_stops,  pred_stops)]:
        pp = np.array(pp)
        if len(pp) > 0 and len(tp) > 0:
            offs = [abs(t - pp[np.argmin(np.abs(pp - t))]) for t in tp]
            avg  = np.mean(offs)
            print(f"    {label}: avg={avg:.1f} frames = {avg/30:.2f}s at 30fps")
        else:
            print(f"    {label}: {len(tp)} true, {len(pp)} predicted")


# Training configurations to test


# Architecture variants 
def make_architectures(input_dim):
    return {
        'Dense_relu':   lambda d: build_dense(
            d, hidden_units=(128, 64), activation='relu', dropout_rate=0.2, n_classes=2),
        'Conv1D':       lambda d: build_conv1d(
            d, window_size=30, filters=(64, 128), dropout_rate=0.2, n_classes=2),
        'LSTM':         lambda d: build_lstm(
            d, window_size=30, lstm_units=(64, 32), dropout_rate=0.2, n_classes=2),
        'GRU':          lambda d: build_gru(
            d, window_size=30, gru_units=(64, 32), dropout_rate=0.2, n_classes=2),
    }

OPTIMIZERS  = ['adam', 'rmsprop']  
BATCH_SIZES = [32, 64]              

all_results = []


# Problem A (Kinect 39 features) and Problem B (PoseNet 26 features)

print("PROBLEM A: Kinect (x,y,z) - 39 features")

archs_A = make_architectures(X_A.shape[1])

for arch_name, build_fn in archs_A.items():
    for opt in OPTIMIZERS:
        for bs in BATCH_SIZES:
            result = run_10fold_cv(
                X_trainval=X_A_tv, y_trainval=y_A_tv,
                X_test=X_A_test,   y_test=y_A_test,
                build_fn=build_fn, input_dim=X_A.shape[1],
                optimizer=opt, batch_size=bs,
                class_weights=class_weights_A,
                problem_name='A_Kinect', arch_name=arch_name,
                use_oversampling=False, n_folds=3
            )
            all_results.append(result)


print("PROBLEM B: PoseNet (x,y) — 26 features")

archs_B = make_architectures(X_B.shape[1])

for arch_name, build_fn in archs_B.items():
    for opt in OPTIMIZERS:
        for bs in BATCH_SIZES:
            result = run_10fold_cv(
                X_trainval=X_B_tv, y_trainval=y_B_tv,
                X_test=X_B_test,   y_test=y_B_test,
                build_fn=build_fn, input_dim=X_B.shape[1],
                optimizer=opt, batch_size=bs,
                class_weights=class_weights_B,
                problem_name='B_PoseNet', arch_name=arch_name,
                use_oversampling=False, n_folds=3
            )
            all_results.append(result)


# Final results table and comparison

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('avg_f1', ascending=False)

print(f"\n{'='*70}")
print("FINAL RESULTS — All experiments sorted by START F1")
print(f"{'='*70}")
print(results_df[['problem','architecture','optimizer','batch_size',
            'avg_f1','avg_p','avg_r','test_f1','avg_epochs']].to_string(index=False))

results_df.to_csv(str(RESULTS_DIR / 'all_results.csv'), index=False)

# Best per problem
best_A = results_df[results_df['problem'] == 'A_Kinect'].iloc[0]
best_B = results_df[results_df['problem'] == 'B_PoseNet'].iloc[0]

print(f"\n{'='*60}")
print("COMPARISON: Problem A (Kinect) vs Problem B (PoseNet)")
print(f"{'='*60}")
print(f"{'Metric':<28} {'Problem A':>12} {'Problem B':>12}")
print('-' * 55)
print(f"{'Input features':<28} {X_A.shape[1]:>12} {X_B.shape[1]:>12}")
print(f"{'Best arch':<28} {best_A['architecture']:>12} {best_B['architecture']:>12}")
print(f"{'Best optimizer':<28} {best_A['optimizer']:>12} {best_B['optimizer']:>12}")
print(f"{'CV F1 Score':<28} {best_A['avg_f1']:>12.3f} {best_B['avg_f1']:>12.3f}")
print(f"{'Test F1 Score':<28} {best_A['test_f1']:>12.3f} {best_B['test_f1']:>12.3f}")

diff = best_A['avg_f1'] - best_B['avg_f1']
if diff > 0.02:
    conclusion = (f"Kinect (A) outperforms PoseNet (B) by {diff:.3f} F1. "
                  f"Depth (z) helps detect start/stop transitions.")
elif diff < -0.02:
    conclusion = (f"PoseNet (B) matches Kinect (A). "
                  f"2D coordinates are sufficient for start/stop detection.")
else:
    conclusion = (f"Similar performance (diff={diff:.3f}). "
                  f"Z coordinate adds minimal value for this task.")

print(f"\nConclusion: {conclusion}")
print(f"\nAll results saved to: {RESULTS_DIR}")
