import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

THIS_DIR        = Path(__file__).parent
PROJECT_ROOT    = THIS_DIR.parent
DATASETS_DIR    = PROJECT_ROOT / 'Datasets_all'
POSENET_DIR     = DATASETS_DIR / 'posenet_data'
KINECT_DIR      = DATASETS_DIR / 'kinect_good_preprocessed'
RESULTS_DIR     = THIS_DIR / 'one_step_results'
RESULTS_DIR.mkdir(exist_ok=True)

#  Constants 
JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot',
]
N_JOINTS    = len(JOINTS)    # 13
N_INPUT     = N_JOINTS * 2  # 26  — PoseNet xpn,ypn
N_OUTPUT    = N_JOINTS * 3  # 39  — Kinect xk,yk,zk
WINDOW_SIZE = 30

print(f'One-Step Network')
print(f'  Input  (PoseNet x,y)     : {N_INPUT}')
print(f'  Output (Kinect x,y,z)    : {N_OUTPUT}')
print(f'  Joints                   : {N_JOINTS}')
print(f'  Window size              : {WINDOW_SIZE}')


#  DATA LOADING

def load_single_pair(posenet_csv, kinect_csv):
    pn_df = pd.read_csv(str(posenet_csv))
    k_df  = pd.read_csv(str(kinect_csv))
    pn_df.columns = pn_df.columns.str.strip()
    k_df.columns  = k_df.columns.str.strip()

    # PoseNet input: x,y for each joint -> 26 values
    pn_cols = []
    for j in JOINTS:
        pn_cols += [f'{j}_x', f'{j}_y']
    X = pn_df[pn_cols].values.astype(np.float32)      # (n_frames, 26)

    # Kinect target: x,y,z for each joint -> 39 values
    k_cols = []
    for j in JOINTS:
        k_cols += [f'{j}_x', f'{j}_y', f'{j}_z']
    y = k_df[k_cols].values.astype(np.float32)         # (n_frames, 39)

    return X, y


def load_all_sequences(posenet_dir, kinect_dir):
    posenet_files = set(f.name for f in Path(posenet_dir).glob('*.csv'))
    kinect_files  = set(f.name for f in Path(kinect_dir).glob('*.csv'))
    common        = sorted(posenet_files & kinect_files)

    print(f'Found {len(common)} aligned PoseNet+Kinect file pairs')

    sequences, file_names = [], []
    skipped = 0

    for name in common:
        pn_path = Path(posenet_dir) / name
        k_path  = Path(kinect_dir)  / name
        X, y = load_single_pair(pn_path, k_path)

        # Skip if frame counts differ too much
        if abs(len(X) - len(y)) > 5:
            print(f'  Skipping {name}: frame mismatch ({len(X)} vs {len(y)})')
            skipped += 1
            continue

        # Use shorter length if slight mismatch
        n = min(len(X), len(y))
        sequences.append((X[:n], y[:n]))
        file_names.append(name)

    print(f'Loaded: {len(sequences)} sessions  |  Skipped: {skipped}')
    return sequences, file_names


def split_sessions(sequences, file_names, test_size=0.2, random_state=42):
    all_idx = list(range(len(sequences)))
    tr_idx, te_idx = train_test_split(all_idx, test_size=test_size,
                                      random_state=random_state, shuffle=True)
    tr_seq   = [sequences[i]  for i in tr_idx]
    te_seq   = [sequences[i]  for i in te_idx]
    tr_files = [file_names[i] for i in tr_idx]
    te_files = [file_names[i] for i in te_idx]
    print(f'Split: {len(tr_seq)} train sessions, {len(te_seq)} test sessions')
    return tr_seq, te_seq, tr_files, te_files


def flatten(sessions):
    X = np.concatenate([s[0] for s in sessions], axis=0)
    y = np.concatenate([s[1] for s in sessions], axis=0)
    return X, y


def make_windows(sessions, window_size=WINDOW_SIZE, stride=5):
    X_list, y_list = [], []
    for X, y in sessions:
        n = len(X)
        for start in range(0, n - window_size + 1, stride):
            X_list.append(X[start:start+window_size])
            y_list.append(y[start+window_size-1])   # predict last frame's xyz
    X_seq = np.array(X_list, dtype=np.float32)      # (N, window, 26)
    y_seq = np.array(y_list, dtype=np.float32)      # (N, 39)
    return X_seq, y_seq


# LOAD AND PREPARE DATA

print(f'\nLoading data from:')
print(f'  PoseNet: {POSENET_DIR}')
print(f'  Kinect : {KINECT_DIR}')

sequences, file_names = load_all_sequences(POSENET_DIR, KINECT_DIR)

if len(sequences) == 0:
    print('\nERROR: No data loaded.')
    print('Run generate_posenet_data.py first to create PoseNet CSV files.')
    sys.exit(1)

# Split by session first (no leakage)
tr_seq, te_seq, tr_files, te_files = split_sessions(
    sequences, file_names, test_size=0.2, random_state=42)

# Flat data for Dense model
X_flat_tr, y_flat_tr = flatten(tr_seq)
X_flat_te, y_flat_te = flatten(te_seq)
print(f'\nFlat — Train: X={X_flat_tr.shape} y={y_flat_tr.shape}')
print(f'Flat — Test : X={X_flat_te.shape} y={y_flat_te.shape}')

# Windowed data for Conv1D/LSTM/GRU
X_seq_tr, y_seq_tr = make_windows(tr_seq)
X_seq_te, y_seq_te = make_windows(te_seq)
print(f'Seq  — Train: X={X_seq_tr.shape} y={y_seq_tr.shape}')
print(f'Seq  — Test : X={X_seq_te.shape} y={y_seq_te.shape}')

# Normalisation — fit on train only
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_flat_tr_sc = scaler_X.fit_transform(X_flat_tr)
y_flat_tr_sc = scaler_y.fit_transform(y_flat_tr)
X_flat_te_sc = scaler_X.transform(X_flat_te)
y_flat_te_sc = scaler_y.transform(y_flat_te)

n_tr, w, f = X_seq_tr.shape
n_te        = X_seq_te.shape[0]
X_seq_tr_sc = scaler_X.transform(X_seq_tr.reshape(-1, N_INPUT)).reshape(n_tr, w, f)
X_seq_te_sc = scaler_X.transform(X_seq_te.reshape(-1, N_INPUT)).reshape(n_te, w, f)
y_seq_tr_sc = scaler_y.transform(y_seq_tr)
y_seq_te_sc = scaler_y.transform(y_seq_te)

print(f'\nNormalisation done.')


# MODEL ARCHITECTURES

def build_dense(hidden_units=(128, 64), activation='relu',
                dropout_rate=0.2, l2_reg=1e-4):
    inp = keras.Input(shape=(N_INPUT,), name='posenet_xy')
    x = inp
    for i, u in enumerate(hidden_units):
        x = layers.Dense(u, activation=activation,
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
                         name=f'dense_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_{i+1}')(x)
    out = layers.Dense(N_OUTPUT, activation='linear', name='xyz_out')(x)
    return keras.Model(inp, out, name='Dense_1Step')


def build_conv1d(filters=(64, 128), kernel_size=3, pool_size=2,
                 dense_units=(64,), activation='relu', dropout_rate=0.2):
    inp = keras.Input(shape=(WINDOW_SIZE, N_INPUT), name='posenet_seq')
    x = inp
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size, activation=activation,
                          padding='same', name=f'conv_{i+1}')(x)
        x = layers.MaxPooling1D(pool_size, padding='same', name=f'pool_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_conv_{i+1}')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    for i, u in enumerate(dense_units):
        x = layers.Dense(u, activation=activation, name=f'fc_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)
    out = layers.Dense(N_OUTPUT, activation='linear', name='xyz_out')(x)
    return keras.Model(inp, out, name='Conv1D_1Step')


def build_lstm(lstm_units=(64, 32), dense_units=(32,),
               dropout_rate=0.2, recurrent_dropout=0.0):
    inp = keras.Input(shape=(WINDOW_SIZE, N_INPUT), name='posenet_seq')
    x = inp
    for i, u in enumerate(lstm_units):
        rs = (i < len(lstm_units) - 1)
        x = layers.LSTM(u, return_sequences=rs, dropout=dropout_rate,
                        recurrent_dropout=recurrent_dropout,
                        name=f'lstm_{i+1}')(x)
    for i, u in enumerate(dense_units):
        x = layers.Dense(u, activation='relu', name=f'fc_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)
    out = layers.Dense(N_OUTPUT, activation='linear', name='xyz_out')(x)
    return keras.Model(inp, out, name='LSTM_1Step')


def build_gru(gru_units=(64, 32), dense_units=(32,),
              dropout_rate=0.2, recurrent_dropout=0.0):
    inp = keras.Input(shape=(WINDOW_SIZE, N_INPUT), name='posenet_seq')
    x = inp
    for i, u in enumerate(gru_units):
        rs = (i < len(gru_units) - 1)
        x = layers.GRU(u, return_sequences=rs, dropout=dropout_rate,
                       recurrent_dropout=recurrent_dropout,
                       name=f'gru_{i+1}')(x)
    for i, u in enumerate(dense_units):
        x = layers.Dense(u, activation='relu', name=f'fc_{i+1}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'drop_fc_{i+1}')(x)
    out = layers.Dense(N_OUTPUT, activation='linear', name='xyz_out')(x)
    return keras.Model(inp, out, name='GRU_1Step')


# COMPILE AND TRAIN

def compile_model(model, optimizer='adam', loss='mse', lr=1e-3):
    opt_map = {
        'adam':    keras.optimizers.Adam(learning_rate=lr),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=lr),
        'sgd':     keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    }
    model.compile(
        optimizer=opt_map.get(optimizer, keras.optimizers.Adam(lr)),
        loss=loss,
        metrics=[keras.metrics.MeanAbsoluteError(name='mae'),
                 keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model


def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=100, batch_size=32, patience=10):
    cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience,
        restore_best_weights=True, verbose=1)
    return model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size,
                     callbacks=[cb], verbose=1)


# EVALUATION

def evaluate_xyz(model, X_te_sc, y_te_sc, name='Model'):
    pred_sc = model.predict(X_te_sc, verbose=0)
    pred    = scaler_y.inverse_transform(pred_sc)   # back to metres
    true    = scaler_y.inverse_transform(y_te_sc)

    # Split 39 columns into x, y, z groups
    px, py, pz = pred[:, 0::3], pred[:, 1::3], pred[:, 2::3]
    tx, ty, tz = true[:, 0::3], true[:, 1::3], true[:, 2::3]

    mae_x = np.mean(np.abs(px - tx))
    mae_y = np.mean(np.abs(py - ty))
    mae_z = np.mean(np.abs(pz - tz))
    mae_o = np.mean(np.abs(pred - true))

    print(f'\n── {name} — Coordinate Errors ───────────────')
    print(f'  MAE X (left/right) : {mae_x*100:.2f} cm')
    print(f'  MAE Y (up/down)    : {mae_y*100:.2f} cm')
    print(f'  MAE Z (depth)      : {mae_z*100:.2f} cm')
    print(f'  MAE overall        : {mae_o*100:.2f} cm')

    print(f'\n── Per-Joint Error (cm) ──────────────────────')
    print(f'{"Joint":<20} {"X":>7} {"Y":>7} {"Z":>7} {"Avg":>7}')
    print('─' * 47)
    per_joint = {}
    for ji, jn in enumerate(JOINTS):
        ex = np.mean(np.abs(px[:, ji] - tx[:, ji])) * 100
        ey = np.mean(np.abs(py[:, ji] - ty[:, ji])) * 100
        ez = np.mean(np.abs(pz[:, ji] - tz[:, ji])) * 100
        ea = (ex + ey + ez) / 3
        per_joint[jn] = {'x': ex, 'y': ey, 'z': ez, 'avg': ea}
        print(f'{jn:<20} {ex:>6.2f}  {ey:>6.2f}  {ez:>6.2f}  {ea:>6.2f}')

    avg = np.mean([v['avg'] for v in per_joint.values()])
    print(f'\n  Average across all joints: {avg:.3f} cm')
    return {'mae_x': mae_x, 'mae_y': mae_y, 'mae_z': mae_z,
            'mae_overall': mae_o, 'per_joint': per_joint}


def show_conversion_example(model, X_te_sc, y_te_sc, n=3):
    X_orig = scaler_X.inverse_transform(X_te_sc[:n])
    pred   = scaler_y.inverse_transform(model.predict(X_te_sc[:n], verbose=0))
    true   = scaler_y.inverse_transform(y_te_sc[:n])

    print(f'\n── Sample Conversion: (xpn,ypn) -> (xk,yk,zk) ──')
    print(f'Showing head joint only\n')
    for i in range(n):
        print(f'Frame {i+1}:')
        print(f'  INPUT  PoseNet  head_xpn={X_orig[i,0]:.4f}  head_ypn={X_orig[i,1]:.4f}')
        print(f'  PRED   Kinect   head_xk={pred[i,0]:.3f}m  '
              f'head_yk={pred[i,1]:.3f}m  head_zk={pred[i,2]:.3f}m')
        print(f'  ACTUAL Kinect   head_xk={true[i,0]:.3f}m  '
              f'head_yk={true[i,1]:.3f}m  head_zk={true[i,2]:.3f}m')
        print()


def plot_training(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title(f'{name} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(history.history['mae'],     label='Train')
    axes[1].plot(history.history['val_mae'], label='Val')
    axes[1].set_title(f'{name} — MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / f'{name}_training.png'), dpi=100)
    plt.close()
    print(f'  Plot saved: {name}_training.png')


# RUN ALL EXPERIMENTS

ARCHITECTURES = {
    'Dense_shallow': {'fn': build_dense,
                      'p':  {'hidden_units': (64,)},            'data': 'flat'},
    'Dense_deep':    {'fn': build_dense,
                      'p':  {'hidden_units': (256, 128, 64, 32)}, 'data': 'flat'},
    'Dense_wide':    {'fn': build_dense,
                      'p':  {'hidden_units': (512, 256)},        'data': 'flat'},
    'Conv1D':        {'fn': build_conv1d,
                      'p':  {'filters': (64, 128), 'dense_units': (64,)}, 'data': 'seq'},
    'LSTM':          {'fn': build_lstm,
                      'p':  {'lstm_units': (64, 32), 'dense_units': (32,)}, 'data': 'seq'},
    'GRU':           {'fn': build_gru,
                      'p':  {'gru_units': (64, 32), 'dense_units': (32,)},  'data': 'seq'},
}

OPTIMIZERS  = ['adam', 'rmsprop', 'sgd']
all_results = []

for arch, cfg in ARCHITECTURES.items():
    for opt in OPTIMIZERS:
        run = f'{arch}_{opt}'
        print(f'\n{"="*60}')
        print(f'  {run}')
        print(f'{"="*60}')

        if cfg['data'] == 'flat':
            Xtr, ytr = X_flat_tr_sc, y_flat_tr_sc
            Xte, yte = X_flat_te_sc, y_flat_te_sc
        else:
            Xtr, ytr = X_seq_tr_sc, y_seq_tr_sc
            Xte, yte = X_seq_te_sc, y_seq_te_sc

        # 10% of train as validation
        sp      = int(len(Xtr) * 0.9)
        Xv, yv  = Xtr[sp:], ytr[sp:]
        Xtr, ytr = Xtr[:sp], ytr[:sp]

        model   = cfg['fn'](**cfg['p'])
        model   = compile_model(model, optimizer=opt, loss='mse', lr=1e-3)
        history = train_model(model, Xtr, ytr, Xv, yv,
                              epochs=100, batch_size=32, patience=10)
        plot_training(history, run)
        results = evaluate_xyz(model, Xte, yte, run)

        # Show conversion example for first model only
        if arch == 'Dense_shallow' and opt == 'adam':
            show_conversion_example(model, Xte, yte)

        # Save weights
        model.save_weights(str(RESULTS_DIR / f'{run}.weights.h5'))

        all_results.append({
            'model': arch, 'optimizer': opt,
            'mae_x_cm':       results['mae_x']       * 100,
            'mae_y_cm':       results['mae_y']       * 100,
            'mae_z_cm':       results['mae_z']       * 100,
            'mae_overall_cm': results['mae_overall'] * 100,
            'epochs':         len(history.history['loss']),
        })


#  FINAL RESULTS TABLE

df = pd.DataFrame(all_results).sort_values('mae_overall_cm')
print(f'\n{"="*65}')
print('FINAL RESULTS — sorted by overall MAE')
print(f'{"="*65}')
print(df.to_string(index=False))
df.to_csv(str(RESULTS_DIR / 'results_summary.csv'), index=False)

best = df.iloc[0]
print(f'\nBEST MODEL : {best["model"]} + {best["optimizer"]}')
print(f'  MAE X    : {best["mae_x_cm"]:.2f} cm')
print(f'  MAE Y    : {best["mae_y_cm"]:.2f} cm')
print(f'  MAE Z    : {best["mae_z_cm"]:.2f} cm  (depth)')
print(f'  Overall  : {best["mae_overall_cm"]:.2f} cm')
