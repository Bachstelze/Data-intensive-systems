import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import re

# Paths
CUT_DIR    = Path('A15_Data/a15_cut_augmented')    
SCORES_CSV = Path('A15_Data/a15_augmented_data.csv')
RESULTS_DIR = Path('A15_results')
RESULTS_DIR.mkdir(exist_ok=True)

JOINTS = [
    'head', 'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
    'left_hand', 'right_hand', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_foot', 'right_foot'
]

C = 10   


# Load and prepare data

def sample_frames(df, c=C):
    indices = np.linspace(0, len(df)-1, c).astype(int)
    sampled = df.iloc[indices]
    frames  = []
    for _, row in sampled.iterrows():
        joints = [[row[f'{j}_x'], row[f'{j}_y'], row[f'{j}_z']]
                  for j in JOINTS]
        frames.append(joints)
    return np.array(frames, dtype=np.float32)   



def load_dataset():
    scores_df = pd.read_csv(SCORES_CSV)
    scores_df.columns = scores_df.columns.str.strip()

    X, y, names = [], [], []

    for _, row in scores_df.iterrows():
        csv_path = CUT_DIR / f"{row['clip']}.csv"

        if not csv_path.exists():
            print(f"  Missing: {csv_path.name}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if len(df) < C:
            print(f"  Too short ({len(df)} frames): {csv_path.name}")
            continue

        frames = sample_frames(df)
        X.append(frames)
        y.append(float(row['score_rescaled']))  
        names.append(row['clip'])

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset loaded:")
    print(f"  Clips:        {len(X)}")
    print(f"  Score range:  {y.min():.2f} to {y.max():.2f}")
    print(f"  Score mean:   {y.mean():.2f} ± {y.std():.2f}")

    return X, y, names


X_raw, y, names = load_dataset()

X_flat = X_raw.reshape(len(X_raw), -1)
X_seq  = X_raw.reshape(len(X_raw), C, 13*3)

original_names = [re.sub(r'_(mirror|rotate_pos|rotate_neg|stretch)$', '', n)
                  for n in names]

unique_originals = list(set(original_names))
np.random.seed(42)
np.random.shuffle(unique_originals)

n_test      = max(1, int(len(unique_originals) * 0.1))
test_clips  = set(unique_originals[:n_test])
train_clips = set(unique_originals[n_test:])

# Get indices for each split
train_idx = [i for i, n in enumerate(original_names) if n in train_clips]
test_idx  = [i for i, n in enumerate(original_names) if n in test_clips]

X_flat_tv = X_flat[train_idx]
X_flat_te = X_flat[test_idx]
X_seq_tv  = X_seq[train_idx]
X_seq_te  = X_seq[test_idx]
y_tv      = y[train_idx]
y_te      = y[test_idx]

# Define architectures 

def build_dense(input_dim, hidden=(64, 32), dropout=0.2):
    inp = keras.Input(shape=(input_dim,))
    x   = inp
    for u in hidden:
        x = layers.Dense(u, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='linear')(x)   
    return keras.Model(inp, out, name='Dense')


def build_cnn(c=C, n_features=39, filters=(32,), kernel=3, dropout=0.2):
    inp = keras.Input(shape=(c, n_features))
    x   = inp
    for f in filters:
        x = layers.Conv1D(f, kernel, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Dropout(dropout)(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)   
    return keras.Model(inp, out, name='CNN')


def build_lstm(c=C, n_features=39, units=(32,), dropout=0.2):
    inp = keras.Input(shape=(c, n_features))
    x   = inp
    for i, u in enumerate(units):
        rs = (i < len(units) - 1)
        x  = layers.LSTM(u, return_sequences=rs, dropout=dropout)(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)   
    return keras.Model(inp, out, name='LSTM')


def build_gru(c=C, n_features=39, units=(32,), dropout=0.2):
    inp = keras.Input(shape=(c, n_features))
    x   = inp
    for i, u in enumerate(units):
        rs = (i < len(units) - 1)
        x  = layers.GRU(u, return_sequences=rs, dropout=dropout)(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)   
    return keras.Model(inp, out, name='GRU')


# Compile

def compile_model(model, optimizer='adam', lr=1e-3):
    opt_map = {
        'adam':    keras.optimizers.Adam(learning_rate=lr),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=lr),
    }
    model.compile(
        optimizer=opt_map[optimizer],
        loss='mae',               
        metrics=['mae', 'mse']    
    )
    return model

# 3-fold CV 

def run_cv(X_tv, y_tv, X_te, y_te,
           build_fn, is_seq,
           optimizer, lr, batch_size,
           arch_name, n_folds=10):

    run_name = f"{arch_name}_{optimizer}_lr{lr}_bs{batch_size}"
    print(f"\n{'='*55}")
    print(f"  {run_name}  ({n_folds}-fold CV)")
    print(f"{'='*55}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_maes = []
    best_mae, best_model, best_scaler = np.inf, None, None

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_tv)):
        print(f"  Fold {fold_idx+1}/{n_folds}", end='  ')

        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]

        # Normalise — fit on train only
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(
            X_tr.reshape(len(X_tr), -1)
        ).reshape(X_tr.shape).astype(np.float32)
        X_val_sc = scaler.transform(
            X_val.reshape(len(X_val), -1)
        ).reshape(X_val.shape).astype(np.float32)

        model = build_fn()
        model = compile_model(model, optimizer=optimizer, lr=lr)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10,
                restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, verbose=0)   
        ]

        model.fit(
            X_tr_sc, y_tr,
            validation_data=(X_val_sc, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        y_pred = model.predict(X_val_sc, verbose=0).flatten()
        mae    = float(np.mean(np.abs(y_val - y_pred)))
        print(f"MAE={mae:.4f}")
        fold_maes.append(mae)

        if mae < best_mae:
            best_mae, best_model, best_scaler = mae, model, scaler

    avg_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    print(f"\n  {n_folds}-FOLD: MAE={avg_mae:.4f} +/- {std_mae:.4f}")

    # Final test evaluation
    X_te_sc = best_scaler.transform(
        X_te.reshape(len(X_te), -1)
    ).reshape(X_te.shape).astype(np.float32)

    y_pred_te = best_model.predict(X_te_sc, verbose=0).flatten()
    y_pred_te = np.clip(y_pred_te, 0.0, 4.0)

    test_mae  = float(np.mean(np.abs(y_te - y_pred_te)))
    test_mse  = float(np.mean((y_te - y_pred_te)**2))
    corr, _   = pearsonr(y_te, y_pred_te)

    print(f"  TEST: MAE={test_mae:.4f}  MSE={test_mse:.4f}  "
          f"Corr={corr:.3f}")

    n_params = best_model.count_params()
    print(f"  Params: {n_params:,}")

    # Save model and scaler
    best_model.save(str(RESULTS_DIR / f'{run_name}_model.keras'))
    joblib.dump(best_scaler, str(RESULTS_DIR / f'{run_name}_scaler.pkl'))

    return {
        'run':      run_name,
        'arch':     arch_name,
        'optimizer':optimizer,
        'lr':       lr,
        'batch':    batch_size,
        'cv_mae':   avg_mae,
        'cv_std':   std_mae,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'corr':     corr,
        'n_params': n_params,
    }


OPTIMIZERS   = ['adam', 'rmsprop']
BATCH_SIZES  = [8, 16]
LEARNING_RATES = [1e-3, 5e-4]
N_FOLDS      = 3

all_results = []

# Architecture configs to test
ARCH_CONFIGS = [
    # Dense variants
    ('Dense_medium', lambda: build_dense(390, hidden=(64,),    dropout=0.2), False),
    ('Dense_large',  lambda: build_dense(390, hidden=(128,64),dropout=0.3), False),
    # CNN variants
    ('CNN_medium',   lambda: build_cnn(filters=(32,),   kernel=3), True),
    # LSTM
    ('LSTM_medium',  lambda: build_lstm(units=(64,),           ), True),
    # GRU
    ('GRU_medium',   lambda: build_gru(units=(64,),            ), True),
]

for arch_name, build_fn, is_seq in ARCH_CONFIGS:
    X_tv = X_seq_tv if is_seq else X_flat_tv
    X_te = X_seq_te if is_seq else X_flat_te

    for opt in OPTIMIZERS:
        for lr in LEARNING_RATES:
            for bs in BATCH_SIZES:
                result = run_cv(
                    X_tv, y_tv, X_te, y_te,
                    build_fn, is_seq,
                    opt, lr, bs,
                    arch_name, N_FOLDS
                )
                all_results.append(result)
                results_df = pd.DataFrame(all_results).sort_values('cv_mae')
                results_df.to_csv(str(RESULTS_DIR / 'all_results.csv'), index=False)
                print(results_df[['arch','optimizer','cv_mae','test_mae',
                      'corr','n_params']].to_string(index=False))

# Evaluation

def full_evaluation(y_true, y_pred, arch_name, results_dir):
    y_pred = np.clip(y_pred, float(y_true.min()), float(y_true.max()))

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred)**2))
    bias = float(np.mean(y_pred - y_true))   
    corr, p_val = pearsonr(y_true, y_pred)

    print(f"  Evaluation: {arch_name}")
    print(f"  MAE         : {mae:.4f}")
    print(f"  MSE         : {mse:.4f}")
    print(f"  Correlation : {corr:.3f} (p={p_val:.4f})")
    print(f"  Bias        : {bias:+.4f} "
          f"({'over-predicts' if bias > 0 else 'under-predicts'})")

    # Plot 1: Predicted vs Ground Truth 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Evaluation — {arch_name}', fontsize=13)

    # Scatter: predicted vs true
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.7, color='steelblue', s=40)
    lims = [min(y_true.min(), y_pred.min()) - 0.05,
            max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Ground Truth Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'Predicted vs True\nCorr={corr:.3f}  MAE={mae:.4f}')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    #  Plot 2: Bland-Altman 
    means = (y_true + y_pred) / 2
    diffs = y_true - y_pred
    md    = np.mean(diffs)
    sd    = np.std(diffs)
    upper = md + 1.96 * sd
    lower = md - 1.96 * sd

    ax = axes[1]
    ax.scatter(means, diffs, alpha=0.7, color='steelblue', s=40)
    ax.axhline(md,    color='red',  lw=2,   label=f'Bias={md:+.3f}')
    ax.axhline(upper, color='gray', lw=1.5, linestyle='--',
               label=f'+1.96SD={upper:.3f}')
    ax.axhline(lower, color='gray', lw=1.5, linestyle='--',
               label=f'-1.96SD={lower:.3f}')
    ax.axhline(0, color='black', lw=0.5, linestyle=':')
    ax.set_xlabel('Mean of True and Predicted')
    ax.set_ylabel('True − Predicted')
    ax.set_title('Bland-Altman Plot\n(bias and limits of agreement)')
    ax.legend(fontsize=8)

    # Plot 3: Outlier analysis
    abs_errors = np.abs(y_true - y_pred)
    outlier_threshold = md + 2 * sd   # points outside 2 SD = outliers
    is_outlier = np.abs(diffs) > abs(outlier_threshold)
    n_outliers = is_outlier.sum()

    ax = axes[2]
    ax.bar(range(len(abs_errors)),
           sorted(abs_errors, reverse=True),
           color=['red' if e > abs(outlier_threshold) else 'steelblue'
                  for e in sorted(abs_errors, reverse=True)])
    ax.axhline(abs(outlier_threshold), color='red', lw=1.5,
               linestyle='--', label=f'Outlier threshold={abs(outlier_threshold):.3f}')
    ax.set_xlabel('Video (sorted by error)')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Outlier Analysis\n{n_outliers} outliers detected')
    ax.legend()

    plt.tight_layout()
    plot_path = results_dir / f'{arch_name}_evaluation.png'
    plt.close()

    #  Bias direction interpretation
    print(f"\n  Bias analysis:")
    if abs(bias) < 0.01:
        print(f"    No systematic bias")
    elif bias > 0:
        print(f"    Model over-predicts by {bias:.4f} on average")
    else:
        print(f"    Model under-predicts by {abs(bias):.4f} on average")

    # Outlier details 
    print(f"\n  Outliers (error > 2SD = {abs(outlier_threshold):.3f}):")
    print(f"    Count: {n_outliers} / {len(y_true)}")
    if n_outliers > 0:
        outlier_errors = abs_errors[is_outlier]
        print(f"    Max outlier error: {outlier_errors.max():.4f}")
        print(f"    Possible causes: model bias in specific score range,")
        print(f"    unusual exercise form, or noisy ground truth label")

    # Sort by true score and check if predictions follow same order
    sorted_idx    = np.argsort(y_true)
    rank_corr     = np.corrcoef(
        np.argsort(y_true), np.argsort(y_pred))[0, 1]

    print(f"\n  Usefulness check (ranking consistency):")
    print(f"    Rank correlation: {rank_corr:.3f}")
    if rank_corr > 0.7:
        print(f"Model correctly ranks better vs worse exercises")
    elif rank_corr > 0.4:
        print(f"Model partially ranks exercises correctly")
    else:
        print(f"Model struggles to distinguish better from worse")

    return {
        'mae':        mae,
        'mse':        mse,
        'corr':       corr,
        'bias':       bias,
        'n_outliers': int(n_outliers),
        'rank_corr':  rank_corr,
        'upper_loa':  upper,   
        'lower_loa':  lower,
    }


# Run evaluation for best model of each architecture family 

print("Evaluation")

eval_results = []

for arch_family in ['Dense', 'CNN', 'LSTM', 'GRU']:
    family_df = results_df[results_df['arch'].str.contains(arch_family)]
    if family_df.empty:
        continue

    best_family = family_df.iloc[0]
    best_run    = best_family['run']
    is_seq      = any(n in arch_family for n in ['CNN', 'LSTM', 'GRU'])

    # Load best model
    model_path  = RESULTS_DIR / f'{best_run}_model.keras'
    scaler_path = RESULTS_DIR / f'{best_run}_scaler.pkl'

    if not model_path.exists():
        continue

    model  = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(str(scaler_path))

    X_te_use = X_seq_te if is_seq else X_flat_te
    X_te_sc  = scaler.transform(
        X_te_use.reshape(len(X_te_use), -1)
    ).reshape(X_te_use.shape).astype(np.float32)

    y_pred = model.predict(X_te_sc, verbose=0).flatten()

    eval_res = full_evaluation(y_te, y_pred, arch_family, RESULTS_DIR)
    eval_res['arch'] = arch_family
    eval_results.append(eval_res)

# Final comparison table
eval_df = pd.DataFrame(eval_results)
eval_df.to_csv(str(RESULTS_DIR / 'evaluation_summary.csv'), index=False)

print(f"\n{'='*55}")
print("EVALUATION SUMMARY — All architectures")
print(f"{'='*55}")
print(eval_df[['arch', 'mae', 'mse', 'corr',
               'bias', 'n_outliers', 'rank_corr']].to_string(index=False))

# MAE comparison bar chart
plt.figure(figsize=(8, 5))
plt.bar(eval_df['arch'], eval_df['mae'], color='steelblue', alpha=0.8)
plt.axhline(eval_df['mae'].min(), color='red', lw=1.5,
            linestyle='--', label=f"Best MAE={eval_df['mae'].min():.4f}")
plt.xlabel('Architecture')
plt.ylabel('Test MAE')
plt.title('MAE Comparison — Dense vs CNN vs LSTM vs GRU')
plt.legend()
plt.tight_layout()
plt.close()


# Save best model with pipeline-ready filenames
import shutil
import json

best       = results_df.iloc[0]
best_run   = best['run']

# Copy best model to pipeline folder with standard names
pipeline_model  = Path('models/scoring_model.keras')
pipeline_scaler = Path('models/scoring_scaler.pkl')

shutil.copy(
    str(RESULTS_DIR / f'{best_run}_model.keras'),
    str(pipeline_model)
)
shutil.copy(
    str(RESULTS_DIR / f'{best_run}_scaler.pkl'),
    str(pipeline_scaler)
)
print(f"\nPipeline models saved:")
print(f"  {pipeline_model}")
print(f"  {pipeline_scaler}")

# Save training summary JSON
training_summary = {
    "best_architecture":  best['arch'],
    "best_optimizer":     best['optimizer'],
    "best_lr":            best['lr'],
    "best_batch":         int(best['batch']),
    "cv_folds":           N_FOLDS,
    "cv_mae":             round(float(best['cv_mae']), 4),
    "cv_std":             round(float(best['cv_std']), 4),
    "test_mae":           round(float(best['test_mae']), 4),
    "test_mse":           round(float(best['test_mse']), 4),
    "correlation":        round(float(best['corr']), 4),
    "n_params":           int(best['n_params']),
    "n_training_videos":  len(y_tv),
    "n_test_videos":      len(y_te),
    "score_range_min":    round(float(y.min()), 4),
    "score_range_max":    round(float(y.max()), 4),
    "model_path":         str(pipeline_model),
    "scaler_path":        str(pipeline_scaler),
}

with open(str(RESULTS_DIR / 'training_summary.json'), 'w') as f:
    json.dump(training_summary, f, indent=2)
print(f"  training_summary.json saved")