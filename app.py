import os
import tempfile
import json
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
except Exception:
    tf = None

# -------------------------
# Config
# -------------------------
FEATURES_FILE = "models/eeg_features.npz"
AE_MODEL_FILE = "models/ae_anomaly_results.npz"
CLASSIFIER_FILE = "models/finetuned_model.h5"
MODEL2_FILE = "models/model1.h5"
DEFAULT_FS = 500

# -------------------------
# .mat loader
# -------------------------

def _is_numeric_ndarray(x):
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def _collect_numeric_arrays(obj, collected):
    if _is_numeric_ndarray(obj):
        collected.append(obj)
        return
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for el in obj.flat:
            _collect_numeric_arrays(el, collected)
        return
    if isinstance(obj, (list, tuple)):
        for el in obj:
            _collect_numeric_arrays(el, collected)
        return
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            _collect_numeric_arrays(v, collected)
        return
    if isinstance(obj, np.ndarray) and getattr(obj.dtype, "names", None):
        for name in obj.dtype.names:
            _collect_numeric_arrays(obj[name], collected)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numeric_arrays(v, collected)
        return


def load_mat_largest_array(path, debug=False):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    numeric_arrays = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        _collect_numeric_arrays(v, numeric_arrays)
    if not numeric_arrays:
        raise ValueError("No numeric arrays found in the .mat file.")
    best = max(numeric_arrays, key=lambda arr: getattr(arr, "size", 0))
    if debug:
        print("Debug arrays found:", [(a.shape, getattr(a, "dtype", None)) for a in numeric_arrays])
    return np.array(best, dtype=np.float32)


def normalize_matrix_shape(arr):
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        if arr.shape[0] <= 64 and arr.shape[1] > 1000:
            return arr.T
        return arr
    return arr.reshape(arr.shape[0], -1)

# -------------------------
# Segmenting & normalization
# -------------------------

def segment_recording(arr, fs, window_samples, step_samples, expected_channels=None):
    if expected_channels is None:
        expected_channels = arr.shape[1]
    if arr.shape[1] < expected_channels:
        pad_width = expected_channels - arr.shape[1]
        arr = np.concatenate([arr, np.zeros((arr.shape[0], pad_width), dtype=arr.dtype)], axis=1)
    elif arr.shape[1] > expected_channels:
        arr = arr[:, :expected_channels]

    segs = []
    starts = []
    for start in range(0, arr.shape[0] - window_samples + 1, step_samples):
        seg = arr[start:start + window_samples, :].astype(np.float32)
        mean = np.mean(seg, axis=0, keepdims=True)
        std = np.std(seg, axis=0, keepdims=True) + 1e-8
        seg = (seg - mean) / std
        segs.append(seg)
        starts.append(start)
    if len(segs) == 0:
        return np.zeros((0, window_samples, expected_channels), dtype=np.float32), np.array([])
    return np.stack(segs, axis=0), np.array(starts)

# -------------------------
# Bandpower features + IF
# -------------------------

def bandpower_features(seg, fs=DEFAULT_FS):
    x = seg.mean(axis=1)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    psd = np.abs(np.fft.rfft(x)) ** 2
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
    feats = []
    tot = psd.sum() + 1e-8
    for a, b in bands:
        idx = np.where((freqs >= a) & (freqs < b))[0]
        feats.append(psd[idx].sum() / tot if idx.size > 0 else 0.0)
    feats.append(np.var(x))
    return np.array(feats, dtype=np.float32)


def build_isolationforest_from_features():
    if not os.path.exists(FEATURES_FILE):
        return None, None
    try:
        D = np.load(FEATURES_FILE)
        Xfeat = D['X']
        y = D['y']
        normal = Xfeat[y == 0]
        if normal.size == 0:
            return None, None
        feat_mat = np.vstack([bandpower_features(seg) for seg in normal])
        clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
        clf.fit(feat_mat)
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(feat_mat)
        return clf, scaler
    except Exception as e:
        print("IsolationForest build failed:", e)
        return None, None

IF_CLF, IF_SCALER = build_isolationforest_from_features()

# -------------------------
# Load classifiers & AE
# -------------------------

MODEL = None
MODEL_INPUT_SHAPE = None
MODEL2 = None
MODEL2_INPUT_SHAPE = None

if os.path.exists(CLASSIFIER_FILE) and tf is not None:
    try:
        MODEL = tf.keras.models.load_model(CLASSIFIER_FILE)
        try:
            MODEL_INPUT_SHAPE = MODEL.input_shape[1:]
        except Exception:
            MODEL_INPUT_SHAPE = None
        print("Loaded classifier:", CLASSIFIER_FILE, "input shape:", MODEL_INPUT_SHAPE)
    except Exception as e:
        print("Failed to load classifier:", e)
        MODEL = None
else:
    if not os.path.exists(CLASSIFIER_FILE):
        print("Classifier file not found:", CLASSIFIER_FILE)
    else:
        print("TensorFlow not available; skipping classifier load.")

if os.path.exists(MODEL2_FILE) and tf is not None:
    try:
        MODEL2 = tf.keras.models.load_model(MODEL2_FILE)
        try:
            MODEL2_INPUT_SHAPE = MODEL2.input_shape[1:]
        except Exception:
            MODEL2_INPUT_SHAPE = None
        print("Loaded classifier2:", MODEL2_FILE, "input shape:", MODEL2_INPUT_SHAPE)
    except Exception as e:
        print("Failed to load classifier2:", e)
        MODEL2 = None
else:
    if not os.path.exists(MODEL2_FILE):
        print("Classifier2 file not found:", MODEL2_FILE)

AE_MODEL = None
AE_NPZ_DATA = None
if os.path.exists(AE_MODEL_FILE):
    if AE_MODEL_FILE.endswith(".npz"):
        try:
            AE_NPZ_DATA = dict(np.load(AE_MODEL_FILE, allow_pickle=True))
            print("Loaded AE npz artifact:", AE_MODEL_FILE, "keys:", list(AE_NPZ_DATA.keys()))
        except Exception as e:
            print("Failed to load AE npz file:", e)
            AE_NPZ_DATA = None
    else:
        if tf is None:
            print("TensorFlow not available; cannot load AE model:", AE_MODEL_FILE)
        else:
            try:
                AE_MODEL = tf.keras.models.load_model(AE_MODEL_FILE)
                print("Loaded AE model:", AE_MODEL_FILE)
            except Exception as e:
                print("Failed to load AE model:", e)
                AE_MODEL = None
else:
    print("No AE model/artifact found at:", AE_MODEL_FILE)

# -------------------------
# Prediction helpers
# -------------------------

def predict_in_batches(model, X, batch_size=128):
    if X.shape[0] == 0:
        return np.zeros((0,))
    out = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        p = model.predict(batch, verbose=0)
        out.append(p)
    return np.vstack(out)


def get_model_mean_prob(model, segments, model_input_shape=None, batch_size=64):
    if model is None or segments.shape[0] == 0:
        return None, np.array([])
    seg_for_model = segments.copy()
    if model_input_shape is not None and model_input_shape[0] is not None:
        expected_t = int(model_input_shape[0])
        if seg_for_model.shape[1] > expected_t:
            seg_for_model = seg_for_model[:, :expected_t, :]
        elif seg_for_model.shape[1] < expected_t:
            pad_t = expected_t - seg_for_model.shape[1]
            seg_for_model = np.pad(seg_for_model, ((0, 0), (0, pad_t), (0, 0)), mode='constant')
    preds = predict_in_batches(model, seg_for_model, batch_size=batch_size)
    if preds.size == 0:
        return None, np.array([])
    try:
        if preds.ndim == 2 and preds.shape[1] == 2:
            ex = np.exp(preds - preds.max(axis=1, keepdims=True))
            probs = ex[:, 1] / ex.sum(axis=1)
        elif preds.ndim == 2 and preds.shape[1] == 1:
            probs = 1.0 / (1.0 + np.exp(-preds.ravel()))
        elif preds.ndim == 1:
            probs = 1.0 / (1.0 + np.exp(-preds.ravel()))
        else:
            probs = preds[:, -1]
        return float(np.mean(probs)), probs
    except Exception as e:
        print("Failed to normalize model preds:", e)
        try:
            pl = np.clip(preds.ravel(), 0.0, 1.0)
            return float(pl.mean()), pl
        except Exception:
            return None, np.array([])

# -------------------------
# Anomaly scoring
# -------------------------

def anomaly_scores_for_segments(segments):
    if segments.shape[0] == 0:
        return np.array([])
    if AE_MODEL is not None:
        try:
            pred = AE_MODEL.predict(segments, batch_size=64, verbose=0)
            mse = np.mean((pred - segments) ** 2, axis=(1, 2))
            return mse
        except Exception as e:
            print("AE model predict failed:", e)
    if AE_NPZ_DATA is not None:
        if 'mean_segment' in AE_NPZ_DATA:
            mean_seg = np.array(AE_NPZ_DATA['mean_segment'], dtype=np.float32)
            st, sc = segments.shape[1], segments.shape[2]
            mt, mc = mean_seg.shape
            if mt > st:
                mean_seg = mean_seg[:st, :]
            elif mt < st:
                mean_seg = np.pad(mean_seg, ((0, st - mt), (0, 0)), mode='constant')
            if mean_seg.shape[1] > sc:
                mean_seg = mean_seg[:, :sc]
            elif mean_seg.shape[1] < sc:
                mean_seg = np.pad(mean_seg, ((0, 0), (0, sc - mean_seg.shape[1])), mode='constant')
            mse = np.mean((segments - mean_seg[None, ...]) ** 2, axis=(1, 2))
            return mse
        if 'mse_per_seg' in AE_NPZ_DATA:
            try:
                dist = np.array(AE_NPZ_DATA['mse_per_seg']).ravel()
                raw_scores = np.array([np.mean(seg ** 2) for seg in segments])
                mean_dist = dist.mean(); std_dist = dist.std() + 1e-8
                return (raw_scores - mean_dist) / std_dist
            except Exception as e:
                print("AE npz 'mse_per_seg' handling failed:", e)
        if 'mse_mean' in AE_NPZ_DATA and 'mse_std' in AE_NPZ_DATA:
            try:
                mean_dist = float(AE_NPZ_DATA['mse_mean']); std_dist = float(AE_NPZ_DATA['mse_std']) + 1e-8
                raw_scores = np.array([np.mean(seg ** 2) for seg in segments])
                return (raw_scores - mean_dist) / std_dist
            except Exception as e:
                print("AE npz mse_mean/mse_std handling failed:", e)
    if IF_CLF is not None and IF_SCALER is not None:
        try:
            feats = np.vstack([bandpower_features(seg) for seg in segments])
            feats_s = IF_SCALER.transform(feats)
            scores = -IF_CLF.decision_function(feats_s)
            return scores
        except Exception as e:
            print("IsolationForest scoring failed:", e)
    scores = np.array([np.mean(seg ** 2) for seg in segments])
    return scores

# -------------------------
# Plotting
# -------------------------

def plot_eeg_and_annotations(arr, fs, starts, window_samples, step_samples, per_seg_plot_vals, used_score_threshold, persist_seconds, title="EEG"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]})
    n_segments = len(per_seg_plot_vals)
    ch = arr.shape[1]
    seg_matrix = np.zeros((ch, n_segments), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = arr[s:s + window_samples, :min(ch, arr.shape[1])]
        if seg.size == 0:
            seg_matrix[:, i] = 0
        else:
            if seg.ndim == 2:
                m = np.mean(np.abs(seg), axis=0)
                seg_matrix[:m.size, i] = m
            else:
                seg_matrix[0, i] = np.mean(np.abs(seg))
    if len(starts) > 0:
        x_min = float(starts[0]) / fs
        x_max = float(starts[-1] + window_samples) / fs
    else:
        x_min = 0.0
        x_max = float(arr.shape[0]) / fs
    im = axes[0].imshow(seg_matrix, aspect='auto', origin='lower', extent=[x_min, x_max, 0, ch], cmap='viridis')
    axes[0].set_ylabel("Channel index")
    axes[0].set_title(f"{title} — Heatmap (segment-aggregated energy)")
    fig.colorbar(im, ax=axes[0], label='mean|amplitude|')
    t = np.arange(arr.shape[0]) / fs
    ch0 = arr[:, 0] if arr.shape[1] > 0 else arr[:, :1].reshape(-1)
    axes[1].plot(t, ch0, lw=0.7)
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    raw_pred = (per_seg_plot_vals >= 0.5).astype(int) if per_seg_plot_vals.size > 0 else np.array([])
    step_seconds = step_samples / fs
    persist_windows = max(1, int(np.ceil(persist_seconds / step_seconds))) if raw_pred.size > 0 else 1
    if raw_pred.size > 0:
        conv = np.convolve(raw_pred, np.ones(persist_windows, dtype=int), mode='same')
        smoothed = (conv >= persist_windows).astype(int)
    else:
        smoothed = np.array([])
    if smoothed.size > 0 and smoothed.any():
        starts_idx = np.where(np.diff(np.concatenate([[0], smoothed])) == 1)[0]
        ends_idx = np.where(np.diff(np.concatenate([smoothed, [0]])) == -1)[0]
        for s_i, e_i in zip(starts_idx, ends_idx):
            s_time = (starts[s_i] / fs)
            e_time = ((starts[min(e_i, len(starts) - 1)] + window_samples) / fs)
            axes[1].axvspan(s_time, e_time, color='orange', alpha=0.25)
            axes[0].axvspan(s_time, e_time, color='orange', alpha=0.25)
    axes[1].set_title("Channel 0 waveform with shaded (AE) anomaly regions (orange)")
    plt.tight_layout()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------
# Ensemble implementations
# -------------------------

def majority_vote(probs_list, ae_conf, classifier_threshold=0.5, ae_threshold=0.5):
    votes = []
    for p in probs_list:
        if p is None:
            votes.append(None)
        else:
            votes.append(int(p >= classifier_threshold))
    ae_vote = int(ae_conf >= ae_threshold) if ae_conf is not None else None
    votes.append(ae_vote)
    concrete = [v for v in votes if v is not None]
    if len(concrete) == 0:
        return "No models available", votes, {}
    s = sum(concrete)
    final = "Likely Epilepsy" if s > len(concrete) / 2.0 else "Likely Normal"
    details = {"sum": int(s), "n_voters": len(concrete)}
    return final, votes, details


def weighted_vote_soft(probs_list, ae_conf, weights, final_threshold=0.5):
    assert len(weights) == len(probs_list) + 1
    vals = []
    present_w = []
    for p, w in zip(probs_list, weights[:-1]):
        if p is None:
            vals.append(0.0)
            present_w.append(0.0)
        else:
            vals.append(float(p))
            present_w.append(float(w))
    if ae_conf is None:
        vals.append(0.0)
        present_w.append(0.0)
    else:
        vals.append(float(ae_conf))
        present_w.append(float(weights[-1]))
    present_w = np.array(present_w, dtype=np.float32)
    total_w = present_w.sum()
    if total_w <= 0:
        return "No models available", vals, {"weighted_score": 0.0}
    norm_w = present_w / total_w
    vals_arr = np.array(vals, dtype=np.float32)
    weighted_score = float((vals_arr * norm_w).sum())
    final = "Likely Epilepsy" if weighted_score >= final_threshold else "Likely Normal"
    details = {"weighted_score": weighted_score, "norm_weights": norm_w.tolist()}
    return final, vals, details


def soft_average(probs_list, ae_conf, weights, final_threshold=0.5):
    return weighted_vote_soft(probs_list, ae_conf, weights, final_threshold=final_threshold)


def normalize_ae_confidence(ae_conf, ref_dist=None):
    if ae_conf is None:
        return None
    if ref_dist is None or len(ref_dist) == 0:
        return float(np.clip(ae_conf, 0.0, 1.0))
    try:
        ref = np.array(ref_dist).ravel()
        pct = float((ref < ae_conf).mean())
        return float(np.clip(pct, 0.0, 1.0))
    except Exception:
        return float(np.clip(ae_conf, 0.0, 1.0))

# -------------------------
# Main analyzer
# -------------------------

def analyze_mat_file(mat_file, fs_input, persist_seconds, ensemble_method='Soft Average (C)', classifier_threshold=0.5, ae_threshold=0.5, weights_json='[0.33,0.33,0.34]', ae_percentile=95):
    tmp_path = None
    if mat_file is None:
        return None, "No file provided.", "N/A", "No verdict"
    try:
        if hasattr(mat_file, "name") and os.path.exists(mat_file.name):
            path = mat_file.name
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mat")
            tmp.write(mat_file.read())
            tmp.flush()
            tmp.close()
            path = tmp.name
            tmp_path = path
        try:
            arr_raw = load_mat_largest_array(path)
        except Exception as e:
            return None, f"Error loading .mat: {e}", "N/A", "No verdict (load failed)"
        arr = normalize_matrix_shape(arr_raw)
        fs = int(fs_input) if fs_input and fs_input > 0 else DEFAULT_FS
        if MODEL_INPUT_SHAPE is not None and MODEL_INPUT_SHAPE[0] is not None:
            window_samples = int(MODEL_INPUT_SHAPE[0])
            step_samples = max(1, window_samples // 2)
            expected_channels = MODEL_INPUT_SHAPE[1] if MODEL_INPUT_SHAPE is not None else arr.shape[1]
        else:
            window_samples = int(2.0 * fs)
            step_samples = int(1.0 * fs)
            expected_channels = arr.shape[1]
        segments, starts = segment_recording(arr, fs, window_samples, step_samples, expected_channels=expected_channels)
        if segments.shape[0] == 0:
            return None, "Recording too short for segmentation.", "N/A", "No verdict (too short)"
        mean1, probs1 = get_model_mean_prob(MODEL, segments, MODEL_INPUT_SHAPE) if MODEL is not None else (None, np.array([]))
        mean2, probs2 = get_model_mean_prob(MODEL2, segments, MODEL2_INPUT_SHAPE) if MODEL2 is not None else (None, np.array([]))
        scores = anomaly_scores_for_segments(segments)
        if AE_NPZ_DATA is not None and 'mse_per_seg' in AE_NPZ_DATA:
            ref = np.array(AE_NPZ_DATA['mse_per_seg']).ravel()
            used_score_threshold = float(np.percentile(ref, ae_percentile))
        else:
            used_score_threshold = float(np.percentile(scores, ae_percentile)) if scores.size > 0 else 1.0
        if used_score_threshold <= 0:
            per_seg_plot_vals = np.clip(scores, 0.0, 1.0) if scores.size > 0 else np.zeros_like(scores)
        else:
            per_seg_plot_vals = np.clip(scores / used_score_threshold, 0.0, 1.0)
        AE_confidence = 0.0
        if scores.size > 0:
            raw_binary = (scores >= used_score_threshold).astype(int)
            step_seconds = step_samples / fs
            persist_windows_local = max(1, int(np.ceil(persist_seconds / step_seconds)))
            conv = np.convolve(raw_binary, np.ones(persist_windows_local, dtype=int), mode='same')
            smoothed = (conv >= 1).astype(int)
            if smoothed.any():
                starts_idx = np.where(np.diff(np.concatenate([[0], smoothed])) == 1)[0]
                ends_idx = np.where(np.diff(np.concatenate([smoothed, [0]])) == -1)[0]
                region_means = []
                for s_i, e_i in zip(starts_idx, ends_idx):
                    e_i_clamped = min(e_i, len(scores))
                    region_scores = scores[s_i:e_i_clamped]
                    if region_scores.size > 0:
                        region_means.append(np.mean(region_scores))
                if region_means:
                    peak_region_mean = max(region_means)
                    if AE_NPZ_DATA is not None and 'mse_per_seg' in AE_NPZ_DATA:
                        ref = np.array(AE_NPZ_DATA['mse_per_seg']).ravel()
                        AE_confidence = float((ref < peak_region_mean).mean())
                    else:
                        lo = np.percentile(scores, 1)
                        hi = np.percentile(scores, 99)
                        AE_confidence = float(np.clip((peak_region_mean - lo) / (hi - lo + 1e-12), 0.0, 1.0))
        AE_confidence = float(AE_confidence)
        if AE_confidence > 0.7:
          AE_confidence = 0.7
        elif AE_confidence < 0.3:
          AE_confidence = 0.3
        try:
            weights = json.loads(weights_json)
            if not isinstance(weights, list) or len(weights) != 3:
                raise ValueError("weights must be a list of 3 numbers")
            weights = [float(w) for w in weights]
        except Exception:
            weights = [0.33, 0.33, 0.34]
        ref_dist = None
        if AE_NPZ_DATA is not None and 'mse_per_seg' in AE_NPZ_DATA:
            try:
                ref_dist = np.array(AE_NPZ_DATA['mse_per_seg']).ravel()
            except Exception:
                ref_dist = None
        ae_conf_norm = normalize_ae_confidence(AE_confidence, ref_dist=ref_dist)
        probs_list = [mean1, mean2]
        if ensemble_method == 'Majority':
            final_label, votes, details = majority_vote(probs_list, ae_conf_norm, classifier_threshold=classifier_threshold, ae_threshold=ae_threshold)
        elif ensemble_method in ('Weighted Vote (B)', 'Soft-weighted Vote (B)'):
            final_label, votes, details = weighted_vote_soft(probs_list, ae_conf_norm, weights, final_threshold=classifier_threshold)
        else:
            final_label, values, details = soft_average(probs_list, ae_conf_norm, weights, final_threshold=classifier_threshold)
            votes = values
        metrics_text_lines = []
        metrics_text_lines.append(f"Model1 mean_prob: {mean1:.3f}" if mean1 is not None else "Model1: N/A")
        metrics_text_lines.append(f"Model2 mean_prob: {mean2:.3f}" if mean2 is not None else "Model2: N/A")
        metrics_text_lines.append(f"AE_confidence (raw): {AE_confidence:.3f}")
        metrics_text_lines.append(f"AE_confidence (norm): {ae_conf_norm:.3f}" if ae_conf_norm is not None else "AE_confidence (norm): N/A")
        metrics_text_lines.append(f"Ensemble: {ensemble_method}")
        metrics_text_lines.append(f"Final decision: {final_label}")
        metrics_text_lines.append(f"Details: {json.dumps(details)}")
        figbuf = plot_eeg_and_annotations(arr, fs, starts, window_samples, step_samples, per_seg_plot_vals, used_score_threshold, persist_seconds, title=os.path.basename(path))
        classifier_confidence_str = "{:.3f}".format(mean1) if mean1 is not None else "N/A"
        return figbuf, "\n".join(metrics_text_lines), classifier_confidence_str, final_label
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks() as demo:
    gr.Markdown("# SeizAware — EEG Analyzer ")
    with gr.Row():
        with gr.Column(scale=1):
            mat_in = gr.File(label="Upload .mat EEG file", file_types=[".mat"])
            fs_in = gr.Number(value=DEFAULT_FS, label="Sampling frequency (Hz)")
            persist = gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Persistence (seconds) for alarm")
            ensemble = gr.Radio(choices=['Soft Average (C)', 'Soft-weighted Vote (B)', 'Majority'], value='Soft Average (C)', label="Ensemble method")
            weights_text = gr.Textbox(value='[0.33, 0.33, 0.34]', label='Weights JSON [model1, model2, AE] for Soft methods')
            classifier_thresh = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Decision threshold (used differently per method)")
            ae_thresh = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="AE threshold for Majority voting")
            ae_pct = gr.Dropdown(choices=[90, 95, 99], value=95, label="AE percentile for thresholding (when ref present)")
            analyze_btn = gr.Button("Analyze")
            notes = gr.Markdown("Choose ensemble strategy: Majority (binary majority), Soft-weighted Vote (B), or Soft Average (C). Weights are normalized internally.")
        with gr.Column(scale=2):
            out_image = gr.Image(type="pil", label="EEG & Annotations")
            out_metrics = gr.Textbox(label="Metrics / Summary", lines=8)
            out_conf = gr.Textbox(label="Model1 mean probability", lines=1)
            out_verdict = gr.Textbox(label="Final verdict", lines=1)

    def on_analyze(file, fs_val, pers, ens, weights_json, cthresh, aethresh, aepct):
        if file is None:
            return None, "No file provided.", "N/A", "No verdict"
        try:
            imgbuf, metrics_text, conf, verdict = analyze_mat_file(file, fs_val, pers, ensemble_method=ens, classifier_threshold=float(cthresh), ae_threshold=float(aethresh), weights_json=weights_json, ae_percentile=int(aepct))
            from PIL import Image
            img = Image.open(imgbuf)
            return img, metrics_text, conf, verdict
        except Exception as e:
            return None, "Error during analysis: " + str(e), "N/A", "No verdict"

    analyze_btn.click(on_analyze, inputs=[mat_in, fs_in, persist, ensemble, weights_text, classifier_thresh, ae_thresh, ae_pct], outputs=[out_image, out_metrics, out_conf, out_verdict])

if __name__ == '__main__':
    demo.launch(server_name="127.0.0.1", share=False)
