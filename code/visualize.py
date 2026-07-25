"""
Visualization script for generating paper-quality figures.
Produces architecture diagram, comparison plots, reliability diagrams,
uncertainty decomposition, ablation, and sensitivity plots.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

from config import OUTPUT, PLOTS_DIR, RANDOM_SEEDS, ECE_N_BINS
from evaluate import compute_ece

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ------------------------------------------------------------------------------
# Figure 1: Architecture Diagram (schematic)
# ------------------------------------------------------------------------------

def draw_architecture_diagram(save_path):
    """Draw schematic of EDL-UQ architecture."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, color='lightblue'):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, wrap=True)

    # Input
    box(0.2, 2.2, 1.2, 1.6, 'Input\nFeatures\n(~80-d)', 'lightyellow')

    # Backbone layers
    box(1.8, 2.2, 1.4, 1.6, 'MLP\nBackbone\n[128,64,32]', 'lightblue')
    box(3.5, 2.2, 1.4, 1.6, 'BatchNorm\n+ ReLU\n+ Dropout', 'lightblue')

    # Evidence layer
    box(5.2, 2.2, 1.4, 1.6, 'Evidence\nLayer\nSoftplus', 'lightgreen')

    # Output branches
    box(7.0, 4.0, 1.6, 1.2, 'Dirichlet\nParameters\n$\\alpha = e + 1$', 'plum')
    box(7.0, 2.2, 1.6, 1.6, 'Expected\nProbability\n$\\bar{p} = \\alpha / \\alpha_0$', 'plum')
    box(7.0, 0.4, 1.6, 1.2, 'Uncertainty\nDecomposition\n$H_{total}, H_{alea}, H_{epi}$', 'salmon')

    # Arrows
    for x1, x2 in [(1.4, 1.8), (3.2, 3.5), (4.9, 5.2), (6.6, 7.0)]:
        ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Branch arrows
    ax.annotate('', xy=(7.8, 4.6), xytext=(7.8, 3.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.annotate('', xy=(7.8, 3.0), xytext=(7.8, 3.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.annotate('', xy=(7.8, 1.6), xytext=(7.8, 2.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    # Loss box
    box(5.0, 0.2, 1.6, 1.0, 'Loss = CE + \\lambda KL', 'wheat')
    ax.annotate('', xy=(5.8, 1.2), xytext=(5.8, 2.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0, linestyle='--'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved architecture diagram: {save_path}")


# ------------------------------------------------------------------------------
# Figure 2: Method Comparison Bar Chart
# ------------------------------------------------------------------------------

def draw_comparison_plot(results_json_path, save_path):
    """Bar chart comparing methods across metrics."""
    with open(results_json_path, 'r') as f:
        data = json.load(f)
    agg = data.get('aggregated', {})

    methods = ['EDL-UQ', 'LogisticRegression', 'RandomForest', 'XGBoost',
               'LSTM', 'GRU', 'BNN', 'MCDropout']
    metrics = ['accuracy', 'f1_macro', 'auc', 'ece']
    metric_labels = ['Accuracy', 'F1-Macro', 'AUC', 'ECE']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        means = []
        stds = []
        valid_methods = []
        for m in methods:
            if m in agg and metric in agg[m]:
                means.append(agg[m][metric]['mean'])
                stds.append(agg[m][metric].get('std', 0.0))
                valid_methods.append(m)
        x = np.arange(len(valid_methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors[:len(valid_methods)],
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        # Highlight best (lowest for ECE, highest for others)
        if means:
            if metric == 'ece':
                best_idx = int(np.argmin(means))
            else:
                best_idx = int(np.argmax(means))
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved comparison plot: {save_path}")


# ------------------------------------------------------------------------------
# Figure 3: Ablation Study Bar Chart
# ------------------------------------------------------------------------------

def draw_ablation_plot(ablation_json_path, save_path):
    """Bar chart for ablation study results."""
    if not os.path.exists(ablation_json_path):
        print(f"[Visualize] Ablation results not found at {ablation_json_path}, skipping.")
        return
    with open(ablation_json_path, 'r') as f:
        data = json.load(f)
    agg = data.get('aggregated', {})

    variants = list(agg.keys())
    metrics = ['accuracy', 'f1_macro', 'auc', 'ece']
    metric_labels = ['Accuracy', 'F1-Macro', 'AUC', 'ECE']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        means = []
        stds = []
        valid = []
        for v in variants:
            if metric in agg[v]:
                means.append(agg[v][metric]['mean'])
                stds.append(agg[v][metric].get('std', 0.0))
                valid.append(v)
        x = np.arange(len(valid))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color='steelblue',
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(valid, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if means:
            if metric == 'ece':
                best_idx = int(np.argmin(means))
            else:
                best_idx = int(np.argmax(means))
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved ablation plot: {save_path}")


# ------------------------------------------------------------------------------
# Figure 4: Sensitivity Analysis
# ------------------------------------------------------------------------------

def draw_sensitivity_plot(sensitivity_json_path, save_path):
    """Line plots for parameter sensitivity."""
    if not os.path.exists(sensitivity_json_path):
        print(f"[Visualize] Sensitivity results not found, skipping.")
        return
    with open(sensitivity_json_path, 'r') as f:
        data = json.load(f)

    params = list(data.keys())
    n_params = len(params)
    if n_params == 0:
        return
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for ax, param_name in zip(axes, params):
        pdict = data[param_name]
        values = []
        f1s = []
        for k, v in sorted(pdict.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0])):
            values.append(k)
            f1s.append(v.get('f1_macro_mean', v.get('mean', 0)))
        ax.plot(range(len(values)), f1s, marker='o', color='darkgreen', linewidth=2, markersize=6)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=30, ha='right', fontsize=9)
        ax.set_xlabel(param_name)
        ax.set_ylabel('F1-Macro')
        ax.set_title(f'Sensitivity: {param_name}', fontweight='bold')
        ax.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved sensitivity plot: {save_path}")


# ------------------------------------------------------------------------------
# Figure 5: Uncertainty Decomposition
# ------------------------------------------------------------------------------

def draw_uncertainty_decomposition(results_json_path, save_path):
    """Histogram of uncertainty components for EDL-UQ."""
    # We need per-sample uncertainties; load from a representative seed
    from evaluate import evaluate_all_models
    from data_loader import preprocess_and_split
    from models import build_model
    import torch
    from config import DEVICE, CHECKPOINT_DIR

    seed = 42
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(seed=seed, save=False)
    edl_path = os.path.join(CHECKPOINT_DIR, f"edl_seed{seed}.pth")
    if not os.path.exists(edl_path):
        print("[Visualize] EDL checkpoint not found for uncertainty plot, skipping.")
        return

    model = build_model('edl', X_test.shape[1], num_classes=2,
                        hidden_dims=[128,64,32], dropout_rate=0.3).to(DEVICE)
    model.load_state_dict(torch.load(edl_path, map_location=DEVICE))
    model.eval()

    # Batch inference
    all_htotal = []
    all_halea = []
    all_hepi = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            Xb = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            unc = model.predict_uncertainty(Xb)
            all_htotal.append(unc['H_total'].cpu().numpy())
            all_halea.append(unc['H_alea'].cpu().numpy())
            all_hepi.append(unc['H_epi'].cpu().numpy())

    H_total = np.concatenate(all_htotal)
    H_alea = np.concatenate(all_halea)
    H_epi = np.concatenate(all_hepi)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title, color in zip(axes,
                                       [H_total, H_alea, H_epi],
                                       ['Total Uncertainty', 'Aleatoric Uncertainty', 'Epistemic Uncertainty'],
                                       ['royalblue', 'seagreen', 'coral']):
        ax.hist(data, bins=50, color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Uncertainty (nats)')
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontweight='bold')
        ax.axvline(np.median(data), color='darkred', linestyle='--', linewidth=1.5, label=f'Median={np.median(data):.3f}')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved uncertainty decomposition plot: {save_path}")


# ------------------------------------------------------------------------------
# Figure 6: Reliability Diagram
# ------------------------------------------------------------------------------

def draw_reliability_diagram(results_json_path, save_path):
    """Reliability diagram for EDL-UQ and selected baselines."""
    from evaluate import evaluate_all_models
    from data_loader import preprocess_and_split
    from models import build_model, SklearnWrapper, EDLMLP
    import torch
    from config import DEVICE, CHECKPOINT_DIR

    seed = 42
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(seed=seed, save=False)

    methods_to_plot = ['EDL-UQ', 'RandomForest', 'XGBoost', 'MCDropout']
    colors = {'EDL-UQ': '#e41a1c', 'RandomForest': '#377eb8',
              'XGBoost': '#4daf4a', 'MCDropout': '#984ea3'}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Perfect calibration')

    for mname in methods_to_plot:
        pth = os.path.join(CHECKPOINT_DIR, f"{mname.lower()}_seed{seed}.pth")
        pkl = os.path.join(CHECKPOINT_DIR, f"{mname.lower()}_seed{seed}.pkl")
        try:
            if os.path.exists(pkl):
                wrapper = pickle.load(open(pkl, 'rb'))
                probs = wrapper.predict_proba(X_test)[:, 1]
            elif os.path.exists(pth):
                if mname == 'EDL-UQ':
                    model = build_model('edl', X_test.shape[1], num_classes=2,
                                        hidden_dims=[128,64,32], dropout_rate=0.3).to(DEVICE)
                elif mname == 'MCDropout':
                    model = build_model('mcdropout', X_test.shape[1], num_classes=2,
                                        hidden_dims=[128,64,32], dropout_rate=0.3).to(DEVICE)
                else:
                    continue
                model.load_state_dict(torch.load(pth, map_location=DEVICE))
                model.eval()
                if isinstance(model, EDLMLP):
                    with torch.no_grad():
                        probs = model.predict_probs(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()[:, 1]
                else:
                    with torch.no_grad():
                        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
            else:
                continue
            ece, bin_accs, bin_confs, _ = compute_ece(y_test, probs, n_bins=15)
            ax.plot(bin_confs, bin_accs, 'o-', color=colors.get(mname, 'gray'),
                    linewidth=1.5, markersize=5, label=f"{mname} (ECE={ece:.3f})")
        except Exception as e:
            print(f"[Visualize] Reliability diagram skip {mname}: {e}")

    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved reliability diagram: {save_path}")


# ------------------------------------------------------------------------------
# Figure 7: t-SNE Feature Visualization (optional)
# ------------------------------------------------------------------------------

def draw_tsne_plot(save_path):
    """t-SNE visualization of test set features colored by true label."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[Visualize] sklearn not available for t-SNE, skipping.")
        return
    from data_loader import preprocess_and_split

    seed = 42
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = preprocess_and_split(seed=seed, save=False)
    # Subsample for speed
    n_sample = min(3000, len(X_test))
    idx = np.random.RandomState(seed).choice(len(X_test), n_sample, replace=False)
    X_sub = X_test[idx]
    y_sub = y_test[idx]

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    emb = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y_sub, cmap='coolwarm', alpha=0.6, s=15, edgecolors='none')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('RainTomorrow')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualize] Saved t-SNE plot: {save_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def generate_all_figures():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results_json = OUTPUT['results_json']

    draw_architecture_diagram(OUTPUT['plot_architecture'])
    if os.path.exists(results_json):
        draw_comparison_plot(results_json, OUTPUT['plot_comparison'])
    draw_uncertainty_decomposition(results_json, OUTPUT['plot_uncertainty'])
    draw_reliability_diagram(results_json, OUTPUT['plot_reliability'])
    draw_tsne_plot(OUTPUT['plot_tsne'])

    # Ablation and sensitivity if available
    abl_json = OUTPUT['ablation_results']
    if os.path.exists(abl_json):
        draw_ablation_plot(abl_json, OUTPUT['plot_ablation'])
    sens_json = OUTPUT['sensitivity_results']
    if os.path.exists(sens_json):
        draw_sensitivity_plot(sens_json, OUTPUT['plot_sensitivity'])

    print("\n[Visualize] All figures generated.")


if __name__ == "__main__":
    generate_all_figures()
