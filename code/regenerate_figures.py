"""
Regenerate Figures 2-6 without in-figure titles.
Figure order (by appearance in paper):
  Fig 2: Method comparison
  Fig 3: Ablation study
  Fig 4: Sensitivity analysis
  Fig 5: Robustness analysis (was fig6)
  Fig 6: Uncertainty analysis (was fig5)
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')
FIG_PAPER = os.path.join(BASE, 'paper', 'figures')
FIG_PLOTS = os.path.join(BASE, 'results', 'plots')
os.makedirs(FIG_PAPER, exist_ok=True)
os.makedirs(FIG_PLOTS, exist_ok=True)


def save_both(fig, name):
    for d in (FIG_PAPER, FIG_PLOTS):
        fig.savefig(os.path.join(d, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


# ---- Figure 2: Method Comparison ----
def fig2_comparison():
    print("[Fig 2] Method comparison")
    with open(os.path.join(RES, 'main_results_v3.json')) as f:
        data = json.load(f)
    agg = data.get('aggregated', {})
    methods = ['EDL-Fixed', 'LogisticRegression', 'RandomForest', 'XGBoost',
               'LSTM', 'GRU', 'BNN', 'MCDropout']
    metrics = [('accuracy', 'Accuracy'), ('f1_macro', 'F1-Macro'),
               ('ece', 'ECE'), ('uncertainty_auroc', 'Unc-AUROC')]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        means, stds, valid = [], [], []
        for m in methods:
            if m in agg and metric in agg[m]:
                means.append(agg[m][metric]['mean'])
                stds.append(agg[m][metric].get('std', 0))
                valid.append(m)
        x = np.arange(len(valid))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors[:len(valid)],
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(valid, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(label)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if means:
            best = int(np.argmin(means)) if metric == 'ece' else int(np.argmax(means))
            bars[best].set_edgecolor('black'); bars[best].set_linewidth(2)
    plt.tight_layout()
    save_both(fig, 'fig2_comparison.png')


# ---- Figure 3: Ablation ----
def fig3_ablation():
    print("[Fig 3] Ablation study")
    path = os.path.join(RES, 'ablation_results.json')
    if not os.path.exists(path):
        print("  ablation_results.json not found, skip"); return
    with open(path) as f:
        data = json.load(f)
    components = data.get('components', data)
    variants = list(components.keys()) if isinstance(components, dict) else []
    if not variants:
        print("  No ablation data, skip"); return
    metrics = [('accuracy', 'Accuracy'), ('f1_macro', 'F1-Macro'),
               ('ece', 'ECE'), ('uncertainty_auroc', 'Unc-AUROC')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        means, valid = [], []
        for v in variants:
            d = components[v] if isinstance(components[v], dict) else {}
            if metric in d:
                means.append(d[metric])
                valid.append(v)
        if not means:
            continue
        x = np.arange(len(valid))
        bars = ax.bar(x, means, color='steelblue',
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(valid, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(label)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        best = int(np.argmin(means)) if metric == 'ece' else int(np.argmax(means))
        bars[best].set_edgecolor('black'); bars[best].set_linewidth(2)
    plt.tight_layout()
    save_both(fig, 'fig3_ablation.png')


# ---- Figure 4: Sensitivity ----
def fig4_sensitivity():
    print("[Fig 4] Sensitivity analysis")
    csv_path = os.path.join(RES, 'sensitivity_summary_v2.csv')
    json_path = os.path.join(RES, 'sensitivity_results.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            sdata = json.load(f)
        # data is nested under 'raw' key
        raw = sdata.get('raw', sdata)
        params = list(raw.keys())
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        params = df['parameter'].unique().tolist()
        raw = None
    else:
        print("  No sensitivity data, skip"); return

    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.5))
    if len(params) == 1:
        axes = [axes]
    for ax, pname in zip(axes, params):
        if raw is not None:
            pdict = raw.get(pname, {})
            # Sort by numeric value of parameter
            sorted_items = sorted(pdict.items(),
                                  key=lambda x: float(x[0]) if str(x[0]).replace('.','',1).isdigit() else 0)
            vals = [float(k) for k, _ in sorted_items]
            f1s = [v.get('f1_macro', v.get('f1_macro_mean', 0)) for _, v in sorted_items]
            ax.plot(vals, f1s, marker='o', color='darkgreen', linewidth=2, markersize=6)
            ax.set_xticks(vals)
            ax.set_xticklabels([str(v) for v in vals], rotation=30, ha='right', fontsize=8)
        else:
            sub = df[df['parameter'] == pname].sort_values('value')
            ax.plot(sub['value'], sub['f1_macro'], marker='o', color='darkgreen', linewidth=2, markersize=6)
        ax.set_xlabel(pname)
        ax.set_ylabel('F1-Macro')
        ax.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_both(fig, 'fig4_sensitivity.png')


# ---- Figure 5: Robustness (was fig6) ----
def fig5_robustness():
    print("[Fig 5] Robustness analysis")
    csv_path = os.path.join(RES, 'robustness_results_v2.csv')
    if not os.path.exists(csv_path):
        print("  robustness_results_v2.csv not found, skip"); return
    df = pd.read_csv(csv_path)
    noise = df[df['perturbation'] == 'Gaussian_Noise'].sort_values('level')
    missing = df[df['perturbation'] == 'Feature_Missing'].sort_values('level')
    clean = df[df['perturbation'] == 'Clean']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: accuracy & F1
    for d, label, marker in [(noise, 'Gaussian Noise', 'o'), (missing, 'Feature Missing', 's')]:
        if len(d) > 0:
            ax1.plot(d['level'], d['accuracy'], marker=marker, color='steelblue', linewidth=2, label=f'Acc ({label})')
            ax1.plot(d['level'], d['f1_macro'], marker=marker, color='coral', linewidth=2, linestyle='--', label=f'F1 ({label})')
    if len(clean) > 0:
        ax1.axhline(clean.iloc[0]['accuracy'], color='steelblue', linestyle=':', alpha=0.5)
        ax1.axhline(clean.iloc[0]['f1_macro'], color='coral', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Perturbation Level')
    ax1.set_ylabel('Score')
    ax1.legend(fontsize=7, loc='lower left')
    ax1.grid(linestyle='--', alpha=0.5)

    # Right: S and Unc-AUROC
    for d, label, marker in [(noise, 'Gaussian Noise', 'o'), (missing, 'Feature Missing', 's')]:
        if len(d) > 0:
            ax2.plot(d['level'], d['S_mean'], marker=marker, color='seagreen', linewidth=2, label=f'S ({label})')
            ax2.plot(d['level'], d['uncertainty_auroc'], marker=marker, color='purple', linewidth=2, linestyle='--', label=f'Unc-AUROC ({label})')
    if len(clean) > 0:
        ax2.axhline(clean.iloc[0]['S_mean'], color='seagreen', linestyle=':', alpha=0.5)
        ax2.axhline(clean.iloc[0]['uncertainty_auroc'], color='purple', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Perturbation Level')
    ax2.set_ylabel('Value')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_both(fig, 'fig5_robustness.png')


# ---- Figure 6: Uncertainty analysis (was fig5) ----
def fig6_uncertainty():
    print("[Fig 6] Uncertainty analysis")
    json_path = os.path.join(RES, 'uncertainty_analysis_v2.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(RES, 'uncertainty_analysis.json')
    if not os.path.exists(json_path):
        print("  uncertainty_analysis.json not found, skip"); return
    with open(json_path) as f:
        udata = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): H_T mean for correct vs incorrect (bar chart with error bars)
    ht = udata.get('H_total', {})
    correct_mean = ht.get('correct_mean', 0)
    error_mean = ht.get('error_mean', 0)
    ht_std = ht.get('std', 0)
    bars = ax1.bar(['Correct', 'Incorrect'], [correct_mean, error_mean],
                   yerr=[ht_std, ht_std], capsize=5,
                   color=['steelblue', 'coral'], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Total Uncertainty (mean)')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    # Add value labels
    for bar, val in zip(bars, [correct_mean, error_mean]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Panel (b): Selective prediction curve
    rra = udata.get('rejection_rate_analysis', [])
    if rra:
        rej = [r['rejection_rate'] for r in rra]
        acc = [r['accuracy_retained'] for r in rra]
        rand = [r.get('random_baseline', 0) for r in rra]
        oracle = [r.get('oracle_baseline', 1) for r in rra]
        ax2.plot(rej, acc, marker='o', color='darkgreen', linewidth=2, label='EDL-Fixed')
        ax2.plot(rej, rand, linestyle='--', color='gray', linewidth=1, label='Random')
        ax2.plot(rej, oracle, linestyle=':', color='black', linewidth=1, label='Oracle')
    ax2.set_xlabel('Rejection Rate')
    ax2.set_ylabel('Retained Accuracy')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_both(fig, 'fig6_uncertainty.png')


if __name__ == '__main__':
    fig2_comparison()
    fig3_ablation()
    fig4_sensitivity()
    fig5_robustness()
    fig6_uncertainty()
    print("\nAll figures regenerated (no titles, correct order).")
