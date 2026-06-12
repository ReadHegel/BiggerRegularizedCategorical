import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from collections import defaultdict


api = wandb.Api()
runs = api.runs("aleksanderwojsz-team/rl-project")

# benchmark_idx (step0/step1/step2/step3), model size, arch -> {timestep: [values]}
data = defaultdict(lambda: defaultdict(list))
for run in runs:
    parts = run.name.split('_')
    if len(parts) >= 5 and parts[0] == 'DMC' and parts[1] == 'DOGS':
        arch, size = parts[2], parts[3]
        if size in ['2mln', '8mln', '32mln'] and arch in ['bro', 'flashsac', 'xqc', 'simbaV2']:
            keys = ["timestep"]
            for i in range(4):
                keys.append(f"seed{i}/return")
                keys.append(f"seed{i}/critic_pnorm")
                keys.append(f"seed{i}/actor_pnorm")
            history = run.history(keys=keys, pandas=False)
            for step in history:
                t = step["timestep"]
                rets = [step[f"seed{i}/return"] for i in range(4)]
                for b in range(4):
                    if rets[b] is not None:
                        key = (b, size, arch)
                        data[key][t].append(rets[b])
                
                key = ("averaged", size, arch)
                data[key][t].append(np.mean(rets))
                
                pnorms = [step[f"seed{i}/critic_pnorm"] for i in range(4) if f"seed{i}/critic_pnorm" in step and step[f"seed{i}/critic_pnorm"] is not None]
                if pnorms:
                    data[("critic_pnorm", size, arch)][t].append(np.mean(pnorms))
                    
                actor_pnorms = [step[f"seed{i}/actor_pnorm"] for i in range(4) if f"seed{i}/actor_pnorm" in step and step[f"seed{i}/actor_pnorm"] is not None]
                if actor_pnorms:
                    data[("actor_pnorm", size, arch)][t].append(np.mean(actor_pnorms))


###################### LLM generated:
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})
colors = {'bro': '#d62728', 'flashsac': '#2ca02c', 'xqc': '#1f77b4', 'simbaV2': '#ff7f0e'} # red, green, blue, orange
linestyles = {'bro': '-', 'flashsac': '--', 'xqc': '-.', 'simbaV2': ':'}
labels = {'bro': 'BroNet', 'flashsac': 'FlashSAC', 'xqc': 'XQC', 'simbaV2': 'SimbaV2'}
sizes = ['2mln', '8mln']
benchmarks = [0, 1, 2, 3, 'averaged']

task_names = {
    0: 'dog-stand',
    1: 'dog-walk',
    2: 'dog-trot',
    3: 'dog-run',
    'averaged': 'Average'
}

fig, axes = plt.subplots(5, 2, figsize=(9.5, 15), sharex='col', sharey=True)
fig.suptitle("Model Scaling Performance: DMC Dog", fontsize=18, y=0.99)

for row_idx, b in enumerate(benchmarks):
    row_title = task_names[b]
    for col_idx, size in enumerate(sizes):
        ax = axes[row_idx, col_idx]
        
        # Subplot Titles & Axis Labels
        display_sizes = {'2mln': '2M', '8mln': '8M', '32mln': '32M'}
        if row_idx == 0:
            ax.set_title(f"Network size: {display_sizes.get(size, size)}", fontsize=15, pad=10)
        if col_idx == 0:
            ax.set_ylabel(f"{row_title}\nEpisode return", fontsize=13)
        if row_idx == 4:
            ax.set_xlabel("Timesteps", fontsize=13)
            
        # Subtle grid for clean aesthetics
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_xlim(50000, 500000)
        
        for arch, color in colors.items():
            arch_data = data.get((b, size, arch), {})
            if not arch_data:
                continue
            
            timesteps = sorted(arch_data.keys())
            means = [np.mean(arch_data[t]) for t in timesteps]
            stds = [np.std(arch_data[t]) if len(arch_data[t]) > 1 else 0.0 for t in timesteps]
            

            
            ax.plot(
                timesteps, means, 
                label=labels[arch], 
                color=color, 
                linestyle=linestyles[arch], 
                linewidth=1.75
            )
            if any(s > 0 for s in stds):
                ax.fill_between(
                    timesteps, 
                    np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), 
                    color=color, 
                    alpha=0.12
                )
        
        # Consistent ticks spaced out nicely
        ax.set_xticks([100000, 200000, 300000, 400000, 500000])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e3:.0f}k" if x < 1e6 else f"{x/1e6:.1f}M"))
        ax.set_ylim(0, 1000)
        ax.set_yticks([0, 250, 500, 750, 1000])

# Single unified legend on the first subplot
axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='none')

plt.tight_layout()
plt.subplots_adjust(top=0.94)

script_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(script_dir, "step_vs_performance.png")
pdf_path = os.path.join(script_dir, "step_vs_performance.pdf")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

fig1, axes1 = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
fig1.suptitle("Model Scaling Performance: DMC Dog", fontsize=16, y=0.99)

for col_idx, size in enumerate(sizes):
    ax = axes1[col_idx]
    
    # Subplot Titles & Axis Labels
    ax.set_title(f"Network size: {display_sizes.get(size, size)}", fontsize=15, pad=10)
    if col_idx == 0:
        ax.set_ylabel("Episode return", fontsize=13)
    ax.set_xlabel("Timesteps", fontsize=13)
        
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xlim(50000, 500000)
    
    for arch, color in colors.items():
        arch_data = data.get(("averaged", size, arch), {})
        if not arch_data:
            continue
        
        timesteps = sorted(arch_data.keys())
        means = [np.mean(arch_data[t]) for t in timesteps]
        stds = [np.std(arch_data[t]) if len(arch_data[t]) > 1 else 0.0 for t in timesteps]
        

        
        ax.plot(
            timesteps, means, 
            label=labels[arch], 
            color=color, 
            linestyle=linestyles[arch], 
            linewidth=1.75
        )
        if any(s > 0 for s in stds):
            ax.fill_between(
                timesteps, 
                np.array(means) - np.array(stds), 
                np.array(means) + np.array(stds), 
                color=color, 
                alpha=0.12
            )
            
    ax.set_xticks([100000, 200000, 300000, 400000, 500000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e3:.0f}k" if x < 1e6 else f"{x/1e6:.1f}M"))
    ax.set_ylim(0, 1000)
    ax.set_yticks([0, 250, 500, 750, 1000])

# Single unified legend on the first subplot
axes1[0].legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='none')

fig1.tight_layout()
fig1.subplots_adjust(top=0.82)

fig1_png_path = os.path.join(script_dir, "figure1.png")
fig1_pdf_path = os.path.join(script_dir, "figure1.pdf")

fig1.savefig(fig1_png_path, dpi=300, bbox_inches='tight')
fig1.savefig(fig1_pdf_path, bbox_inches='tight')

# Generate the 2x2 Network diagnostics plot (critic_pnorm & actor_pnorm for 2M & 8M)
fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8.5))
fig2.suptitle("Network Parameter Norms during Training: DMC Dog (mean $\\pm$ std)", fontsize=16, y=0.99)

metrics = ['critic_pnorm', 'actor_pnorm']
metric_labels = {
    'critic_pnorm': 'Critic parameter norm',
    'actor_pnorm': 'Actor parameter norm'
}
display_sizes = {'2mln': '2M', '8mln': '8M'}

for row_idx, metric in enumerate(metrics):
    for col_idx, size in enumerate(sizes):
        ax = axes2[row_idx, col_idx]
        
        if row_idx == 0:
            ax.set_title(f"Network size: {display_sizes.get(size, size)}", fontsize=14, pad=10)
        if col_idx == 0:
            ax.set_ylabel(metric_labels[metric], fontsize=13)
        if row_idx == 1:
            ax.set_xlabel("Timesteps", fontsize=13)
            
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_xlim(50000, 500000)
        
        # No log scale needed for parameter norms
            
        for arch, color in colors.items():
            arch_data = data.get((metric, size, arch), {})
            if not arch_data:
                continue
                
            timesteps = sorted(arch_data.keys())
            means = [np.mean(arch_data[t]) for t in timesteps]
            stds = [np.std(arch_data[t]) if len(arch_data[t]) > 1 else 0.0 for t in timesteps]
            

                
            ax.plot(
                timesteps, means, 
                label=labels[arch], 
                color=color, 
                linestyle=linestyles[arch], 
                linewidth=1.75
            )
            if any(s > 0 for s in stds):
                ax.fill_between(
                    timesteps, 
                    np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), 
                    color=color, 
                    alpha=0.12
                )
                
        ax.set_xticks([100000, 200000, 300000, 400000, 500000])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e3:.0f}k" if x < 1e6 else f"{x/1e6:.1f}M"))

# Single unified legend on the first subplot
axes2[0, 0].legend(loc="best", frameon=True, framealpha=0.9, edgecolor='none')

fig2.tight_layout()
fig2.subplots_adjust(top=0.92)

fig2_png_path = os.path.join(script_dir, "parameter_norms.png")
fig2_pdf_path = os.path.join(script_dir, "parameter_norms.pdf")

fig2.savefig(fig2_png_path, dpi=300, bbox_inches='tight')
fig2.savefig(fig2_pdf_path, bbox_inches='tight')
######################
