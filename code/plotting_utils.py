import os
import sys
sys.path.insert(1, '../')
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
import matplotlib.lines as mlines



def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_naive_coverage(df, alpha, plot_title):
    col = ["indianred"]
    sns.set_theme(font_scale=1.5, style='white', palette=col, rc={'lines.linewidth': 3})
    ylab = "coverage"
    xlab = "$n_{LLM}~/~n_{human}$"
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x=xlab, y=ylab, alpha=0.8, ax=ax)
    plt.ylim([-0.05,1])
    plt.axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.8) 
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(plot_title, bbox_inches='tight')
    plt.show()


def plot_eff_sample_size(df, plot_title):
    df = df[df.estimator != "LLM only"]
    ylab = "$n_{\mathrm{effective}}$"
    xlab = "$n_{\mathrm{human}}$"
    col = ['tab:orange', "tab:green", "tab:blue"]
    desired_order = ["confidence-driven", "human + LLM (non-adaptive)", "human only"]
    markers = ["o", "s", "D"]  # Different marker shapes
    sns.set_theme(font_scale=2.2, style='white', palette=col, rc={'lines.linewidth': 3})
    df_mean = df.groupby([xlab, 'estimator'])[ylab].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=df_mean, x=xlab, y=ylab, hue='estimator', hue_order=desired_order, ax=ax,
                 style='estimator', markers=markers, dashes=[(2, 2)], linewidth=2, alpha=0.7, legend=False)
    sns.scatterplot(data=df_mean, x=xlab, y=ylab, hue='estimator', hue_order=desired_order, ax=ax, s=70, alpha=0.7, edgecolor='black', style='estimator', markers=markers)
    ax.legend().remove()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(plot_title, bbox_inches='tight')
    plt.show()


def plot_coverage(df, alpha, plot_title):
    temp_df = df.copy()
    llm_only_mean = df[df.estimator == "LLM only"]["coverage"].mean()
    temp_df.loc[temp_df.estimator == "LLM only", "coverage"] = llm_only_mean
    col = ['tab:orange', "tab:green", "tab:blue", "tab:red"]
    markers = {"confidence-driven": "o", "human + LLM (non-adaptive)": "s", "human only": "D", "LLM only": "^"}  # Map markers to estimators
    sns.set_theme(font_scale=2.2, style='white', palette=col, rc={'lines.linewidth': 3})
    ylab = "coverage"
    xlab = "$n_{\mathrm{human}}$"
    fig, ax = plt.subplots()
    df_mean = temp_df.groupby([xlab, 'estimator'])[ylab].mean().reset_index()
    desired_order = ["confidence-driven", "human + LLM (non-adaptive)", "human only", "LLM only"]
    sns.lineplot(data=df_mean, x=xlab, y=ylab, hue='estimator', hue_order=desired_order, ax=ax,
                 style='estimator', markers=markers, dashes=[(2, 2)], linewidth=2, alpha=0.7, legend=False)
    sns.scatterplot(data=df_mean, x=xlab, y=ylab, hue="estimator", hue_order=desired_order, ax=ax, s=70, alpha=0.7, edgecolor='black', style='estimator', markers=markers)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend().remove()
    plt.ylim([-0.05,1.05])
    plt.axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.7) 
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.savefig(plot_title, bbox_inches='tight')
    plt.show()


def plot_intervals(df, true_val, num_trials, estimand_title, plot_title, n_ind=-1):
    num_ints = 5
    theta_true = true_val
    inds = np.random.choice(num_trials, num_ints)
    ns = df["$n_{\mathrm{human}}$"].unique()
    desired_order = ["confidence-driven", "human + LLM (non-adaptive)", "human only", "LLM only"]
    col = ['tab:orange', "tab:green", "tab:blue", "tab:red"]
    n_example = ns[n_ind]
    ints = [ [] for _ in range(len(desired_order)) ]
    for i in range(len(desired_order)):
        for j in range(num_ints):
            ind = inds[j]
            ints[i].append([df[(df.estimator == desired_order[i]) & (df["$n_{\mathrm{human}}$"] == n_example)].iloc[ind].lb, df[(df.estimator == desired_order[i]) & (df["$n_{\mathrm{human}}$"] == n_example)].iloc[ind].ub])
    gap = 0.03
    start = [0.5, 0.35, 0.2, 0.05]
    linewidth_inner = 8
    linewidth_outer = 9
    lighten_const = 0.5
    foreground_col = "black"
    sns.set_theme(font_scale=2.2, style='white', palette=col, rc={'lines.linewidth': 3})
    fig, ax = plt.subplots()
    ax.axvline(true_val, color='gray', linestyle='dashed')
    for i in reversed(range(num_ints)):
        for j in range(len(desired_order)):
            ax.plot([ints[j][i][0] , ints[j][i][1] ],[start[j]+i*gap,start[j]+i*gap], linewidth=linewidth_inner, color=lighten_color(col[j],lighten_const), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=foreground_col), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=foreground_col), pe.Normal()],  solid_capstyle='butt')
    ax.set_xlabel(estimand_title, fontsize=22)
    ax.set_yticks([])
    sns.despine(top=True, right=True, left=True)
    plt.tight_layout()
    plt.savefig(plot_title, bbox_inches='tight')
    plt.show()


def save_legend(df, alpha, legend_filename):
    temp_df = df.copy()
    llm_only_mean = df[df.estimator == "LLM only"]["coverage"].mean()
    temp_df.loc[temp_df.estimator == "LLM only", "coverage"] = llm_only_mean
    col = ['tab:orange', "tab:green", "tab:blue", "tab:red"]
    markers = {"confidence-driven": "o", "human + LLM (non-adaptive)": "s", "human only": "D", "LLM only": "^"}
    sns.set_theme(font_scale=2.2, style='white', palette=col, rc={'lines.linewidth': 3})
    ylab = "coverage"
    xlab = "$n_{\mathrm{human}}$"
    
    handles = [
        mlines.Line2D([], [], color=col[i], marker=markers[estimator], linestyle='None', 
                      markersize=10, markeredgewidth=1.5, markeredgecolor='black', alpha=0.7)
        for i, estimator in enumerate(markers.keys())
    ]
    
    fig, ax = plt.subplots(figsize=(8, 2))  # Adjust the figsize to control the layout
    legend = ax.legend(handles, markers.keys(), loc='center', ncol=len(markers), frameon=True, 
                       framealpha=0.2, edgecolor='black')
    ax.axis('off')
    
    plt.savefig(legend_filename, bbox_inches='tight', pad_inches=0)
    plt.close()