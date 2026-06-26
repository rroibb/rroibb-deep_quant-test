import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

FONT_PATH = None
for p in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc']:
    if os.path.exists(p):
        FONT_PATH = p
        break

if FONT_PATH:
    plt.rcParams['font.family'] = FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#2E86DE', '#A23B72', '#F18F01', '#5ABCB9', '#E4572E', '#3B1F2B']


def plot_cumulative_returns(results, output_dir='.'):
    fig, ax = plt.subplots(figsize=(14, 7))
    bench = next(iter(results.values()))['Cum_Benchmark']
    ax.plot(bench.index, bench.values * 100, color='gray', linewidth=2,
            linestyle='--', alpha=0.7, label=f'Benchmark')
    for i, (name, daily) in enumerate(results.items()):
        cum = daily['Cum_Strategy'] * 100
        ax.plot(cum.index, cum.values, color=COLORS[i], linewidth=2, label=name)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_title('Strategy Cumulative Returns Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(loc='upper left', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'cumulative_returns.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Chart] Cumulative Returns -> {path}")
    return path


def plot_drawdown(results, output_dir='.'):
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, daily) in enumerate(results.items()):
        cum = daily['Cum_Strategy']
        cum_max = cum.cummax()
        dd = (cum / cum_max - 1) * 100
        ax.plot(dd.index, dd.values, color=COLORS[i], linewidth=1.5,
                label=f'{name}')
    ax.set_title('Strategy Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='lower left', fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'drawdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Chart] Drawdown -> {path}")
    return path


def plot_metrics_bar(results, output_dir='.'):
    names = list(results.keys())
    metrics = {'Total Return (%)': [], 'Ann Return (%)': [], 'Sharpe': [], 'Max DD (%)': []}
    for name, daily in results.items():
        total = daily['Cum_Strategy'].iloc[-1] * 100
        ann = daily['Strategy_Ret'].mean() * 240 * 100
        vol = daily['Strategy_Ret'].std() * np.sqrt(240)
        sharpe = (daily['Strategy_Ret'].mean() * 240 - 0.03) / vol if vol > 1e-8 else 0
        cum_max = daily['Cum_Strategy'].cummax()
        mdd = (cum_max - daily['Cum_Strategy']).max() * 100
        metrics['Total Return (%)'].append(total)
        metrics['Ann Return (%)'].append(ann)
        metrics['Sharpe'].append(sharpe)
        metrics['Max DD (%)'].append(mdd)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (metric, values) in enumerate(metrics.items()):
        ax = axes[idx]
        bars = ax.bar(names, values, color=COLORS[:len(names)], edgecolor='white')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values)*0.02),
                    f'{val:.2f}', ha='center', fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, 'metrics_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Chart] Metrics -> {path}")
    return path


def plot_monthly_returns(daily, name, output_dir='.'):
    monthly = daily['Strategy_Ret'].resample('ME').sum() * 100
    if len(monthly) < 2:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = ['#FF6B6B' if v < 0 else '#51CF66' for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors_bar, edgecolor='white')
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{name} - Monthly Returns', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(monthly.values):
        ax.text(i, v + (0.5 if v >= 0 else -1.5), f'{v:.1f}%', ha='center', fontsize=8)
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'n')
    path = os.path.join(output_dir, f'monthly_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Chart] Monthly({name}) -> {path}")
    return path


def plot_dashboard(results, output_dir='.'):
    """综合仪表盘: 累计收益+回撤+月度收益+指标 四合一"""
    names = list(results.keys())
    bench = next(iter(results.values()))['Cum_Benchmark']

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30,
                          height_ratios=[2.5, 2, 1.5])

    # ── 左上: 累计收益对比 (大图, 占2列) ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(bench.index, bench.values * 100, color='gray', linewidth=2.5,
             linestyle='--', alpha=0.6, label='Benchmark (Equal-Weight Index)')
    for i, (name, daily) in enumerate(results.items()):
        cum = daily['Cum_Strategy'] * 100
        ax1.plot(cum.index, cum.values, color=COLORS[i], linewidth=2.2,
                 label=name, marker='.', markevery=len(cum)//8, markersize=5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax1.grid(True, alpha=0.3)

    # 在终点标注数值
    for i, (name, daily) in enumerate(results.items()):
        final_val = daily['Cum_Strategy'].iloc[-1] * 100
        ax1.annotate(f'{final_val:.1f}%', xy=(daily.index[-1], final_val),
                     xytext=(5, 5), textcoords='offset points',
                     color=COLORS[i], fontsize=8, fontweight='bold')

    # ── 右上: 关键指标汇总表 (占1列) ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    bench_ann = bench.mean() * 240 * 100
    header = ['Strategy', 'Total', 'Ann.', 'Sharpe', 'MDD', 'Excess']
    rows = [header]
    for name, daily in results.items():
        total = daily['Cum_Strategy'].iloc[-1] * 100
        ann = daily['Strategy_Ret'].mean() * 240 * 100
        vol = daily['Strategy_Ret'].std() * np.sqrt(240)
        sharpe = (daily['Strategy_Ret'].mean()*240-0.03)/vol if vol > 1e-8 else 0
        mdd = (daily['Cum_Strategy'].cummax()-daily['Cum_Strategy']).max()*100
        excess = total - bench.iloc[-1]*100
        short = name.split('(')[0].strip().replace('Multimodal Fusion','Multimodal')
        rows.append([short, f'{total:.1f}%', f'{ann:.1f}%', f'{sharpe:.2f}',
                     f'{mdd:.1f}%', f'{excess:+.1f}%'])
    rows.append(['Benchmark', f'{bench.iloc[-1]*100:.1f}%', '', '', '', ''])

    tbl = ax2.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for key, cell in tbl.get_celld().items():
        row, col = key
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#2E86DE')
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_fontsize(9)
        elif row == len(rows)-1:
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(fontweight='bold')
            cell.set_fontsize(9)
        else:
            cell.set_facecolor('#FAFAFA' if row % 2 == 0 else 'white')
            cell.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    # ── 左下: 回撤曲线 (占1列) ──
    ax3 = fig.add_subplot(gs[1, :])
    for i, (name, daily) in enumerate(results.items()):
        cum = daily['Cum_Strategy']
        dd = (cum / cum.cummax() - 1) * 100
        ax3.fill_between(dd.index, dd.values, 0,
                         color=COLORS[i], alpha=0.15)
        ax3.plot(dd.index, dd.values, color=COLORS[i], linewidth=1.5,
                 label=f'{name.split("(")[0].strip()}')
    ax3.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.legend(loc='lower left', fontsize=9, framealpha=0.9, ncol=3)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax3.grid(True, alpha=0.3)

    # ── 右下: 月度收益分组对比 ──
    ax4 = fig.add_subplot(gs[2, :])
    all_monthly = {}
    month_labels = None
    for name, daily in results.items():
        m = daily['Strategy_Ret'].resample('ME').sum() * 100
        all_monthly[name] = m
        if month_labels is None:
            month_labels = [d.strftime('%Y-%m') for d in m.index]

    if month_labels and len(month_labels) > 1:
        x = np.arange(len(month_labels))
        n = len(results)
        bar_w = 0.8 / n
        for i, (name, monthly) in enumerate(all_monthly.items()):
            vals = monthly.values
            offset = (i - n/2 + 0.5) * bar_w
            short = name.split('(')[0].strip().replace('Multimodal Fusion','Multimodal')
            bars = ax4.bar(x + offset, vals, bar_w, label=short,
                           color=COLORS[i], edgecolor='white', alpha=0.85)
            for bar, val in zip(bars, vals):
                if abs(val) > 2:
                    ax4.text(bar.get_x() + bar.get_width()/2,
                             val + (0.5 if val >= 0 else -1.0),
                             f'{val:.1f}', ha='center', fontsize=7)

        ax4.set_xticks(x)
        ax4.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
        ax4.set_title('Monthly Returns by Strategy', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Return (%)', fontsize=11)
        ax4.axhline(y=0, color='black', linewidth=0.5)
        ax4.legend(loc='upper left', fontsize=9, ncol=3, framealpha=0.9)
        ax4.grid(axis='y', alpha=0.3)

    fig.suptitle('DeepQuant Multi-Model Backtest Dashboard',
                 fontsize=20, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, 'dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [Chart] Dashboard -> {path}")
    return path


def plot_all(results, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    p = plot_dashboard(results, output_dir)
    if p:
        paths.append(p)
    p = plot_cumulative_returns(results, output_dir)
    if p:
        paths.append(p)
    p = plot_drawdown(results, output_dir)
    if p:
        paths.append(p)
    p = plot_metrics_bar(results, output_dir)
    if p:
        paths.append(p)
    for name, daily in results.items():
        p = plot_monthly_returns(daily, name, output_dir)
        if p:
            paths.append(p)
    return paths