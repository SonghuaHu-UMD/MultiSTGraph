import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib as mpl
from libcity.model import loss

pd.options.mode.chained_assignment = None
results_path = r'C:\Users\huson\PycharmProjects\MultiSTGraph\libcity\cache'

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Read metrics of multiple models
filenames = glob.glob(results_path + r"\*")
filenames = [ec for ec in filenames if 'dataset_cache' not in ec]
all_results = pd.DataFrame()
for ec in filenames:
    nec = glob.glob(ec + '\\evaluate_cache\\*.csv')
    if len(nec) > 0:
        nec = nec[0]
        fec = pd.read_csv(nec)
        fec['Model_name'] = nec.split('_')[7]
        all_results = all_results.append(fec)
all_results = all_results.reset_index()
all_results_avg = all_results.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
all_results_avg.to_csv(r"D:\ST_Graph\Results\results_avg_single_hourly_nonst.csv")

# Plot each step for all models
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Set2.colors)
fig, ax = plt.subplots(figsize=(10, 6))
for em in set(all_results_avg.head(8)['Model_name']):
    temp = all_results[all_results['Model_name'] == em]
    ax.plot(temp['index'], temp['masked_MAPE'], label=em, lw=2)
plt.legend(ncol=3)
plt.tight_layout()

# Plot a county sub
filenames = glob.glob(results_path + r"\63589\evaluate_cache\*.npz")
Predict_R = np.load(filenames[0])
sh = Predict_R['prediction'].shape
print(sh)  # testing length, prediction length, number of nodes, output dim
# fig, ax = plt.subplots(figsize=(12, 6))
# for kk in range(0, Predict_R['prediction'].shape[2]):
#     ax.plot(Predict_R['prediction'][:, 0, kk, 0], label='prediction')
#     ax.plot(Predict_R['truth'][:, 0, kk, 0], label='truth')
# plt.legend()
# plt.tight_layout()

# Re-transform the data
ct_visit_mstd = pd.read_pickle(r'D:\ST_Graph\Results\cts_visit_mstd.pkl')
ct_visit_mstd = ct_visit_mstd.sort_values(by='CTSFIPS').reset_index(drop=True)
ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
ct_id = np.tile(ct_visit_mstd[['CTSFIPS']].values, (sh[0], sh[1], 1, sh[3]))
ahead_step = np.tile(np.expand_dims(np.array(range(0, 24)), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
Predict_Real = pd.DataFrame(
    {'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(), 'All_m': ct_ma.flatten(),
     'All_std': ct_sa.flatten(), 'CTSFIPS': ct_id.flatten(), 'ahead_step': ahead_step.flatten()})
Predict_Real['prediction_t'] = Predict_Real['prediction'] * Predict_Real['All_std'] + Predict_Real['All_m']
Predict_Real['truth_t'] = Predict_Real['truth'] * Predict_Real['All_std'] + Predict_Real['All_m']
Predict_Real['mae'] = abs(Predict_Real['prediction_t'] - Predict_Real['truth_t'])
Predict_Real['mape'] = abs(Predict_Real['prediction_t'] - Predict_Real['truth_t']) / Predict_Real['truth_t']
Predict_Real.loc[Predict_Real['truth_t'] < 1, 'mape'] = np.nan  # mape is nan if truth is zero
print(Predict_Real['mape'].mean())
print(Predict_Real['mae'].mean())
Predict_Real.groupby(['ahead_step']).mean()[['mae']].plot()

fig, ax = plt.subplots(figsize=(12, 6))
for kk in list(ct_visit_mstd['CTSFIPS'])[0:1]:
    temp = Predict_Real[(Predict_Real['CTSFIPS'] == kk) & (Predict_Real['ahead_step'] == 1)]
    ax.plot(temp['prediction_t'], label='prediction')
    ax.plot(temp['truth_t'], label='truth')
plt.legend()
plt.tight_layout()
