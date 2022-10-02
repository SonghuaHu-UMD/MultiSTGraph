import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import matplotlib as mpl
from libcity.model import loss
from sklearn.metrics import r2_score, explained_variance_score
import datetime
import math
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

pd.options.mode.chained_assignment = None
results_path = r'D:\ST_Graph\results_record\\'

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Plot prediction results
# Settings
t_s = datetime.datetime(2019, 1, 1)  # datetime.datetime(2019, 3, 1)
t_e = datetime.datetime(2019, 6, 1)  # datetime.datetime(2019, 7, 1)
area_c, n_steps, m_name, sunit = '_DC', 24, 'MultiATGCN', 'CTractFIPS'
time_sp = t_s.strftime('%Y%m%d') + t_e.strftime('%m%d') + area_c
t_st = t_s + datetime.timedelta(days=28)
t_et = t_e - datetime.timedelta(hours=1)  # print((t_et - t_st).days * 0.15 * 24)
split_time = t_et - datetime.timedelta(hours=math.ceil((t_et - t_st).days * 0.15 * 24))
filenames = glob.glob(results_path + r"%s steps\Baselines\%s\*" % (n_steps, time_sp))
for kk in filenames:
    filename = glob.glob(kk + r"\\evaluate_cache\*.npz")
    model_name = glob.glob(kk + '\\model_cache\\*.m')[0].split('\\')[-1].split('_')[0]
    if model_name == m_name: break

# Get data
Predict_R = np.load(filename[0])
sh = Predict_R['prediction'].shape
print(sh)  # no of batches, output_window, no of nodes, output dim
ct_visit_mstd = pd.read_pickle(r'D:\ST_Graph\Results\%s_%s_visit_mstd.pkl' % (sunit, time_sp)).sort_values(
    by=sunit).reset_index(drop=True)
ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
ct_id = np.tile(ct_visit_mstd[[sunit]].values, (sh[0], sh[1], 1, sh[3]))
ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
ht_id = np.tile(np.expand_dims(np.array(range(0, sh[0])), axis=(1, 2, 3)), (1, sh[1], sh[2], sh[3]))
P_R = pd.DataFrame(
    {'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(), 'A_m': ct_ma.flatten(),
     'A_std': ct_sa.flatten(), sunit: ct_id.flatten(), 'ahead_step': ahead_step.flatten(), 'hour_id': ht_id.flatten()})
P_R['prediction_t'] = P_R['prediction'] * P_R['A_std'] + P_R['A_m']
P_R['truth_t'] = P_R['truth'] * P_R['A_std'] + P_R['A_m']
P_R.loc[P_R['prediction_t'] < 0, 'prediction_t'] = 0
# Add time
P_R['Date'] = split_time + pd.to_timedelta(P_R['hour_id'] + 1, 'h')
# P_R = P_R[P_R['Date'] <= t_et].reset_index(drop=True)
# Add external variables
external = pd.read_pickle(r'D:\ST_Graph\Results\weather_2019_bmc.pkl')
external['Date'] = pd.to_datetime(external['DATE']).dt.tz_localize(None)
P_R = P_R.merge(external[['Date', 'wind', 'temp', 'rain', 'snow', 'vis']], on='Date', how='left')
holidays = calendar().holidays(start=P_R['Date'].dt.date.min(), end=P_R['Date'].dt.date.max())
P_R['holiday'] = P_R['Date'].dt.date.astype('datetime64').isin(holidays).astype(int)
P_R['weekend'] = P_R['Date'].dt.dayofweek.isin([5, 6]).astype(int)

# Plot a county
P_R['MAPE'] = abs(P_R['prediction_t'] - P_R['truth_t']) / P_R['truth_t']
P_R_mape = P_R[P_R['truth_t'] > 10].groupby([sunit]).mean()['MAPE'].sort_values()
P_R_truth = P_R.groupby([sunit]).mean()['truth_t'].sort_values()
fig, ax = plt.subplots(figsize=(12, 6))
for kk in ['51013980200']:
    temp = P_R[(P_R[sunit] == kk) & (P_R['ahead_step'] == 0)]
    temp = temp.set_index(temp['Date'])
    ax.plot(temp['prediction_t'], label='prediction')
    ax.plot(temp['truth_t'], label='truth')
    ax.plot(temp['rain'], label='rain')
    ax.plot(temp['holiday'], label='holiday')
    ax.plot(temp['weekend'], label='weekend')
plt.legend()
plt.tight_layout()


