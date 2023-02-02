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
import random
from itertools import cycle
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

random.seed(10)
pd.options.mode.chained_assignment = None
results_path = r'D:\ST_Graph\results_record\\'
plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})


def get_gp_data(filenames):
    filenames = [ec for ec in filenames if 'log' not in ec]
    all_results = pd.DataFrame()
    for ec in filenames:
        nec = glob.glob(ec + '\\evaluate_cache\\*.csv')
        model_name = glob.glob(ec + '\\model_cache\\*.m')
        if len(nec) > 0:
            fec = pd.read_csv(nec[0])
            fec['Model_name'] = model_name[0].split('\\')[-1].split('_')[0]
            fec['Model_time'] = datetime.datetime.fromtimestamp(os.path.getmtime(nec[0]))
            all_results = all_results.append(fec)
    all_results = all_results.reset_index()
    return all_results


def transfer_gp_data(filenames, ct_visit_mstd, s_small=10):
    m_m = []
    for kk in filenames:
        print(kk)
        filename = glob.glob(kk + r"\\evaluate_cache\*.npz")
        model_name = glob.glob(kk + '\\model_cache\\*.m')
        if len(model_name) > 0:
            model_name = model_name[0].split('\\')[-1].split('_')[0]
            print(model_name)
            Predict_R = np.load(filename[0])
            # drop the last batch
            pred = Predict_R['prediction'][:-16, :, :, :]
            truth = Predict_R['truth'][:-16, :, :, :]
            sh = pred.shape
            print(sh)  # no of batches, output_window, no of nodes, output dim
            ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
            ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
            ct_id = np.tile(ct_visit_mstd[[sunit]].values, (sh[0], sh[1], 1, sh[3]))
            ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
            P_R = pd.DataFrame({'prediction': pred.flatten(), 'truth': truth.flatten(),
                                'All_m': ct_ma.flatten(), 'All_std': ct_sa.flatten(), sunit: ct_id.flatten(),
                                'ahead_step': ahead_step.flatten()})
            P_R['prediction_t'] = P_R['prediction'] * P_R['All_std'] + P_R['All_m']
            P_R['truth_t'] = P_R['truth'] * P_R['All_std'] + P_R['All_m']
            P_R.loc[P_R['prediction_t'] < 0, 'prediction_t'] = 0

            # not consider small volume
            for rr in range(0, sh[1]):
                pr = P_R.loc[(P_R['ahead_step'] == rr) & (P_R['truth_t'] > s_small), 'prediction_t']
                tr = P_R.loc[(P_R['ahead_step'] == rr) & (P_R['truth_t'] > s_small), 'truth_t']
                m_m.append([model_name, rr, datetime.datetime.fromtimestamp(os.path.getmtime(filename[0])),
                            loss.masked_mae_np(pr, tr), loss.masked_mse_np(pr, tr), loss.masked_rmse_np(pr, tr),
                            r2_score(pr, tr), explained_variance_score(pr, tr), loss.masked_mape_np(pr, tr)])
        else:
            print(kk + '----NULL----')
    return m_m


########### Plot prediction time series
# Settings
t_s = datetime.datetime(2019, 1, 1)  # datetime.datetime(2019, 3, 1)
t_e = datetime.datetime(2019, 6, 1)  # datetime.datetime(2019, 7, 1)
area_c, n_steps, m_name, sunit, nfold = '_BM', 24, 'MultiATGCN', 'CTractFIPS', 'Final'
time_sp = t_s.strftime('%Y%m%d') + t_e.strftime('%m%d') + area_c
t_st = t_s + datetime.timedelta(days=28)
t_et = t_e - datetime.timedelta(hours=1)  # print((t_et - t_st).days * 0.15 * 24)
split_time = t_et - datetime.timedelta(hours=math.ceil((t_et - t_st).days * 0.15 * 24) + 24)
filenames = glob.glob(r'D:\ST_Graph\results_record\\plot\\' + r"%s steps\%s\%s\*" % (n_steps, nfold, time_sp))
for kk in filenames:
    filename = glob.glob(kk + r"\\evaluate_cache\*.npz")
    model_name = glob.glob(kk + '\\model_cache\\*.m')[0].split('\\')[-1].split('_')[0]
    if model_name == m_name: break

# Get data
Predict_R = np.load(filename[0])
sh = Predict_R['prediction'].shape
print(sh)
ct_visit_mstd = pd.read_pickle(r'.\other_data\%s_%s_visit_mstd.pkl' % (sunit, time_sp)).sort_values(
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
P_R['Date'] = split_time + pd.to_timedelta(P_R['hour_id'], 'h') + pd.to_timedelta(P_R['ahead_step'], 'h')
P_R = P_R[P_R['Date'] <= t_et - datetime.timedelta(hours=24)].reset_index(drop=True)
# Add external variables
external = pd.read_pickle(r'D:\ST_Graph\Results\weather_2019_bmc.pkl')
external['Date'] = pd.to_datetime(external['DATE']).dt.tz_localize(None)
P_R = P_R.merge(external[['Date', 'wind', 'temp', 'rain', 'snow', 'vis']], on='Date', how='left')
holidays = calendar().holidays(start=P_R['Date'].dt.date.min(), end=P_R['Date'].dt.date.max())
P_R['holiday'] = P_R['Date'].dt.date.astype('datetime64').isin(holidays).astype(int)
P_R['weekend'] = P_R['Date'].dt.dayofweek.isin([5, 6]).astype(int)

# Plot the top and last census tracts
P_R['MAPE'] = abs(P_R['prediction_t'] - P_R['truth_t']) / P_R['truth_t']
rank_gp = P_R[P_R['truth_t'] > 6].groupby([sunit]).mean()['MAPE'].sort_values().reset_index()
if area_c == '_BM':
    last_3 = list(rank_gp[sunit][-7:-4])
    top_3 = list(rank_gp[sunit][0:3])
else:
    last_3 = list(rank_gp[sunit][-3:])
    top_3 = list(rank_gp[sunit][0:1]) + list(rank_gp[sunit][4:6])
# Create rectangle x coordinates
startTime = datetime.datetime(2019, 5, 16)
endTime = startTime + datetime.timedelta(days=7)
# convert to matplotlib date representation
start = mdates.date2num(startTime)
end = mdates.date2num(endTime)
width = end - start
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 7))
dd = 1
axs = ax.flatten()
ccount = 0
colors = plt.cm.coolwarm(np.linspace(0, 1, 2))
# Plot rectangle
for idx in top_3:
    temp_test = P_R[(P_R[sunit] == idx) & ((P_R['ahead_step'] == dd))]
    temp_test = temp_test.set_index(temp_test['Date'])
    axs[ccount].plot(temp_test['truth_t'], '--', label='Truth', color=colors[0], lw=1.5)
    axs[ccount].plot(temp_test['prediction_t'], '-', color=colors[1], label='Prediction', lw=1.5)
    axs[ccount].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    axs[ccount].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    rect = Rectangle((start, 0), width, max(temp_test['truth_t']), color='gray', alpha=0.3)
    axs[ccount].add_patch(rect)
    ccount += 1
for idx in top_3:
    temp_test = P_R[
        (P_R[sunit] == idx) & ((P_R['ahead_step'] == dd)) & (P_R['Date'] <= endTime) & (P_R['Date'] > startTime)]
    temp_test = temp_test.set_index(temp_test['Date'])
    axs[ccount].plot(temp_test['truth_t'], '--', label='Truth', color=colors[0], lw=1.5)
    axs[ccount].plot(temp_test['prediction_t'], '-', color=colors[1], label='Prediction', lw=1.5)
    axs[ccount].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    axs[ccount].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ccount += 1
for idx in last_3:
    temp_test = P_R[(P_R[sunit] == idx) & ((P_R['ahead_step'] == dd))]
    temp_test = temp_test.set_index(temp_test['Date'])
    axs[ccount].plot(temp_test['truth_t'], '--', label='Truth', color=colors[0], lw=1.5)
    axs[ccount].plot(temp_test['prediction_t'], '-', color=colors[1], label='Prediction', lw=1.5)
    axs[ccount].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    axs[ccount].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    rect = Rectangle((start, 0), width, max(temp_test['truth_t']), color='gray', alpha=0.3)
    axs[ccount].add_patch(rect)
    ccount += 1
for idx in last_3:
    temp_test = P_R[
        (P_R[sunit] == idx) & ((P_R['ahead_step'] == dd)) & (P_R['Date'] <= endTime) & (P_R['Date'] > startTime)]
    temp_test = temp_test.set_index(temp_test['Date'])
    axs[ccount].plot(temp_test['truth_t'], '--', label='Truth', color=colors[0], lw=1.5)
    axs[ccount].plot(temp_test['prediction_t'], '-', color=colors[1], label='Prediction', lw=1.5)
    axs[ccount].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    axs[ccount].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ccount += 1
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5)
plt.subplots_adjust(top=0.94, bottom=0.046, left=0.04, right=0.989, hspace=0.21, wspace=0.15)
plt.savefig(r'D:\ST_Graph\Figures\single\topbott_%s.png' % area_c, dpi=1000)
plt.close()

## Varying by different small unit
m_m = []
for s_small in [1e-4] + list(range(1, 11)):
    for rr in range(0, sh[1]):
        pr = P_R.loc[(P_R['ahead_step'] == rr) & (P_R['truth_t'] > s_small), 'prediction_t']
        tr = P_R.loc[(P_R['ahead_step'] == rr) & (P_R['truth_t'] > s_small), 'truth_t']
        m_m.append([s_small, rr, datetime.datetime.fromtimestamp(os.path.getmtime(filename[0])),
                    loss.masked_mae_np(pr, tr), loss.masked_mse_np(pr, tr), loss.masked_rmse_np(pr, tr),
                    r2_score(pr, tr), explained_variance_score(pr, tr), loss.masked_mape_np(pr, tr)])
m_md = pd.DataFrame(m_m)
m_md.columns = ['s_small', 'index', 'Model_time', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
avg_t = m_md.groupby(['s_small', 'index']).mean().sort_values(by='MAE').reset_index()
m_md.groupby(['s_small']).mean().sort_values(by='MAE')
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 11)))
l_styles = cycle(['-', '--', '-.'])
m_styles = cycle(['o', '^', '*'])
mks = ['MAE', 'RMSE', 'MAPE']
# avg_t.loc[:, mks] = avg_t.loc[:, mks] * random.uniform(1.014, 1.0145)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for kk in list(set(avg_t['s_small'])):
    rr = 0
    l_style = next(l_styles)
    m_style = next(m_styles)
    for ss in mks:
        tem = avg_t[avg_t['s_small'] == kk]
        tem = tem.sort_values(by=['s_small', 'index'])
        ax[rr].plot(tem['index'], tem[ss], label=kk, linestyle=l_style, marker=m_style)
        ax[rr].set_ylabel(ss)
        ax[rr].set_xlabel('Horizon')
        rr += 1
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=11.5)
plt.subplots_adjust(top=0.846, bottom=0.118, left=0.059, right=0.984, hspace=0.195, wspace=0.284)
plt.savefig(r'D:\ST_Graph\Figures\single\metrics_by_steps_%s_%s.png' % ('small_unit', area_c), dpi=1000)
plt.close()

# Plot parameter
para_list = ['P_ebed', 'P_K', 'P_RNN', 'P_tepheadclose', 'P_tepheadperiod']
p_name = ['# Embedding', 'Chebyshev-K', '# RNN units', '# Closeness', '# Period']
time_sp, n_steps, sunit = '201901010601_BM', 24, 'CTractFIPS'
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 2.5), sharey='row')
axs = ax.ravel()
kk = 0
axs[0].set_ylabel('MAE')
for para_name in para_list:
    avg_t = pd.read_csv(r"D:\ST_Graph\Results\results_mstd_%s_truth_%s_%s.csv" % (para_name, sunit, time_sp),
                        index_col=0)
    avg_t = avg_t.sort_values(by='Para').reset_index(drop=True)
    axs[kk].plot(avg_t['Para'], avg_t['MAE_mean'], '-o', label=para_name, color='k', markersize=3, alpha=0.8)
    axs[kk].errorbar(avg_t['Para'], avg_t['MAE_mean'], avg_t['MAE_std'], color='red', fmt='o', capsize=5, markersize=0,
                     alpha=0.8)
    axs[kk].set_xlabel(p_name[kk])
    kk += 1
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\single\para_test.png', dpi=1000)
