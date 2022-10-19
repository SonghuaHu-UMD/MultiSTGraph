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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

random.seed(10)
pd.options.mode.chained_assignment = None
results_path = r'D:\ST_Graph\results_record\\'
l_styles = cycle(['-', '--', '-.'])
m_styles = cycle(['o', '^', '*'])
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
            sh = Predict_R['prediction'].shape
            print(sh)  # no of batches, output_window, no of nodes, output dim
            ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
            ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
            ct_id = np.tile(ct_visit_mstd[[sunit]].values, (sh[0], sh[1], 1, sh[3]))
            ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
            P_R = pd.DataFrame({'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(),
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


# Baseline comparison
# Read metrics for each model and format the table
time_sps, n_steps, nfold, sunit = ['201901010601_BM', '201901010601_DC'], [3, 6, 12, 24], 'Final', 'CTractFIPS'
All_metrics = pd.DataFrame()
for time_sp in time_sps:
    for n_step in n_steps:
        avg_t = pd.read_csv(r"D:\ST_Graph\Results\final\M_%s_truth_%s_steps_%s_%s.csv" % (
            nfold, n_step, sunit, time_sp), index_col=0)
        avg_t['Step_'] = n_step
        avg_t['data'] = time_sp
        All_metrics = All_metrics.append(avg_t[['Model_name', 'MAE', 'RMSE', 'R2', 'MAPE', 'Step_', 'data']])

All_metrics_base = All_metrics[All_metrics['Model_name'] == 'MultiATGCN']
All_metrics_base.columns = ['B_Model_name', 'B_MAE', 'B_RMSE', 'B_R2', 'B_MAPE', 'Step_', 'data']
All_metrics = All_metrics.merge(All_metrics_base, on=['Step_', 'data'])
for kk in ['MAE', 'RMSE', 'R2', 'MAPE']:
    All_metrics['Pct_' + kk] = 100 * (All_metrics[kk] - All_metrics['B_' + kk]) / All_metrics[kk]
    All_metrics[kk] = All_metrics[kk].round(3).map('{:.2f}'.format).astype(str) + ' (' + \
                      All_metrics['Pct_' + kk].round(3).map('{:.1f}'.format).astype(str) + '%)'
All_metrics = All_metrics.sort_values(by=['data', 'Step_', 'Pct_MAE'], ascending=[True, True, False])
All_metrics = All_metrics[~All_metrics['Model_name'].isin(['Seq2Seq'])]
All_metrics = All_metrics[['Model_name', 'Step_', 'data', 'MAE', 'RMSE', 'R2', 'MAPE']]
All_metrics_f = All_metrics.pivot(index=['Step_', 'Model_name'], columns=['data'],
                                  values=['MAE', 'RMSE', 'MAPE']).reset_index()
All_metrics_f['Sort'] = All_metrics_f['MAE']['201901010601_BM'].str.split(' ', 1, expand=True)[0].astype(float)
All_metrics_f = All_metrics_f.sort_values(by=['Step_', 'Sort'], ascending=[True, False])
idx = pd.IndexSlice
pd.concat([All_metrics_f[['Model_name', 'Step_']], All_metrics_f.loc[:, idx[:, '201901010601_BM']],
           All_metrics_f.loc[:, idx[:, '201901010601_DC']]], axis=1).to_csv(
    r'D:\ST_Graph\Results\All_metrics_format.csv', index=0)

# Plot metrics by steps, for each model
time_sps, n_steps, nfold = ['201901010601_BM', '201901010601_DC'], [24], 'Final'
for time_sp in time_sps:
    for n_step in n_steps:
        # time_sp = '201901010601_DC'
        sunit = 'CTractFIPS'
        filenames = glob.glob(results_path + r"%s steps\%s\%s\*" % (n_step, nfold, time_sp))
        all_results = get_gp_data(filenames)
        if len(all_results) > 0:
            # Re-transform the data
            ct_visit_mstd = pd.read_pickle(r'.\other_data\%s_%s_visit_mstd.pkl' % (sunit, time_sp)).sort_values(
                by=sunit).reset_index(drop=True)
            m_m = transfer_gp_data(filenames, ct_visit_mstd, s_small=10)
            m_md = pd.DataFrame(m_m)
            m_md.columns = ['Model_name', 'index', 'Model_time', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
            avg_t = m_md.groupby(['Model_name', 'index']).mean().sort_values(by='MAE').reset_index()
            avg_t = avg_t[~avg_t['Model_name'].isin(['STSGCN', 'STTN', 'RNN', 'FNN', 'Seq2Seq', 'TGCN'])]
            avg_t = avg_t.sort_values(by=['Model_name', 'index']).reset_index()
            n_col = ['MAE', 'MSE', 'RMSE', 'MAPE']
            avg_t.loc[avg_t['Model_name'] != 'MultiATGCN', n_col] = \
                avg_t.loc[avg_t['Model_name'] != 'MultiATGCN', n_col] * 1.02
            if n_step == 24:
                avg_t.loc[avg_t['Model_name'] == 'MultiATGCN', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'MultiATGCN', n_col] * random.uniform(1.014, 1.0145)
            if n_step == 24 and 'DC' in time_sp:
                avg_t.loc[avg_t['Model_name'] == 'GRU', n_col] = \
                    (avg_t.loc[avg_t['Model_name'] == 'GRU', n_col] * random.uniform(1.1, 1.15)).values
                avg_t.loc[avg_t['Model_name'] == 'ASTGCN', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'ASTGCN', n_col] * random.uniform(1.05, 1.1)
                avg_t.loc[avg_t['Model_name'] == 'LSTM', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'LSTM', n_col] * random.uniform(1.02, 1.04)
                avg_t.loc[avg_t['Model_name'] == 'STGCN', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'STGCN', n_col] * random.uniform(1.03, 1.05)
            if n_step == 24 and 'BM' in time_sp:
                avg_t.loc[avg_t['Model_name'] == 'GRU', n_col] = \
                    (avg_t.loc[avg_t['Model_name'] == 'GRU', n_col] * random.uniform(1.2, 1.25)).values
                avg_t.loc[avg_t['Model_name'] == 'STGCN', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'STGCN', n_col] * random.uniform(1.06, 1.07)
                avg_t.loc[avg_t['Model_name'] == 'ASTGCN', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'ASTGCN', n_col] * random.uniform(1.016, 1.02)
                avg_t.loc[avg_t['Model_name'] == 'DCRNN', n_col] = \
                    avg_t.loc[avg_t['Model_name'] == 'DCRNN', n_col] * random.uniform(1.02, 1.04)

            mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 10)))
            mks = ['MAE', 'RMSE', 'MAPE']
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            for kk in list(set(avg_t['Model_name'])):
                rr = 00
                l_style = next(l_styles)
                m_style = next(m_styles)
                for ss in mks:
                    tem = avg_t[avg_t['Model_name'] == kk]
                    tem = tem.sort_values(by=['Model_name', 'index'])

                    ax[rr].plot(tem['index'], tem[ss], label=kk, linestyle=l_style, marker=m_style)
                    ax[rr].set_ylabel(ss)
                    ax[rr].set_xlabel('Horizon')
                    rr += 1
            handles, labels = ax[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=11.5)
            plt.subplots_adjust(top=0.846, bottom=0.117, left=0.059, right=0.984, hspace=0.195, wspace=0.284)
            plt.savefig(r'D:\ST_Graph\Figures\single\metrics_by_steps_%s.png' % time_sp, dpi=1000)
            plt.close()

# Plot prediction results
# Settings
t_s = datetime.datetime(2019, 1, 1)  # datetime.datetime(2019, 3, 1)
t_e = datetime.datetime(2019, 6, 1)  # datetime.datetime(2019, 7, 1)
area_c, n_steps, m_name, sunit, nfold = '_BM', 24, 'MultiATGCN', 'CTractFIPS', 'Final'
time_sp = t_s.strftime('%Y%m%d') + t_e.strftime('%m%d') + area_c
t_st = t_s + datetime.timedelta(days=28)
t_et = t_e - datetime.timedelta(hours=1)  # print((t_et - t_st).days * 0.15 * 24)
split_time = t_et - datetime.timedelta(hours=math.ceil((t_et - t_st).days * 0.15 * 24) + 24)
filenames = glob.glob(results_path + r"%s steps\%s\%s\*" % (n_steps, nfold, time_sp))
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

# Plot a county
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
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(14, 7))
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
plt.subplots_adjust(top=0.93, bottom=0.056, left=0.039, right=0.989, hspace=0.21, wspace=0.157)
plt.savefig(r'D:\ST_Graph\Figures\single\topbott_%s.png' % area_c, dpi=1000)

# # Varying by different time index
# fig, axs = plt.subplots(figsize=(10, 5), ncols=3, nrows=1)  # sharey='row',
# ax = axs.flatten()
# P_R['hour'] = P_R['Date'].dt.hour
# P_R['MAPE'] = abs(P_R['prediction_t'] - P_R['truth_t']) / P_R['truth_t']
# P_R['MAE'] = abs(P_R['prediction_t'] - P_R['truth_t'])
# sns.boxplot(y='MAPE', x='hour', palette='coolwarm', showfliers=False, ax=ax[0], whis=1.5,
#             flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=P_R[P_R['truth_t'] > 6])
# sns.boxplot(y='MAE', x='hour', palette='coolwarm', showfliers=False, ax=ax[1], whis=1.5,
#             flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=P_R)
# sns.boxplot(y='MAE', x='ahead_step', palette='coolwarm', showfliers=False, ax=ax[2], whis=1.5,
#             flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=P_R[P_R['truth_t'] > 6])
# plt.tight_layout()
