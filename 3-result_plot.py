import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib as mpl
from libcity.model import loss
from sklearn.metrics import r2_score, explained_variance_score

pd.options.mode.chained_assignment = None
results_path = r'D:\ST_Graph\Results\\'

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Read metrics of multiple models
filenames = glob.glob(results_path + r"gp_single_weather\*")
filenames = [ec for ec in filenames if 'dataset_cache' not in ec]
all_results = pd.DataFrame()
for ec in filenames:
    nec = glob.glob(ec + '\\evaluate_cache\\*.csv')
    if len(nec) > 0:
        nec = nec[0]
        fec = pd.read_csv(nec)
        fec['Model_name'] = nec.split('_')[-6]
        all_results = all_results.append(fec)
all_results = all_results.reset_index()
all_results_avg = all_results.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
all_results_avg.to_csv(r"D:\ST_Graph\Results\results_avg_single_hourly_gp_weather.csv")

# Re-transform the data
# Prepare group mean and std
ct_visit_mstd = pd.read_pickle(r'D:\ST_Graph\Results\cts_visit_mstd.pkl')
ct_visit_mstd = ct_visit_mstd.sort_values(by='CTSFIPS').reset_index(drop=True)

# Read prediction result
filenames = glob.glob(results_path + r"gp_single\*")
filenames = [ec for ec in filenames if 'dataset_cache' not in ec]
m_m = []
for kk in filenames:
    print(kk)
    filename = glob.glob(kk + r"\\evaluate_cache\*.npz")
    Predict_R = np.load(filename[0])
    sh = Predict_R['prediction'].shape
    ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
    ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
    ct_id = np.tile(ct_visit_mstd[['CTSFIPS']].values, (sh[0], sh[1], 1, sh[3]))
    ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
    Predict_Real = pd.DataFrame(
        {'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(),
         'All_m': ct_ma.flatten(), 'All_std': ct_sa.flatten(), 'CTSFIPS': ct_id.flatten(),
         'ahead_step': ahead_step.flatten()})
    Predict_Real['prediction_t'] = Predict_Real['prediction'] * Predict_Real['All_std'] + Predict_Real['All_m']
    Predict_Real['truth_t'] = Predict_Real['truth'] * Predict_Real['All_std'] + Predict_Real['All_m']
    Predict_Real.loc[Predict_Real['truth_t'] < 10, 'prediction_t'] = np.nan  # not consider small volume
    for rr in range(0, sh[1]):
        pr = Predict_Real.loc[(Predict_Real['ahead_step'] == rr) & (Predict_Real['truth_t'] > 10), 'prediction_t']
        tr = Predict_Real.loc[(Predict_Real['ahead_step'] == rr) & (Predict_Real['truth_t'] > 10), 'truth_t']
        m_m.append([filename[0].split('_')[9], rr, loss.masked_mae_np(pr, tr), loss.masked_mse_np(pr, tr),
                    loss.masked_rmse_np(pr, tr), r2_score(pr, tr), explained_variance_score(pr, tr),
                    loss.masked_mape_np(pr, tr)])

m_md = pd.DataFrame(m_m)
m_md.columns = ['Model_name', 'index', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
all_results_avg_t = m_md.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
all_results_avg_t.to_csv(r"D:\ST_Graph\Results\results_avg_single_hourly_gp_truth.csv")

# Plot a county
fig, ax = plt.subplots(figsize=(12, 6))
for kk in list(ct_visit_mstd['CTSFIPS'])[0:1]:
    temp = Predict_Real[(Predict_Real['CTSFIPS'] == kk) & (Predict_Real['ahead_step'] == 1)]
    ax.plot(temp['prediction_t'], label='prediction')
    ax.plot(temp['truth_t'], label='truth')
plt.legend()
plt.tight_layout()
