import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import matplotlib as mpl
from libcity.model import loss
from sklearn.metrics import r2_score, explained_variance_score
import datetime

pd.options.mode.chained_assignment = None
results_path = r'D:\ST_Graph\results_record\\'

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Read metrics of multiple models
time_sps = ['201901010601_DC', '202001010601_DC', '201901010601_BM', '202001010601_BM']
for time_sp in time_sps:
    # time_sp = '202001010601_DC'
    sunit = 'CTractFIPS'
    filenames = glob.glob(results_path + r"24 steps\Baselines\%s\*" % time_sp)
    filenames = [ec for ec in filenames if 'log' not in ec]
    all_results = pd.DataFrame()
    for ec in filenames:
        nec = glob.glob(ec + '\\evaluate_cache\\*.csv')
        model_name = glob.glob(ec + '\\model_cache\\*.m')
        if len(nec) > 0:
            nec = nec[0]
            fec = pd.read_csv(nec)
            fec['Model_name'] = model_name[0].split('\\')[-1].split('_')[0]
            all_results = all_results.append(fec)
    all_results = all_results.reset_index()
    all_results_avg = all_results.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
    all_results_avg.to_csv(r"D:\ST_Graph\Results\results_avg_gp_%s_%s.csv" % (sunit, time_sp))

    # Re-transform the data
    # Prepare group mean and std
    ct_visit_mstd = pd.read_pickle(r'D:\ST_Graph\Results\%s_%s_visit_mstd.pkl' % (sunit, time_sp))
    ct_visit_mstd = ct_visit_mstd.sort_values(by=sunit).reset_index(drop=True)
    # Read prediction result
    m_m = []
    for kk in filenames:
        print(kk)
        filename = glob.glob(kk + r"\\evaluate_cache\*.npz")
        model_name = glob.glob(kk + '\\model_cache\\*.m')
        if len(filename) > 0:
            Predict_R = np.load(filename[0])
            sh = Predict_R['prediction'].shape
            ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
            ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
            ct_id = np.tile(ct_visit_mstd[[sunit]].values, (sh[0], sh[1], 1, sh[3]))
            ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
            Predict_Real = pd.DataFrame(
                {'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(),
                 'All_m': ct_ma.flatten(), 'All_std': ct_sa.flatten(), sunit: ct_id.flatten(),
                 'ahead_step': ahead_step.flatten()})
            Predict_Real['prediction_t'] = Predict_Real['prediction'] * Predict_Real['All_std'] + Predict_Real['All_m']
            Predict_Real['truth_t'] = Predict_Real['truth'] * Predict_Real['All_std'] + Predict_Real['All_m']
            # Predict_Real.loc[Predict_Real['truth_t'] < 0, 'prediction_t'] = 0
            Predict_Real.loc[Predict_Real['prediction_t'] < 0, 'prediction_t'] = 0
            # Predict_Real.loc[Predict_Real['truth_t'] < 10, 'prediction_t'] = np.nan  # not consider small volume
            for rr in range(0, sh[1]):
                pr = Predict_Real.loc[
                    (Predict_Real['ahead_step'] == rr) & (Predict_Real['truth_t'] > 10), 'prediction_t']
                tr = Predict_Real.loc[(Predict_Real['ahead_step'] == rr) & (Predict_Real['truth_t'] > 10), 'truth_t']
                m_m.append(
                    [model_name[0].split('\\')[-1].split('_')[0], rr, loss.masked_mae_np(pr, tr),
                     loss.masked_mse_np(pr, tr),
                     loss.masked_rmse_np(pr, tr), r2_score(pr, tr), explained_variance_score(pr, tr),
                     loss.masked_mape_np(pr, tr)])
        else:
            print(kk + '----NULL----')

    m_md = pd.DataFrame(m_m)
    m_md.columns = ['Model_name', 'index', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
    all_results_avg_t = m_md.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
    all_results_avg_t.to_csv(r"D:\ST_Graph\Results\results_avg_truth_%s_%s.csv" % (sunit, time_sp))

# Read metrics of multiple graphs
time_sps = ['201901010601_DC', '202001010601_DC', '201901010601_BM', '202001010601_BM']
graph_list = ['od-bidirection', 'od-unidirection', 'od', 'dist', 'cosine', 'identity']
for time_sp in time_sps:
    time_sp = '202001010601_DC'
    sunit = 'CTractFIPS'
    filenames = glob.glob(results_path + r"24 steps\Graphs\%s\*" % time_sp)
    filenames = [ec for ec in filenames if 'log' not in ec]
    all_results = pd.DataFrame()
    for ec in filenames:
        nec = glob.glob(ec + '\\evaluate_cache\\*.csv')
        model_name = glob.glob(ec + '\\model_cache\\*.m')
        if len(nec) > 0:
            nec = nec[0]
            fec = pd.read_csv(nec)
            fec['Model_name'] = model_name[0].split('\\')[-1].split('_')[0]
            fec['Model_time'] = datetime.datetime.fromtimestamp(os.path.getmtime(nec))
            all_results = all_results.append(fec)
    all_results = all_results.reset_index()
    all_results = all_results.sort_values(by=['Model_time', 'index']).reset_index(drop=True)
    all_results['Graph'] = np.repeat(graph_list, 24 * 5)
    all_results_avg = all_results.groupby(['Graph']).mean().sort_values(by='MAE').reset_index()
    all_results_avg.to_csv(r"D:\ST_Graph\Results\results_graphs_gp_%s_%s.csv" % (sunit, time_sp))

    # Re-transform the data
    # Prepare group mean and std
    ct_visit_mstd = pd.read_pickle(r'D:\ST_Graph\Results\%s_%s_visit_mstd.pkl' % (sunit, time_sp))
    ct_visit_mstd = ct_visit_mstd.sort_values(by=sunit).reset_index(drop=True)
    # Read prediction result
    m_m = []
    for kk in filenames:
        print(kk)
        filename = glob.glob(kk + r"\\evaluate_cache\*.npz")
        model_name = glob.glob(kk + '\\model_cache\\*.m')
        if len(filename) > 0:
            Predict_R = np.load(filename[0])
            sh = Predict_R['prediction'].shape
            ct_ma = np.tile(ct_visit_mstd[['All_m']].values, (sh[0], sh[1], 1, sh[3]))
            ct_sa = np.tile(ct_visit_mstd[['All_std']].values, (sh[0], sh[1], 1, sh[3]))
            ct_id = np.tile(ct_visit_mstd[[sunit]].values, (sh[0], sh[1], 1, sh[3]))
            ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
            Predict_Real = pd.DataFrame(
                {'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(),
                 'All_m': ct_ma.flatten(), 'All_std': ct_sa.flatten(), sunit: ct_id.flatten(),
                 'ahead_step': ahead_step.flatten()})
            Predict_Real['prediction_t'] = Predict_Real['prediction'] * Predict_Real['All_std'] + Predict_Real['All_m']
            Predict_Real['truth_t'] = Predict_Real['truth'] * Predict_Real['All_std'] + Predict_Real['All_m']
            # Predict_Real.loc[Predict_Real['truth_t'] < 0, 'prediction_t'] = 0
            Predict_Real.loc[Predict_Real['prediction_t'] < 0, 'prediction_t'] = 0
            # Predict_Real.loc[Predict_Real['truth_t'] < 10, 'prediction_t'] = np.nan  # not consider small volume
            for rr in range(0, sh[1]):
                pr = Predict_Real.loc[
                    (Predict_Real['ahead_step'] == rr) & (Predict_Real['truth_t'] > 10), 'prediction_t']
                tr = Predict_Real.loc[(Predict_Real['ahead_step'] == rr) & (Predict_Real['truth_t'] > 10), 'truth_t']
                m_m.append(
                    [model_name[0].split('\\')[-1].split('_')[0], rr,
                     datetime.datetime.fromtimestamp(os.path.getmtime(filename[0])), loss.masked_mae_np(pr, tr),
                     loss.masked_mse_np(pr, tr), loss.masked_rmse_np(pr, tr), r2_score(pr, tr),
                     explained_variance_score(pr, tr), loss.masked_mape_np(pr, tr)])
        else:
            print(kk + '----NULL----')

    m_md = pd.DataFrame(m_m)
    m_md.columns = ['Model_name', 'index', 'Model_time', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
    m_md = m_md.sort_values(by=['Model_time', 'index']).reset_index(drop=True)
    m_md['Graph'] = np.repeat(graph_list, 24 * 5)
    all_results_avg_t = m_md.groupby(['Graph']).mean().sort_values(by='MAE').reset_index()
    all_results_avg_t.to_csv(r"D:\ST_Graph\Results\results_graphs_truth_%s_%s.csv" % (sunit, time_sp))

# Plot a county
fig, ax = plt.subplots(figsize=(12, 6))
for kk in list(ct_visit_mstd[sunit])[0:1]:
    temp = Predict_Real[(Predict_Real[sunit] == kk) & (Predict_Real['ahead_step'] == 0)]
    ax.plot(temp['prediction_t'], label='prediction')
    ax.plot(temp['truth_t'], label='truth')
plt.legend()
plt.tight_layout()
