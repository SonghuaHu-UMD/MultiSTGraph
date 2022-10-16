import pandas as pd
import numpy as np
import glob
import os
from libcity.model import loss
from sklearn.metrics import r2_score, explained_variance_score
import datetime

pd.options.mode.chained_assignment = None
results_path = '.\libcity\cache\\*'

# Give a dir and read all files inside the dir
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
            ct_id = np.tile(ct_visit_mstd[['CTractFIPS']].values, (sh[0], sh[1], 1, sh[3]))
            ahead_step = np.tile(np.expand_dims(np.array(range(0, sh[1])), axis=(1, 2)), (sh[0], 1, sh[2], sh[3]))
            P_R = pd.DataFrame({'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten(),
                                'All_m': ct_ma.flatten(), 'All_std': ct_sa.flatten(), 'CTractFIPS': ct_id.flatten(),
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


data_name = '201901010601_DC'
filenames = glob.glob(results_path)
filenames = [var for var in filenames if 'dataset_cache' not in var]
all_results = get_gp_data(filenames)
if len(all_results) > 0:
    all_results_avg = all_results.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
    all_results_avg.to_csv(r".\results\M_average.csv")

    # Re-transform the data
    ct_visit_mstd = pd.read_pickle(r'.\other_data\%s_%s_visit_mstd.pkl' % ('CTractFIPS', data_name)).sort_values(
        by='CTractFIPS').reset_index(drop=True)
    m_m = transfer_gp_data(filenames, ct_visit_mstd)
    m_md = pd.DataFrame(m_m)
    m_md.columns = ['Model_name', 'index', 'Model_time', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
    all_results_avg_t = m_md.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
    all_results_avg_t.to_csv(r".\results\M_truth.csv")
