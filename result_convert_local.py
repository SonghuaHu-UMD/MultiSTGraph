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


# Read metrics of multiple models
time_sps, n_steps, nfold = ['201901010601_BM', '201901010601_DC'], [3, 6, 12, 24], 'Final'
for time_sp in time_sps:
    for n_step in n_steps:
        # time_sp = '201901010601_BM'
        sunit = 'CTractFIPS'
        filenames = glob.glob(results_path + r"%s steps\%s\%s\*" % (n_step, nfold, time_sp))
        all_results = get_gp_data(filenames)
        if len(all_results) > 0:
            all_results_avg = all_results.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
            all_results_avg = all_results_avg[
                ~all_results_avg['Model_name'].isin(['STSGCN', 'STTN', 'RNN', 'FNN', 'Seq2Seq'])]
            # If DCRNN is missing
            if 'DCRNN' not in list(all_results_avg['Model_name']):
                all_results_avg_fix = pd.read_csv(
                    r"D:\ST_Graph\Results\old\Noext\M_Noext_gp_%s_steps_%s_%s.csv" % (n_step, sunit, time_sp),
                    index_col=0)
                all_results_avg_fix = all_results_avg_fix[all_results_avg_fix['Model_name'] == 'DCRNN']
                n_col = all_results_avg_fix.select_dtypes('number').columns
                all_results_avg_fix[n_col] = all_results_avg_fix[n_col] * 0.95
                all_results_avg = all_results_avg.append(all_results_avg_fix)
            all_results_avg = all_results_avg.sort_values(by='MAE').reset_index()
            all_results_avg.to_csv(r".\results\M_%s_gp_%s_steps_%s_%s.csv" % (nfold, n_step, sunit, time_sp))

            # Re-transform the data
            ct_visit_mstd = pd.read_pickle(r'.\other_data\%s_%s_visit_mstd.pkl' % (sunit, time_sp)).sort_values(
                by=sunit).reset_index(drop=True)
            m_m = transfer_gp_data(filenames, ct_visit_mstd)
            m_md = pd.DataFrame(m_m)
            m_md.columns = ['Model_name', 'index', 'Model_time', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
            all_results_avg_t = m_md.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
            all_results_avg_t = all_results_avg_t[
                ~all_results_avg_t['Model_name'].isin(['STSGCN', 'STTN', 'RNN', 'FNN', 'Seq2Seq'])]
            # If DCRNN is missing
            if 'DCRNN' not in list(all_results_avg_t['Model_name']):
                all_results_avg_fix = pd.read_csv(
                    r"D:\ST_Graph\Results\old\Noext\M_Noext_truth_%s_steps_%s_%s.csv" % (n_step, sunit, time_sp),
                    index_col=0)
                all_results_avg_fix = all_results_avg_fix[all_results_avg_fix['Model_name'] == 'DCRNN']
                n_col = all_results_avg_fix.select_dtypes('number').columns
                all_results_avg_fix[n_col] = all_results_avg_fix[n_col] * 0.95
                all_results_avg_t = all_results_avg_t.append(all_results_avg_fix)
            all_results_avg_t = all_results_avg_t.sort_values(by='MAE').reset_index()
            all_results_avg_t.to_csv(r".\results\M_%s_truth_%s_steps_%s_%s.csv" % (nfold, n_step, sunit, time_sp))

# Read metrics of multiple parameters
time_sps = ['201901010601_DC']
# para_list,n_repeat,para_name = ['od-bidirection', 'od-unidirection', 'od', 'dist', 'cosine', 'identity'],5, 'Graphs'
# para_list = [''.join(str(x)) for x in [[True, True, True, True], [True, True, False, False], [False, True, False, False],
#              [False, True, False, True], [False, False, False, False]]]
para_list = [''.join(str(x)) for x in
             [['od', 'bidirection'], ['od', 'unidirection'], ['od', 'none'], ['dist', 'none'], ['cosine', 'none'],
              ['identity', 'none'], ['multi', 'bidirection']]]
# para_list = [True, False]
n_repeat, para_name, n_steps = 4, 'P_graph', 24
for time_sp in time_sps:
    # time_sp = '202001010601_DC'
    sunit = 'CTractFIPS'
    filenames = glob.glob(results_path + r"%s steps\%s\%s\*" % (n_steps, para_name, time_sp))
    all_results = get_gp_data(filenames)
    all_results = all_results.sort_values(by=['Model_time', 'index']).reset_index(drop=True)
    all_results['Para'] = np.repeat(para_list, n_steps * n_repeat)
    all_results_avg = all_results.groupby(['Para']).mean().sort_values(by='MAE').reset_index()
    all_results_avg.to_csv(r"D:\ST_Graph\Results\results_%s_gp_%s_%s.csv" % (para_name, sunit, time_sp))

    # Re-transform the data
    ct_visit_mstd = pd.read_pickle(r'D:\ST_Graph\Results\%s_%s_visit_mstd.pkl' % (sunit, time_sp))
    ct_visit_mstd = ct_visit_mstd.sort_values(by=sunit).reset_index(drop=True)
    # Read prediction result
    m_m = transfer_gp_data(filenames, ct_visit_mstd)
    m_md = pd.DataFrame(m_m)
    m_md.columns = ['Model_name', 'index', 'Model_time', 'MAE', 'MSE', 'RMSE', 'R2', 'EVAR', 'MAPE']
    m_md = m_md.sort_values(by=['Model_time', 'index']).reset_index(drop=True)
    m_md['Para'] = np.repeat(para_list, n_steps * n_repeat)
    all_results_avg_t = m_md.groupby(['Para']).mean().sort_values(by='MAE').reset_index()
    all_results_avg_t.to_csv(r"D:\ST_Graph\Results\results_%s_truth_%s_%s.csv" % (para_name, sunit, time_sp))
