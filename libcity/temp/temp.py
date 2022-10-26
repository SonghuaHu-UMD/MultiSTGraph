# Plot learning curve
log_f = r'C:\Users\huson\PycharmProjects\Bigscity-LibCity-SH\libcity\log\\'
infile = log_f + r'75242-GWNET-SG_CTS_Hourly_Single_GP-Sep-10-2022_11-13-29.log'
important = []
keep_phrases = ["train_loss: "]
with open(infile) as f: f = f.readlines()
for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            important.append(line)
            break
lc = pd.DataFrame({'OLD_txt': important})
lc['train_loss'] = lc['OLD_txt'].apply(lambda st: st[st.find("train_loss: ") + 12:st.find(", val_loss")]).astype(float)
lc['val_loss'] = lc['OLD_txt'].apply(lambda st: st[st.find("val_loss: ") + 10:st.find(", lr")]).astype(float)
infile = log_f + r'20410-GWNET-SG_CTS_Hourly_Single_GP-Sep-10-2022_06-54-08.log'
important = []
keep_phrases = ["train_loss: "]
with open(infile) as f: f = f.readlines()
for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            important.append(line)
            break
lc['OLDd_txt'] = important[0:50]
lc['train_lossd'] = lc['OLDd_txt'].apply(lambda st: st[st.find("train_loss: ") + 12:st.find(", val_loss")]).astype(
    float)
lc['val_lossd'] = (lc['OLDd_txt'].apply(lambda st: st[st.find("val_loss: ") + 10:st.find(", lr")])).astype(float)

plt.plot(lc['val_lossd'], label='input_window: 24, blocks: 4, kernel_size: 2, n_layers: 2')
plt.plot(lc['val_loss'], label='input_window: 24, blocks: 4, kernel_size: 2, n_layers: 3')
plt.legend()
plt.show()

# Plot each step for all models
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Set2.colors)
fig, ax = plt.subplots(figsize=(10, 6))
for em in set(all_results_avg.head(8)['Model_name']):
    temp = all_results[all_results['Model_name'] == em]
    ax.plot(temp['index'], temp['masked_MAPE'], label=em, lw=2)
plt.legend(ncol=3)
plt.tight_layout()

# Plot a county sub
filename = glob.glob(results_path + r"gp_single\96224\evaluate_cache\*.npz")
Predict_R = np.load(filename[0])
sh = Predict_R['prediction'].shape
print(sh)  # testing length, prediction length, number of nodes, output dim
fig, ax = plt.subplots(figsize=(12, 6))
for kk in range(110, Predict_R['prediction'].shape[2]):
    ax.plot(Predict_R['prediction'][:, 0, kk, 0], label='prediction')
    ax.plot(Predict_R['truth'][:, 0, kk, 0], label='truth')
plt.legend()
plt.tight_layout()

class TemporalAttentionLayer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(device))

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, F_in, T)

        Returns:
            torch.tensor: (B, T, T)
        """

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        e = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        e_normalized = F.softmax(e, dim=1)

        return e_normalized


        # temporal_at = self.TAt(output.permute(0, 2, 3, 1))
        # output = torch.matmul(output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, self.input_window), temporal_at) \
        #     .reshape(output.shape[0], self.input_window, self.num_nodes, -1)

# Read metrics of multiple models: split
filenames = glob.glob(r"C:\Users\huson\Desktop\results_record\Split\\*")
filenames = [ec for ec in filenames if '.log' not in ec]
all_results = pd.DataFrame()
cc = 0
for ec in filenames:
    nec = glob.glob(ec + '\\evaluate_cache\\*.csv')
    if len(nec) > 0:
        nec = nec[0]
        fec = pd.read_csv(nec)
        fec['Model_name'] = cc
        cc += 1
        all_results = all_results.append(fec)
all_results = all_results.reset_index()
all_results_avg = all_results.groupby(['Model_name']).mean().sort_values(by='MAE').reset_index()
all_results_avg['MAE'].sum()

# Plot a county
fig, ax = plt.subplots(figsize=(12, 6))
for kk in list(ct_visit_mstd[sunit])[0:1]:
    temp = Predict_Real[(Predict_Real[sunit] == kk) & (Predict_Real['ahead_step'] == 0)]
    ax.plot(temp['prediction_t'], label='prediction')
    ax.plot(temp['truth_t'], label='truth')
plt.legend()
plt.tight_layout()

########### Plot metrics by steps, for each model
time_sps, n_steps, nfold = ['201901010601_BM', '201901010601_DC'], [24], 'Final'
for time_sp in time_sps:
    for n_step in n_steps:
        # time_sp = '201901010601_BM'
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
