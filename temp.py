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