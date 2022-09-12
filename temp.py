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