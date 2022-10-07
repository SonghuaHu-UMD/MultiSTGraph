import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import glob
import geopandas as gpd
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib as mpl
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import functools as ft

pd.options.mode.chained_assignment = None

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})


def calculate_adjacency_matrix_dist(dist_mx, weight_adj_epsilon=0.0):
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    dist_mx = np.exp(-np.square(dist_mx / std))
    dist_mx[dist_mx < weight_adj_epsilon] = 0
    return dist_mx


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def adj_wide2long(learned_graph, Geo_Info, colname):
    learned_graph.columns = list(Geo_Info['geo_id'])
    learned_graph.set_index(pd.Index(list(Geo_Info['geo_id'])), inplace=True)
    learned_graph = learned_graph.unstack().reset_index()
    learned_graph.columns = ['origin', 'des', colname]
    return learned_graph


# Para setting
results_path = r'D:\\ST_Graph\\Data\\'
geo_path = r'E:\SafeGraph\Open Census Data\Census Website\2019\\'
t_s, t_e = datetime.datetime(2019, 1, 1), datetime.datetime(2019, 6, 1)  # datetime.datetime(2019, 7, 1)
area_c, sunit = '_DC', 'CTractFIPS'
time_sp = t_s.strftime('%Y%m%d') + t_e.strftime('%m%d') + area_c
t_days = (t_e - t_s).days
train_ratio = 0.7
split_time = t_s + datetime.timedelta(days=int(((t_e - t_s).total_seconds() / (24 * 3600)) * train_ratio))
test_time = t_s + datetime.timedelta(days=int(((t_e - t_s).total_seconds() / (24 * 3600)) * (train_ratio + 0.15)))
print(split_time)
f_na, f_nas, f_gp, f_gps = '%s_SG_%s_Hourly' % (time_sp, sunit), '%s_SG_%s_Hourly_Single' % (
    time_sp, sunit), '%s_SG_%s_Hourly_GP' % (time_sp, sunit), '%s_SG_%s_Hourly_Single_GP' % (time_sp, sunit)

# Read geo files
Geo_Info = pd.read_csv(results_path + r'Lib_Data\%s\%s.geo' % (f_nas, f_nas))
Geo_Info['geo_id'] = Geo_Info['geo_id'].astype(str)
CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_tract_2019.shp')
CBG_Info = CBG_Info[CBG_Info['GEOID'].isin(Geo_Info['geo_id'])]
print(len(CBG_Info) == len(Geo_Info))
CBG_Info = CBG_Info.to_crs("EPSG:4326").reset_index(drop=True)

# Read time series
Dyna = pd.read_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_nas, f_nas))
Dyna['time'] = pd.to_datetime(Dyna['time'])
Dyna['GEOID'] = Dyna['entity_id'].astype(str)
Dyna_avg = Dyna.groupby(['GEOID'])['Visits'].mean().reset_index()
poly = CBG_Info.merge(Dyna_avg[['GEOID', 'Visits']], on='GEOID')

# Figure 1: Plot Spatially before group normal
plot_1 = 'Visits'
colormap = 'coolwarm'
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1.5]})
CBG_Info.geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.1, ax=ax[0])
poly.plot(column=plot_1, ax=ax[0], legend=True, scheme='UserDefined', cmap=colormap, linewidth=0, edgecolor='white',
          classification_kwds=dict(bins=[np.quantile(poly[plot_1], 1 / 6), np.quantile(poly[plot_1], 2 / 6),
                                         np.quantile(poly[plot_1], 3 / 6), np.quantile(poly[plot_1], 4 / 6),
                                         np.quantile(poly[plot_1], 5 / 6)]), legend_kwds=dict(frameon=False, ncol=3),
          alpha=0.9)
ax[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax[0].axis('off')
ax[0].set_title('Hourly Population Inflow (Average)', pad=-5)
# Reset Legend
patch_col = ax[0].get_legend()
patch_col.set_bbox_to_anchor((1.05, 0.05))
legend_labels = ax[0].get_legend().get_texts()
for bound, legend_label in \
        zip(['< ' + str(round(np.quantile(poly[plot_1], 1 / 6))),
             str(round(np.quantile(poly[plot_1], 1 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 2 / 6))),
             str(round(np.quantile(poly[plot_1], 2 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 3 / 6))),
             str(round(np.quantile(poly[plot_1], 3 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 4 / 6))),
             str(round(np.quantile(poly[plot_1], 4 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 5 / 6))),
             '> ' + str(round(np.quantile(poly[plot_1], 5 / 6)))], legend_labels):
    legend_label.set_text(bound)
plt.subplots_adjust(top=0.96, bottom=0.137, left=-0.1, right=0.984, hspace=0.2, wspace=-0.1)
# Power low
sns.set_palette("coolwarm")
bins = range(1, int(Dyna[plot_1].max()) + 2, 10)
y_data, x_data = np.histogram(Dyna[plot_1], bins=bins, density=True)
x_data = x_data[:-1]
ax[1].loglog(x_data, y_data, basex=10, basey=10, linestyle='None', marker='o', markersize=5, alpha=0.8,
             fillstyle='none')
ax[1].set_ylabel('Probability Density')
ax[1].set_xlabel('Population Inflow')
plt.savefig(r'D:\ST_Graph\Figures\single\%s_%s_loglogplot.png' % (time_sp, sunit), dpi=1000)
plt.close()

# Figure 2: Daily plot, normalize
Dyna = pd.read_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_gps, f_gps))
Dyna['time'] = pd.to_datetime(Dyna['time'])
Dyna['GEOID'] = Dyna['entity_id'].astype(str)
Dyna['Visits'] = Dyna['Visits'] - Dyna['Visits'].mean() / Dyna['Visits'].std()
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 100)))
fig, ax = plt.subplots(figsize=(10, 6))
for kk in set(Dyna['entity_id']):
    tempfile = Dyna[Dyna['entity_id'] == kk]
    ax.plot(tempfile['time'], tempfile['Visits'], label=kk, alpha=0.4, lw=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_ylabel('Hourly Population Inflow (Normalized)')
ax.set_xlabel('Date')
ax.plot(Dyna.groupby('time')['Visits'].mean(), color='k', alpha=0.6, lw=1.5)
ax.plot([split_time, split_time], [-1, 25], '-.', color='green', alpha=0.6, lw=3)
ax.plot([test_time, test_time], [-1, 25], '-.', color='blue', alpha=0.6, lw=3)
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\single\%s_%s_normal_daily_plot.png' % (time_sp, sunit), dpi=1000)
plt.close()

# Figure 3: weekly plot
Dyna = pd.read_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_gps, f_gps))
Dyna['time'] = pd.to_datetime(Dyna['time'])
Dyna['dayofweek'] = Dyna['time'].dt.dayofweek
Dyna['hour'] = Dyna['time'].dt.hour
Dyna_avg = Dyna.groupby(['entity_id', 'dayofweek', 'hour']).mean()['Visits'].reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
for kk in set(Dyna_avg['entity_id']):
    tempfile = Dyna_avg[Dyna_avg['entity_id'] == kk]
    tempfile = tempfile.sort_values(by=['dayofweek', 'hour']).reset_index(drop=True).reset_index()
    ax.plot(tempfile['index'], tempfile['Visits'], label=kk, alpha=0.4, lw=1)
ax.set_ylabel('Hourly Population Inflow (Normalized)')
ax.set_xlabel('Hour')
Dyna_avg_a = Dyna.groupby(['dayofweek', 'hour']).mean()['Visits'].reset_index().reset_index()
ax.plot(Dyna_avg_a['index'], Dyna_avg_a['Visits'], color='k', alpha=0.6, lw=2)
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\single\%s_%s_normal_weekly_plot.png' % (time_sp, sunit), dpi=1000)
plt.close()

# Figure 4: plot graphs
Geo_Info = pd.read_csv(results_path + r'Lib_Data\%s\%s.geo' % (f_nas, f_nas))
Geo_Info['geo_id'] = Geo_Info['geo_id'].astype(str)

# Learned adj matrix
cache_name = r'C:\Users\huson\PycharmProjects\MultiSTGraph\libcity\cache\4069\model_cache\MultiATGCN_201901010601_DC_SG_CTractFIPS_Hourly_Single_GP.m'
model_state, optimizer_state = torch.load(cache_name)
# learned_graph = torch.matmul(model_state['node_vec1'], model_state['node_vec2'])
learned_graph = pd.DataFrame(
    F.softmax(F.relu(torch.mm(model_state['node_vec1'], model_state['node_vec2']))).cpu().numpy())
# sns.heatmap(learned_graph.cpu(), cmap='coolwarm', square=True)
learned_graph = adj_wide2long(learned_graph, Geo_Info, 'learned_weight')

# Adjacent matrix: Distance
Geo_Info[['x', 'y']] = Geo_Info['coordinates']. \
    str.replace('[', ',').str.replace(']', ',').str.split(r',', expand=True)[[1, 2]].astype(float)
geo_mx = pd.concat([Geo_Info] * len(Geo_Info), ignore_index=True)
geo_mx[['geo_id_1', 'x_1', 'y_1']] = Geo_Info.loc[
    Geo_Info.index.repeat(len(Geo_Info)), ['geo_id', 'x', 'y']].reset_index(drop=True)
geo_mx['dist'] = haversine_array(geo_mx['y'], geo_mx['x'], geo_mx['y_1'], geo_mx['x_1'])
geo_mx = geo_mx.pivot(index='geo_id', columns='geo_id_1', values='dist').values
adj_mx_dis = pd.DataFrame(calculate_adjacency_matrix_dist(geo_mx, 0.0))
adj_mx_dis = adj_wide2long(adj_mx_dis, Geo_Info, 'distance_weight')

# Adjacent matrix: similarity
static = pd.read_csv(results_path + r'Lib_Data\%s\%s.static' % (f_nas, f_nas))
static_euc = cdist(static.values[:, 1:], static.values[:, 1:], metric='euclidean')
static_euc[static_euc == 0] = 1
adj_mx_cos = pd.DataFrame(1 / static_euc)
adj_mx_cos = adj_wide2long(adj_mx_cos, Geo_Info, 'similar_weight')

# Adjacent matrix: OD
od_adj = pd.read_csv(results_path + r'Lib_Data\%s\%s.rel' % (f_nas, f_nas))
adj_mx_od = od_adj[['origin_id', 'destination_id', 'link_weight']]
adj_mx_od.columns = ['origin', 'des', 'od_weight']
adj_mx_od['origin'] = adj_mx_od['origin'].astype(str)
adj_mx_od['des'] = adj_mx_od['des'].astype(str)

# merge all
adj_final = ft.reduce(lambda left, right: pd.merge(left, right, on=['origin', 'des']),
                      [adj_mx_od, adj_mx_cos, adj_mx_dis, learned_graph])
clos = adj_final.select_dtypes(include=[np.number]).columns
adj_final[clos] = (adj_final[clos] - adj_final[clos].min()) / (adj_final[clos].max() - adj_final[clos].min())
print(adj_final[clos].corr())
print(adj_final[clos].describe())

Geo_Infoxy = Geo_Info[['geo_id', 'x', 'y']]
Geo_Infoxy.columns = ['origin', 'O_Lng', 'O_Lat']
adj_final = adj_final.merge(Geo_Infoxy, on='origin')
Geo_Infoxy.columns = ['des', 'D_Lng', 'D_Lat']
adj_final = adj_final.merge(Geo_Infoxy, on='des')

# Plot
for p1 in ['learned_weight', 'similar_weight', 'od_weight', 'distance_weight']:
    fig, ax = plt.subplots(figsize=(9, 7))
    poly.geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.3, ax=ax)
    Cn = adj_final[adj_final[p1] > np.percentile(adj_final[p1], 90)].reset_index(drop=True)
    print(len(Cn))
    for kk in range(0, len(Cn)):
        ax.annotate('', xy=(Cn.loc[kk, 'O_Lng'], Cn.loc[kk, 'O_Lat']),
                    xytext=(Cn.loc[kk, 'D_Lng'], Cn.loc[kk, 'D_Lat']),
                    arrowprops={'arrowstyle': '-', 'lw': (Cn.loc[kk, p1] / max(Cn.loc[:, p1])) * 3,
                                'color': 'royalblue', 'alpha': 0.5, 'connectionstyle': "arc3,rad=0.2"}, va='center')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    ax.set_title('Adjacent matrix (%s)' % p1, pad=-0)
    plt.savefig(r'D:\ST_Graph\Figures\Single\Adjacent_%s.png' % p1, dpi=600)
    plt.close()
