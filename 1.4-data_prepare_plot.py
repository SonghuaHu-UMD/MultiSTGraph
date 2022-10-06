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

pd.options.mode.chained_assignment = None

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

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
if sunit == 'CTSFIPS':
    CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_cty_sub_2019.shp')
elif sunit == 'CTractFIPS':
    CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_tract_2019.shp')
elif sunit == 'CBGFIPS':
    CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_blck_grp_2019.shp')
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
plt.savefig(r'D:\ST_Graph\Figures\single\%s_%s_normal_times_plot.png' % (time_sp, sunit), dpi=1000)
plt.close()
