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
results_path = r'D:\\ST_Graph\\Data\\'

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

# Get county subdivision
f_na, f_nas = 'SG_CTS_Hourly', 'SG_CTS_Hourly_Single'
POI_Type = ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']

# Dynamic
Dyna = pd.read_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_na, f_na))
Dyna['time'] = pd.to_datetime(Dyna['time'])
Dyna['date'] = Dyna['time'].dt.date
Dyna['hour'] = Dyna['time'].dt.hour
Dyna['dayofweek'] = Dyna['time'].dt.dayofweek

# Plot time series
# mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 7)))
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Set2.colors)
fig, ax = plt.subplots(figsize=(10, 6))
Dyna.groupby(['date'])[POI_Type].sum().plot(ax=ax, lw=1.5)
ax.set_xlabel('Date')
ax.set_ylabel('Population Inflow')
plt.legend(ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\Daily_pattern.png', dpi=1000)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
Dyna.groupby(['dayofweek', 'hour'])[POI_Type].sum().plot(ax=ax, lw=1.5)
ax.set_xlabel('Dayofweek, Hour')
ax.set_ylabel('Population Inflow')
plt.legend(ncol=3)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\Hourly_pattern.png', dpi=1000)
plt.close()

# Plot time series for each CTS
Dyna['All'] = Dyna[POI_Type].sum(axis=1)
Dyna = Dyna.rename({'entity_id': 'CTSFIPS'}, axis=1)
Dyna['CTSFIPS'] = Dyna['CTSFIPS'].astype(str)
tempfile = Dyna.groupby(['CTSFIPS', 'date'])['All'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
for kk in set(tempfile['CTSFIPS']):
    ax.plot(tempfile.loc[tempfile['CTSFIPS'] == kk, 'date'], tempfile.loc[tempfile['CTSFIPS'] == kk, 'All'])
ax.set_xlabel('Date')
ax.set_ylabel('Population Inflow')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\Daily_pattern_CTS.png', dpi=1000)
plt.close()

# Geo: PA and OD
# Get PA for each CTS
CTS_Info = pd.read_pickle(r'D:\ST_Graph\Results\CTS_Info.pkl')
CTS_Info['x'] = CTS_Info.centroid.x
CTS_Info['y'] = CTS_Info.centroid.y
data_raw_avg = Dyna.groupby(['CTSFIPS'])[['All'] + POI_Type].mean().reset_index()
poly = CTS_Info.merge(data_raw_avg, on='CTSFIPS')

# Get OD
CTS_OD = pd.read_csv(results_path + r'Lib_Data\%s\%s.rel' % (f_na, f_na))
CTS_OD['origin_id'] = CTS_OD['origin_id'].astype(str)
CTS_OD['destination_id'] = CTS_OD['destination_id'].astype(str)
bmc_zone = CTS_Info[['CTSFIPS', 'x', 'y']]
bmc_zone.columns = ['origin_id', 'O_Lng', 'O_Lat']
CTS_OD = CTS_OD.merge(bmc_zone, on='origin_id')
bmc_zone.columns = ['destination_id', 'D_Lng', 'D_Lat']
CTS_OD = CTS_OD.merge(bmc_zone, on='destination_id')

# Calculate for other POI types
CTS_OD_All = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_PC', 'CTS_OD_Weekly_*.pkl'))))
CTS_OD_All.rename({'CTSFIPS_O': 'origin_id', 'CTSFIPS_D': 'destination_id'}, axis=1, inplace=True)
t_s = datetime.datetime(2019, 1, 1)
t_e = datetime.datetime(2020, 3, 1)
CTS_OD_All['Time'] = pd.to_datetime(CTS_OD_All['Time'])
CTS_OD_All = CTS_OD_All[(CTS_OD_All['Time'] < t_e) & (CTS_OD_All['Time'] >= t_s)].reset_index(drop=True)
CTS_OD_All['All'] = CTS_OD_All[POI_Type].sum(axis=1)

for kk in POI_Type + ['All']:
    CTS_OD_s = CTS_OD_All[['origin_id', 'destination_id'] + [kk]]
    CTS_OD_s = CTS_OD_s.groupby(['origin_id', 'destination_id']).sum().reset_index()
    CTS_D = CTS_OD_s.groupby(['destination_id'])[kk].sum().reset_index()
    CTS_D.columns = ['destination_id', 'Inflow']
    CTS_OD_s = CTS_OD_s.merge(CTS_D, on='destination_id')
    CTS_OD_s[kk] = CTS_OD_s[kk] / CTS_OD_s['Inflow']
    CTS_OD_s.drop(['Inflow'], axis=1, inplace=True)
    CTS_OD = CTS_OD.merge(CTS_OD_s, on=['origin_id', 'destination_id'], how='left')
CTS_OD = CTS_OD[CTS_OD['origin_id'] != CTS_OD['destination_id']].reset_index(drop=True)
CTS_OD = CTS_OD.fillna(0)

for p1 in POI_Type + ['All']:
    fig, ax = plt.subplots(figsize=(9, 7))
    poly.geometry.boundary.plot(color=None, edgecolor='k', linewidth=1, ax=ax)
    # Plot PA
    poly.plot(column=p1, ax=ax, legend=True, scheme='UserDefined', cmap='coolwarm', linewidth=0, edgecolor='white',
              classification_kwds=dict(bins=[np.quantile(poly[p1], 1 / 6), np.quantile(poly[p1], 2 / 6),
                                             np.quantile(poly[p1], 3 / 6), np.quantile(poly[p1], 4 / 6),
                                             np.quantile(poly[p1], 5 / 6)]),
              legend_kwds=dict(frameon=False, ncol=3))
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    ax.set_title('Average Hourly Population Inflow (%s)' % p1, pad=-0)

    # Reset Legend
    patch_col = ax.get_legend()
    patch_col.set_bbox_to_anchor((1, 0.05))
    legend_labels = ax.get_legend().get_texts()
    for bound, legend_label in \
            zip(['< ' + str(round(np.quantile(poly[p1], 1 / 6), 2)),
                 str(round(np.quantile(poly[p1], 1 / 6), 2)) + ' - ' + str(round(np.quantile(poly[p1], 2 / 6), 2)),
                 str(round(np.quantile(poly[p1], 2 / 6), 2)) + ' - ' + str(round(np.quantile(poly[p1], 3 / 6), 2)),
                 str(round(np.quantile(poly[p1], 3 / 6), 2)) + ' - ' + str(round(np.quantile(poly[p1], 4 / 6), 2)),
                 str(round(np.quantile(poly[p1], 4 / 6), 2)) + ' - ' + str(round(np.quantile(poly[p1], 5 / 6), 2)),
                 '> ' + str(round(np.quantile(poly[p1], 5 / 6), 2))], legend_labels):
        legend_label.set_text(bound)
    plt.subplots_adjust(top=0.938, bottom=0.137, left=0.016, right=0.984, hspace=0.2, wspace=0.11)

    # Plot OD
    Cn = CTS_OD[CTS_OD[p1] > 0.001].reset_index(drop=True)
    for kk in range(0, len(Cn)):
        ax.annotate('', xy=(Cn.loc[kk, 'O_Lng'], Cn.loc[kk, 'O_Lat']),
                    xytext=(Cn.loc[kk, 'D_Lng'], Cn.loc[kk, 'D_Lat']),
                    arrowprops={'arrowstyle': '-', 'lw': Cn.loc[kk, p1] * 10, 'color': 'k', 'alpha': 0.8,
                                'connectionstyle': "arc3,rad=0.2"}, va='center')

    plt.savefig(r'D:\ST_Graph\Figures\PA_OD_%s.png' % p1, dpi=600)
    plt.close()
