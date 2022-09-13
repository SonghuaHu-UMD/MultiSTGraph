####################################
# Merge all population inflow and output to lib-city format.
# W/O Group-based standardized
####################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import glob
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import json

pd.options.mode.chained_assignment = None
results_path = r'D:\\ST_Graph\\Data\\'

# Get county subdivision
CTS_Info = pd.read_pickle(r'D:\ST_Graph\Results\CTS_Info.pkl')

# Merge all and output to lib-city format
# Dynamic
t_s = datetime.datetime(2019, 1, 1)
t_e = datetime.datetime(2020, 3, 1)
train_ratio = 0.7
split_time = t_s + datetime.timedelta(days=int(((t_e - t_s).total_seconds() / (24 * 3600)) * train_ratio))
POI_Type = ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']
f_na, f_nas, f_gp, f_gps = 'SG_CTS_Hourly', 'SG_CTS_Hourly_Single', 'SG_CTS_Hourly_GP', 'SG_CTS_Hourly_Single_GP'
f_list = [f_na, f_nas, f_gp, f_gps]
CTS_Hourly = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_PC', 'CTS_Visit_Hourly*.pkl'))))
CTS_Hourly['Time'] = pd.to_datetime(CTS_Hourly['Time'])
CTS_Hourly = CTS_Hourly[(CTS_Hourly['Time'] < t_e) & (CTS_Hourly['Time'] >= t_s)].reset_index(drop=True)
CTS_Hourly = CTS_Hourly.fillna(0)
# CTS_Hourly.groupby(['Time']).sum().plot()
holidays = calendar().holidays(start=CTS_Hourly['Time'].dt.date.min(), end=CTS_Hourly['Time'].dt.date.max())
CTS_Hourly['Holiday'] = CTS_Hourly['Time'].dt.date.astype('datetime64').isin(holidays).astype(int)
CTS_Hourly['dayofweek'] = CTS_Hourly['Time'].dt.dayofweek
CTS_Hourly['Weekend'] = CTS_Hourly['dayofweek'].isin([5, 6]).astype(int)
CTS_Hourly = CTS_Hourly.sort_values(by=['CTSFIPS', 'Time']).reset_index(drop=True).reset_index()
CTS_Hourly['All'] = CTS_Hourly[POI_Type].sum(axis=1)
CTS_Hourly_train = CTS_Hourly[CTS_Hourly['Time'] <= split_time].reset_index(drop=True)
CTS_Hourly['Time'] = CTS_Hourly['Time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
CTS_Hourly['type'] = 'state'
print(CTS_Hourly.isnull().sum())

# Output: without normalize
Dyna = CTS_Hourly[['index', 'type', 'Time', 'CTSFIPS'] + POI_Type]
Dyna.columns = ['dyna_id', 'type', 'time', 'entity_id'] + POI_Type
print(Dyna.isnull().sum())
Dyna.to_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_na, f_na), index=0)
Dyna_s = CTS_Hourly[['index', 'type', 'Time', 'CTSFIPS', 'All']]
Dyna_s.columns = ['dyna_id', 'type', 'time', 'entity_id', 'Visits']
print(Dyna_s.isnull().sum())
Dyna_s[['dyna_id', 'type', 'time', 'entity_id', 'Visits']].to_csv(
    results_path + r'Lib_Data\%s\%s.dyna' % (f_nas, f_nas), index=0)

# Group-normalize
ct_visit_mean = CTS_Hourly_train.groupby(['CTSFIPS'])[POI_Type + ['All']].mean().reset_index()
ct_visit_mean.columns = ['CTSFIPS'] + [var + '_m' for var in list(ct_visit_mean.columns[1:])]
ct_visit_std = CTS_Hourly_train.groupby(['CTSFIPS'])[POI_Type + ['All']].std().reset_index()
ct_visit_std.columns = ['CTSFIPS'] + [var + '_std' for var in list(ct_visit_std.columns[1:])]
ct_visit_mstd = ct_visit_mean.merge(ct_visit_std, on='CTSFIPS')  # some zeros exist
ct_visit_mstd.to_pickle(r'D:\ST_Graph\Results\cts_visit_mstd.pkl')
CTS_Hourly = CTS_Hourly.merge(ct_visit_mstd, on='CTSFIPS')
# for kk in POI_Type + ['All']: CTS_Hourly[kk] = (CTS_Hourly[kk] - CTS_Hourly[kk + '_m']) / CTS_Hourly[kk + '_std']
# For comparison, do not normalize by POI type
for kk in POI_Type: CTS_Hourly[kk] = (CTS_Hourly[kk] - CTS_Hourly[kk + '_m']) / CTS_Hourly['All_std']
CTS_Hourly['All'] = (CTS_Hourly['All'] - CTS_Hourly['All' + '_m']) / CTS_Hourly['All_std']
print((CTS_Hourly['All'] - CTS_Hourly[POI_Type].sum(axis=1)).sum())
CTS_Hourly = CTS_Hourly.fillna(0)
Dyna_gp = CTS_Hourly[['index', 'type', 'Time', 'CTSFIPS'] + POI_Type]
Dyna_gp.columns = ['dyna_id', 'type', 'time', 'entity_id'] + POI_Type
print(Dyna_gp.isnull().sum())
Dyna_gp.to_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_gp, f_gp), index=0)
Dyna_gps = CTS_Hourly[['index', 'type', 'Time', 'CTSFIPS', 'All']]
Dyna_gps.columns = ['dyna_id', 'type', 'time', 'entity_id', 'Visits']
Dyna_gps[['dyna_id', 'type', 'time', 'entity_id', 'Visits']].to_csv(
    results_path + r'Lib_Data\%s\%s.dyna' % (f_gps, f_gps), index=0)

# Geo
CTS_Info['x'] = CTS_Info.centroid.x
CTS_Info['y'] = CTS_Info.centroid.y
CTS_Info['coordinates'] = "[" + CTS_Info['x'].astype(str) + ', ' + CTS_Info['y'].astype(str) + "]"
CTS_Info['type'] = 'Point'
CTS_Info_out = CTS_Info[['CTSFIPS', 'type', 'coordinates']]
CTS_Info_out.columns = ['geo_id', 'type', 'coordinates']
for kk in f_list: CTS_Info_out.to_csv(results_path + r'Lib_Data\%s\%s.geo' % (kk, kk), index=0)

# Rel: build via OD
CTS_OD = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_PC', 'CTS_OD_Weekly_*.pkl'))))
CTS_OD['Volume'] = CTS_OD[POI_Type].sum(axis=1)
CTS_OD = CTS_OD[['CTSFIPS_O', 'CTSFIPS_D', 'Volume']]
CTS_OD = CTS_OD.groupby(['CTSFIPS_O', 'CTSFIPS_D']).sum().reset_index()
CTS_D = CTS_OD.groupby(['CTSFIPS_D'])['Volume'].sum().reset_index()
CTS_D.columns = ['CTSFIPS_D', 'Inflow']
CTS_OD = CTS_OD.merge(CTS_D, on='CTSFIPS_D')
CTS_OD['link_weight'] = CTS_OD['Volume'] / CTS_OD['Inflow']
CTS_OD_W = CTS_OD.pivot(index='CTSFIPS_O', columns='CTSFIPS_D', values='link_weight')
CTS_OD_W = CTS_OD_W.fillna(0).reset_index()
# sns.heatmap(CTS_OD_W.iloc[:, 1:].values)
CTS_OD = pd.melt(CTS_OD_W, id_vars='CTSFIPS_O', value_vars=list(CTS_OD_W.columns)[1:]).reset_index()
CTS_OD['type'] = 'geo'
CTS_OD = CTS_OD[['index', 'type', 'CTSFIPS_O', 'CTSFIPS_D', 'value']]
CTS_OD.columns = ['rel_id', 'type', 'origin_id', 'destination_id', 'link_weight']
for kk in f_list: CTS_OD.to_csv(results_path + r'Lib_Data\%s\%s.rel' % (kk, kk), index=0)

# Ext: add holiday and weekend
CTS_Hourly_ext = CTS_Hourly[['Time', 'Holiday', 'Weekend']]
CTS_Hourly_ext = CTS_Hourly_ext.drop_duplicates().reset_index(drop=True).reset_index()
CTS_Hourly_ext.columns = ['ext_id', 'time', 'holiday', 'weekend']
# Ext: add weather
weather = pd.read_pickle(r'D:\ST_Graph\Results\weather_2019_bmc.pkl')
weather['time'] = weather['DATE'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
wlist = ['wind', 'temp', 'rain', 'snow', 'vis']
weather[wlist] = (weather[wlist] - weather[wlist].mean()) / weather[wlist].std()
CTS_Hourly_ext = CTS_Hourly_ext.merge(weather[wlist + ['time']], on=['time'])
for kk in f_list: CTS_Hourly_ext.to_csv(results_path + r'Lib_Data\%s\%s.ext' % (kk, kk), index=0)

# Configure: multi-POI
config = dict()
config['geo'] = dict()
config['geo']['including_types'] = ['Point']
config['geo']['Point'] = {}
config['rel'] = dict()
config['rel']['including_types'] = ['geo']
config['rel']['geo'] = {'link_weight': 'num'}
config['dyna'] = dict()
config['dyna']['including_types'] = ['state']
config['dyna']['state'] = {'entity_id': 'geo_id', 'Education': 'num', 'Others': 'num', 'Recreation': 'num',
                           'Residential': 'num', 'Restaurant': 'num', 'Retail': 'num', 'Service': 'num'}
config['ext'] = {'ext_id': 'num', 'time': 'other', 'holiday': 'num', 'weekend': 'num'}
config['info'] = dict()
config['info']['data_col'] = ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']
config['info']['weight_col'] = 'link_weight'
config['info']['ext_col'] = ['holiday', 'weekend'] + wlist
config['info']['data_files'] = ['SG_CTS_Hourly']
config['info']['geo_file'] = 'SG_CTS_Hourly'
config['info']['rel_file'] = 'SG_CTS_Hourly'
config['info']['ext_file'] = 'SG_CTS_Hourly'
config['info']['output_dim'] = 7
config['info']['time_intervals'] = 3600
config['info']['init_weight_inf_or_zero'] = 'zero'
config['info']['set_weight_link_or_dist'] = 'dist'
config['info']['calculate_weight_adj'] = False
config['info']['weight_adj_epsilon'] = 0.1
json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_na, 'w', encoding='utf-8'), ensure_ascii=False)
config['info']['data_files'] = ['SG_CTS_Hourly_GP']
config['info']['geo_file'] = 'SG_CTS_Hourly_GP'
config['info']['rel_file'] = 'SG_CTS_Hourly_GP'
config['info']['ext_file'] = 'SG_CTS_Hourly_GP'
json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_gp, 'w', encoding='utf-8'), ensure_ascii=False)

# Configure: Single POI
config['dyna']['state'] = {'entity_id': 'geo_id', 'Visits': 'num'}
config['info']['data_col'] = ['Visits']
config['info']['data_files'] = ['SG_CTS_Hourly_Single']
config['info']['geo_file'] = 'SG_CTS_Hourly_Single'
config['info']['rel_file'] = 'SG_CTS_Hourly_Single'
config['info']['ext_file'] = 'SG_CTS_Hourly_Single'
config['info']['output_dim'] = 1
json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_nas, 'w', encoding='utf-8'), ensure_ascii=False)
config['info']['data_files'] = ['SG_CTS_Hourly_Single_GP']
config['info']['geo_file'] = 'SG_CTS_Hourly_Single_GP'
config['info']['rel_file'] = 'SG_CTS_Hourly_Single_GP'
config['info']['ext_file'] = 'SG_CTS_Hourly_Single_GP'
json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_gps, 'w', encoding='utf-8'), ensure_ascii=False)
