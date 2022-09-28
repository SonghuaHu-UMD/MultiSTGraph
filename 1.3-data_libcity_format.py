####################################
# Convert all dataset to lib-city format: https://github.com/LibCity/Bigscity-LibCity
# W/O Group-based standardized
# W/O POI types
# Different spatial units: county subdivision, census track, census block group
# Different spatial coverage: BM/DC
# Different temporal coverage: 20190101-0601, 20200101-0601
####################################
import pandas as pd
import numpy as np
import datetime
import glob
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import json
import os
import geopandas as gpd

pd.options.mode.chained_assignment = None
results_path = r'D:\\ST_Graph\\Data\\'
geo_path = r'E:\SafeGraph\Open Census Data\Census Website\2019\\'
t_s = datetime.datetime(2020, 1, 1)  # datetime.datetime(2019, 3, 1)
t_e = datetime.datetime(2020, 6, 1)  # datetime.datetime(2019, 7, 1)
area_c = '_BM'
time_sp = t_s.strftime('%Y%m%d') + t_e.strftime('%m%d') + area_c
t_days = (t_e - t_s).days
train_ratio = 0.7
split_time = t_s + datetime.timedelta(days=int(((t_e - t_s).total_seconds() / (24 * 3600)) * train_ratio))
print(split_time)
POI_Type = ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']
# CTS_Info = pd.read_pickle(r'D:\ST_Graph\Results\CTS_Info.pkl')
# CTS_Info = CTS_Info[CTS_Info['CTFIPS'].isin(['24510', '24005', '24027', '24003'])]
# CTS_Info.plot()
if 'BM' in time_sp:
    ct_list = ['24510', '24005']
    n_cts = ['24005400500', '24005402202', '24005408303', '24005411409', '24510170300', '24510260604', '24510270701']
elif 'DC' in time_sp:
    ct_list = ['11001', '51013']
    n_cts = ['11001009802']
for sunit in ['CTractFIPS']:  # CTSFIPS, CBGFIPS
    f_na, f_nas, f_gp, f_gps = '%s_SG_%s_Hourly' % (time_sp, sunit), '%s_SG_%s_Hourly_Single' % (
        time_sp, sunit), '%s_SG_%s_Hourly_GP' % (time_sp, sunit), '%s_SG_%s_Hourly_Single_GP' % (time_sp, sunit)
    f_list = [f_na, f_nas, f_gp, f_gps]
    d_list = [results_path + 'Lib_Data\\' + var for var in f_list]
    for directory in d_list:
        if not os.path.exists(directory): os.makedirs(directory)

    # Merge all and output to lib-city format
    CTS_Hourly = pd.concat(
        map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_Sunit', '%s_Visit_Hourly*.pkl' % sunit))))
    CTS_Hourly['Time'] = pd.to_datetime(CTS_Hourly['Time'])
    CTS_Hourly = CTS_Hourly[(CTS_Hourly['Time'] < t_e) & (CTS_Hourly['Time'] >= t_s)].reset_index(drop=True)
    CTS_Hourly = CTS_Hourly.fillna(0)
    CTS_Hourly['CTFIPS'] = CTS_Hourly[sunit].str[0:5]
    CTS_Hourly = CTS_Hourly[CTS_Hourly['CTFIPS'].isin(ct_list)].reset_index(drop=True)
    # CTS_Hourly[1:].groupby('Time').sum().plot()
    CTS_Hourly['All'] = CTS_Hourly[POI_Type].sum(axis=1)

    # # Drop those without fully days
    # n_days = CTS_Hourly.groupby([sunit])['Time'].count().reset_index()
    # n_cts = list(n_days.loc[n_days['Time'] != n_days['Time'].median(), sunit])
    # # Drop those zones with small population inflow
    # n_flow = CTS_Hourly.groupby([sunit])['All'].sum().reset_index()
    # n_cts = n_cts + list(n_flow.loc[n_flow['All'] <= t_days * 10, sunit])
    # print(n_cts)
    CTS_Hourly = CTS_Hourly[~CTS_Hourly[sunit].isin(n_cts)].reset_index(drop=True)
    print("No of removed unit: %s" % len(n_cts))

    # add holiday/weekend
    holidays = calendar().holidays(start=CTS_Hourly['Time'].dt.date.min(), end=CTS_Hourly['Time'].dt.date.max())
    CTS_Hourly['Holiday'] = CTS_Hourly['Time'].dt.date.astype('datetime64').isin(holidays).astype(int)
    CTS_Hourly['dayofweek'] = CTS_Hourly['Time'].dt.dayofweek
    CTS_Hourly['Weekend'] = CTS_Hourly['dayofweek'].isin([5, 6]).astype(int)
    CTS_Hourly = CTS_Hourly.sort_values(by=[sunit, 'Time']).reset_index(drop=True).reset_index()
    CTS_Hourly_train = CTS_Hourly[CTS_Hourly['Time'] <= split_time].reset_index(drop=True)
    CTS_Hourly['Time'] = CTS_Hourly['Time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    CTS_Hourly['type'] = 'state'
    print('Len of %s : %s' % (sunit, len(set(CTS_Hourly[sunit]))))

    # Output: without normalize
    Dyna = CTS_Hourly[['index', 'type', 'Time', sunit] + POI_Type]
    Dyna.columns = ['dyna_id', 'type', 'time', 'entity_id'] + POI_Type
    print(Dyna.isnull().sum())
    Dyna.to_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_na, f_na), index=0)
    Dyna_s = CTS_Hourly[['index', 'type', 'Time', sunit, 'All']]
    Dyna_s.columns = ['dyna_id', 'type', 'time', 'entity_id', 'Visits']
    print(Dyna_s.isnull().sum())
    Dyna_s[['dyna_id', 'type', 'time', 'entity_id', 'Visits']].to_csv(
        results_path + r'Lib_Data\%s\%s.dyna' % (f_nas, f_nas), index=0)

    # Group-normalize
    ct_visit_mean = CTS_Hourly_train.groupby([sunit])[POI_Type + ['All']].mean().reset_index()
    ct_visit_mean.columns = [sunit] + [var + '_m' for var in list(ct_visit_mean.columns[1:])]
    ct_visit_std = CTS_Hourly_train.groupby([sunit])[POI_Type + ['All']].std().reset_index()
    ct_visit_std.columns = [sunit] + [var + '_std' for var in list(ct_visit_std.columns[1:])]
    ct_visit_mstd = ct_visit_mean.merge(ct_visit_std, on=sunit)  # some zeros exist
    ct_visit_mstd.to_pickle(r'D:\ST_Graph\Results\%s_%s_visit_mstd.pkl' % (sunit, time_sp))
    CTS_Hourly = CTS_Hourly.merge(ct_visit_mstd, on=sunit)
    # for kk in POI_Type + ['All']: CTS_Hourly[kk] = (CTS_Hourly[kk] - CTS_Hourly[kk + '_m']) / CTS_Hourly[kk + '_std']
    for kk in POI_Type: CTS_Hourly[kk] = (CTS_Hourly[kk] - CTS_Hourly[kk + '_m']) / CTS_Hourly['All_std']
    CTS_Hourly['All'] = (CTS_Hourly['All'] - CTS_Hourly['All' + '_m']) / CTS_Hourly['All_std']
    print((CTS_Hourly['All'] - CTS_Hourly[POI_Type].sum(axis=1)).sum())
    CTS_Hourly = CTS_Hourly.fillna(0)

    # Output: with normalize
    Dyna_gp = CTS_Hourly[['index', 'type', 'Time', sunit] + POI_Type]
    Dyna_gp.columns = ['dyna_id', 'type', 'time', 'entity_id'] + POI_Type
    print(Dyna_gp.isnull().sum())
    Dyna_gp.to_csv(results_path + r'Lib_Data\%s\%s.dyna' % (f_gp, f_gp), index=0)
    Dyna_gps = CTS_Hourly[['index', 'type', 'Time', sunit, 'All']]
    Dyna_gps.columns = ['dyna_id', 'type', 'time', 'entity_id', 'Visits']
    Dyna_gps[['dyna_id', 'type', 'time', 'entity_id', 'Visits']].to_csv(
        results_path + r'Lib_Data\%s\%s.dyna' % (f_gps, f_gps), index=0)

    # Geo
    Geo_Info = CTS_Hourly.drop_duplicates(subset=sunit).reset_index(drop=True)
    Geo_Info = Geo_Info[[sunit]]
    # Add lat/lng
    if sunit == 'CTSFIPS':
        CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_cty_sub_2019.shp')
    elif sunit == 'CTractFIPS':
        CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_tract_2019.shp')
    elif sunit == 'CBGFIPS':
        CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_blck_grp_2019.shp')
    CBG_Info = CBG_Info[CBG_Info['GEOID'].isin(Geo_Info[sunit])]
    print(len(CBG_Info) == len(Geo_Info))
    CBG_Info = CBG_Info.to_crs("EPSG:4326")
    CBG_cen = pd.DataFrame({sunit: CBG_Info['GEOID'], 'x': CBG_Info.centroid.x, 'y': CBG_Info.centroid.y}).reset_index(
        drop=True)
    Geo_Info['coordinates'] = "[" + CBG_cen['x'].astype(str) + ', ' + CBG_cen['y'].astype(str) + "]"
    Geo_Info['type'] = 'Point'
    Geo_Info = Geo_Info[[sunit, 'type', 'coordinates']]
    Geo_Info.columns = ['geo_id', 'type', 'coordinates']
    for kk in f_list: Geo_Info.to_csv(results_path + r'Lib_Data\%s\%s.geo' % (kk, kk), index=0)

    # Rel: build via OD
    CTS_OD = pd.concat(
        map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_Sunit', '%s_OD_Weekly_*.pkl' % sunit))))
    CTS_OD['Time'] = pd.to_datetime(CTS_OD['Time'])
    CTS_OD = CTS_OD[CTS_OD['Time'] <= split_time]
    CTS_OD['Volume'] = CTS_OD[POI_Type].sum(axis=1)
    CTS_OD = CTS_OD[[sunit + '_O', sunit + '_D', 'Volume']]
    CTS_OD = CTS_OD.groupby([sunit + '_O', sunit + '_D']).sum().reset_index()
    CTS_D = CTS_OD.groupby([sunit + '_D'])['Volume'].sum().reset_index()
    CTS_D.columns = [sunit + '_D', 'Inflow']
    CTS_OD = CTS_OD.merge(CTS_D, on=sunit + '_D')
    CTS_OD['link_weight'] = CTS_OD['Volume'] / CTS_OD['Inflow']
    list_s = list(set(Geo_Info['geo_id']))
    # Only need those units within geo files
    CTS_OD_f = pd.DataFrame({sunit + '_O': list_s * len(list_s), sunit + '_D': np.repeat(list_s, len(list_s))})
    CTS_OD = CTS_OD.merge(CTS_OD_f, on=[sunit + '_O', sunit + '_D'], how='right')
    CTS_OD = CTS_OD.fillna(0).reset_index()
    CTS_OD['type'] = 'geo'
    CTS_OD = CTS_OD[['type', sunit + '_O', sunit + '_D', 'link_weight']]
    CTS_OD.columns = ['type', 'origin_id', 'destination_id', 'link_weight']
    print(np.sqrt(CTS_OD.shape[0]))
    CTS_OD = CTS_OD.sort_values(by=['origin_id', 'destination_id']).reset_index(drop=True).reset_index()
    CTS_OD.rename({'index': 'rel_id'}, axis=1, inplace=True)
    for kk in f_list: CTS_OD.to_csv(results_path + r'Lib_Data\%s\%s.rel' % (kk, kk), index=0)

    # Ext: holiday and weekend
    CTS_Hourly_ext = CTS_Hourly[['Time', 'Holiday', 'Weekend']]
    CTS_Hourly_ext = CTS_Hourly_ext.drop_duplicates().reset_index(drop=True).reset_index()
    CTS_Hourly_ext.columns = ['ext_id', 'time', 'holiday', 'weekend']

    # Ext: weather
    weather = pd.read_pickle(r'D:\ST_Graph\Results\weather_2019_bmc.pkl')
    weather['time'] = weather['DATE'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    wlist = ['wind', 'temp', 'rain', 'snow', 'vis']
    weather[wlist] = (weather[wlist] - weather[wlist].mean()) / weather[wlist].std()
    CTS_Hourly_ext = CTS_Hourly_ext.merge(weather[wlist + ['time']], on=['time'])

    # Ext: covid cases if during covid
    if '2020' in time_sp:
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
        Case_num = pd.read_csv(url, error_bad_lines=False).rename(columns={'FIPS': 'CTFIPS'})
        Case_num = Case_num.dropna(subset=['CTFIPS']).reset_index(drop=True)
        Case_num = Case_num[Case_num['Country_Region'] == 'US']
        Case_num = Case_num.drop(['UID', 'iso2', 'iso3', 'code3', 'Admin2', 'Province_State', 'Country_Region', 'Lat',
                                  'Long_', 'Combined_Key'], axis=1)
        Case_num = Case_num.melt(id_vars=['CTFIPS'], var_name='date')
        Case_num['value'] = Case_num['value'].astype(int)
        Case_num['date'] = pd.to_datetime(Case_num['date'])
        Case_num = Case_num.sort_values(by=['CTFIPS', 'date']).reset_index(drop=True)
        Case_num['New_cases'] = Case_num.groupby('CTFIPS')['value'].diff()
        Case_num.loc[Case_num['New_cases'].isna(), 'New_cases'] = Case_num.loc[Case_num['New_cases'].isna(), 'value']
        Case_num.rename({'value': 'Agg_cases', 'date': 'Date'}, axis=1, inplace=True)
        Case_num['CTFIPS'] = Case_num['CTFIPS'].astype(int).astype(str).apply(lambda x: x.zfill(5))
        # Some cases is negative, fill with 0
        num = Case_num._get_numeric_data()
        num[num < 0] = 0
        del num
        Case_num = Case_num[Case_num['CTFIPS'].isin(ct_list)].reset_index(drop=True)
        Case_num = Case_num.groupby(['Date']).sum()['New_cases'].reset_index()
        CTS_Hourly_ext['Date'] = pd.to_datetime(CTS_Hourly_ext['time'].str[0:10])
        CTS_Hourly_ext = CTS_Hourly_ext.merge(Case_num, on='Date', how='left')
        CTS_Hourly_ext = CTS_Hourly_ext.fillna(0)
        CTS_Hourly_ext = CTS_Hourly_ext.drop('Date', axis=1)
        CTS_Hourly_ext['New_cases'] = (CTS_Hourly_ext['New_cases'] - CTS_Hourly_ext['New_cases'].mean()) / \
                                      CTS_Hourly_ext['New_cases'].std()
    for kk in f_list: CTS_Hourly_ext.to_csv(results_path + r'Lib_Data\%s\%s.ext' % (kk, kk), index=0)
    # Ext: static
    statics = pd.read_csv(r'D:\ST_Graph\Results\%s_Hourly_GP.static' % sunit)
    statics['geo_id'] = statics['geo_id'].astype(str)
    statics = statics.merge(Geo_Info[['geo_id']], on='geo_id', how='right')
    statics = statics.fillna(statics.mean())
    print(statics.shape)
    for kk in f_list: statics.to_csv(results_path + r'Lib_Data\%s\%s.static' % (kk, kk), index=0)

    # Configure
    # with _GP: it is normalized by groups
    # with _Single: it only covers all volume (without POI types)
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
    config['ext'] = {'ext_id': 'num', 'time': 'other', 'holiday': 'num', 'weekend': 'num', 'wind': 'num', 'temp': 'num',
                     'rain': 'num', 'snow': 'num', 'vis': 'num', 'New_cases': 'num'}
    config['info'] = dict()
    config['info']['data_col'] = ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']
    config['info']['weight_col'] = 'link_weight'
    config['info']['ext_col'] = ['holiday', 'weekend'] + wlist
    if '2020' in time_sp: config['info']['ext_col'] = ['holiday', 'weekend'] + wlist + ['New_cases']
    config['info']['data_files'] = ['%s_SG_%s_Hourly' % (time_sp, sunit)]
    config['info']['geo_file'] = '%s_SG_%s_Hourly' % (time_sp, sunit)
    config['info']['rel_file'] = '%s_SG_%s_Hourly' % (time_sp, sunit)
    config['info']['ext_file'] = '%s_SG_%s_Hourly' % (time_sp, sunit)
    config['info']['output_dim'] = 7
    config['info']['time_intervals'] = 3600
    config['info']['init_weight_inf_or_zero'] = 'zero'
    config['info']['set_weight_link_or_dist'] = 'dist'
    config['info']['calculate_weight_adj'] = False
    config['info']['weight_adj_epsilon'] = 0.1
    json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_na, 'w', encoding='utf-8'), ensure_ascii=False)
    config['info']['data_files'] = ['%s_SG_%s_Hourly_GP' % (time_sp, sunit)]
    config['info']['geo_file'] = '%s_SG_%s_Hourly_GP' % (time_sp, sunit)
    config['info']['rel_file'] = '%s_SG_%s_Hourly_GP' % (time_sp, sunit)
    config['info']['ext_file'] = '%s_SG_%s_Hourly_GP' % (time_sp, sunit)
    json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_gp, 'w', encoding='utf-8'), ensure_ascii=False)

    # Configure: Single POI
    config['dyna']['state'] = {'entity_id': 'geo_id', 'Visits': 'num'}
    config['info']['data_col'] = ['Visits']
    config['info']['data_files'] = ['%s_SG_%s_Hourly_Single' % (time_sp, sunit)]
    config['info']['geo_file'] = '%s_SG_%s_Hourly_Single' % (time_sp, sunit)
    config['info']['rel_file'] = '%s_SG_%s_Hourly_Single' % (time_sp, sunit)
    config['info']['ext_file'] = '%s_SG_%s_Hourly_Single' % (time_sp, sunit)
    config['info']['output_dim'] = 1
    json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_nas, 'w', encoding='utf-8'),
              ensure_ascii=False)
    config['info']['data_files'] = ['%s_SG_%s_Hourly_Single_GP' % (time_sp, sunit)]
    config['info']['geo_file'] = '%s_SG_%s_Hourly_Single_GP' % (time_sp, sunit)
    config['info']['rel_file'] = '%s_SG_%s_Hourly_Single_GP' % (time_sp, sunit)
    config['info']['ext_file'] = '%s_SG_%s_Hourly_Single_GP' % (time_sp, sunit)
    json.dump(config, open(results_path + r'Lib_Data\%s\config.json' % f_gps, 'w', encoding='utf-8'),
              ensure_ascii=False)
    print('------------------Finished: %s!!-----------------' % sunit)
