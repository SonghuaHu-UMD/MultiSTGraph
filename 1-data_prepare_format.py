import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import glob
import geopandas as gpd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import json

pd.options.mode.chained_assignment = None
results_path = r'D:\\ST_Graph\\Data\\'
geo_path = r'E:\SafeGraph\Open Census Data\Census Website\2019\\'

# Get BMC area (TAZ)
bmc_zone = gpd.read_file(r'D:\OD_Social_Connect\BMC\Complete_BMC_Shapefile\Final_BMC_Shapefile.shp')
bmc_zone = bmc_zone[~bmc_zone['TAZID3'].isna()]
bmc_zone['TAZ10'] = bmc_zone['TAZ10'].astype(int).astype(str)
bmc_zone = bmc_zone.to_crs("EPSG:4326")
bmc_cen = gpd.GeoDataFrame(bmc_zone[['TAZ10']], geometry=gpd.points_from_xy(bmc_zone.centroid.x, bmc_zone.centroid.y))
bmc_cen = bmc_cen.set_crs('EPSG:4326')

# Get the county inside BMC area
CT_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\US_county_2019.shp')
CT_Info = CT_Info[CT_Info['STATEFP'].isin(['10', '11', '24', '51'])].reset_index(drop=True)
CT_Info['CTFIPS'] = CT_Info['GISJOIN'].str[1:3] + CT_Info['GISJOIN'].str[4:7]
CT_Info = CT_Info.to_crs('EPSG:4326')
SInCT = gpd.sjoin(bmc_cen, CT_Info, how='inner', op='within').reset_index(drop=True)
CT_Info_sj = CT_Info[CT_Info['CTFIPS'].isin(SInCT['CTFIPS'])]
# CT_Info_sj.plot()
CT_Info_ls = CT_Info_sj[['CTFIPS']].drop_duplicates().reset_index(drop=True)

# Get POI data inside BMC area
allPOI = pd.read_pickle(r'E:\SafeGraph\Core Places US (Nov 2020 - Present)\CoreRecords-CORE_POI-Present.pkl')
allPOI['naics_code'] = allPOI['naics_code'].fillna(-9999)
allPOI['naics_code'] = allPOI['naics_code'].astype('int64').astype(str).apply(lambda x: x.zfill(6))
poly_bound = CT_Info_sj.geometry.bounds
allPOI = allPOI[(allPOI['latitude'] < max(poly_bound['maxy'])) & (allPOI['latitude'] > min(poly_bound['miny'])) & (
        allPOI['longitude'] < max(poly_bound['maxx'])) & (allPOI['longitude'] > min(poly_bound['minx']))]
allPOI = gpd.GeoDataFrame(allPOI, geometry=gpd.points_from_xy(allPOI.longitude, allPOI.latitude))
allPOI = allPOI.set_crs('EPSG:4326')

# SJoin: POI to county subdivision
CTS_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_cty_sub_2019.shp')
CTS_Info = CTS_Info[CTS_Info['STATEFP'].isin(['10', '11', '24', '51'])].reset_index(drop=True)
CTS_Info['CTFIPS'] = CTS_Info['GISJOIN'].str[1:3] + CTS_Info['GISJOIN'].str[4:7]
CTS_Info = CTS_Info[CTS_Info['CTFIPS'].isin(CT_Info_ls['CTFIPS'])]
CTS_Info['CTSFIPS'] = CTS_Info['GEOID'].astype('int64').astype(str).apply(lambda x: x.zfill(10))
CTS_Info = CTS_Info.to_crs('EPSG:4326')
SInCTS = gpd.sjoin(allPOI, CTS_Info, how='inner', op='within').reset_index(drop=True)
BMCPOI = allPOI.merge(SInCTS[['safegraph_place_id', 'CTSFIPS']], on='safegraph_place_id')

fig, ax = plt.subplots(figsize=(10, 8))
allPOI.plot(ax=ax, markersize=1, alpha=0.5)
BMCPOI.plot(ax=ax, markersize=1, color='g', alpha=0.5)
CT_Info_sj.boundary.plot(ax=ax, color='k')
CTS_Info.boundary.plot(ax=ax, color='gray')
plt.tight_layout()
plt.savefig(r'D:\ST_Graph\Figures\zone_sys.png', dpi=1000)
plt.close()

# Reassign POI type
BMCPOI['NAICS02'] = BMCPOI['naics_code'].str[0:2]
BMCPOI['NAICS04'] = BMCPOI['naics_code'].str[0:4]
BMCPOI.loc[BMCPOI['NAICS02'].isin(
    ['00', '11', '21', '22', '23', '31', '32', '33', '48', '49', '51', '55']), 'top_category'] = 'Others'
BMCPOI.loc[BMCPOI['NAICS04'].isin(
    ['5321', '5322', '5323', '5324', '5331', '8111', '8112', '8113', '8114', '8121', '8122', '8123',
     '8129']), 'top_category'] = 'Service'
BMCPOI.loc[BMCPOI['NAICS02'].isin(['52', '54']), 'top_category'] = 'Service'
BMCPOI.loc[BMCPOI['NAICS04'].isin(['8131', '8132', '8133', '8134', '8139', '8141']), 'top_category'] = 'Service'
BMCPOI.loc[BMCPOI['NAICS02'].isin(['56', '92', ]), 'top_category'] = 'Service'
BMCPOI.loc[BMCPOI['NAICS02'].isin(['42', '44', '45']), 'top_category'] = 'Retail'
BMCPOI.loc[BMCPOI['NAICS04'].isin(['5311', '5312', '5313']), 'top_category'] = 'Residential'
BMCPOI.loc[BMCPOI['NAICS02'].isin(['61']), 'top_category'] = 'Education'
BMCPOI.loc[BMCPOI['NAICS04'].isin(['6244']), 'top_category'] = 'Education'
BMCPOI.loc[BMCPOI['NAICS04'].isin(
    ['6211', '6212', '6213', '6221', '6222', '6223', '6231', '6214', '6215', '6216', '6219', '6232', '6233', '6239',
     '6241', '6242', '6243', '0624']), 'top_category'] = 'Service'
BMCPOI.loc[BMCPOI['NAICS02'].isin(['71']), 'top_category'] = 'Recreation'
BMCPOI.loc[BMCPOI['NAICS04'].isin(['7211', '7212', '7213']), 'top_category'] = 'Restaurant'
BMCPOI.loc[BMCPOI['NAICS04'].isin(['7223', '7224', '7225']), 'top_category'] = 'Restaurant'
BMCPOI['top_category'] = BMCPOI['top_category'].fillna('Others')
print((set(BMCPOI['top_category'])))

# SJoin: CBG to county subdivision
CBG_Info = gpd.GeoDataFrame.from_file(geo_path + r'nhgis0011_shape\\US_blck_grp_2019.shp')
CBG_Info['CTFIPS'] = CBG_Info['GISJOIN'].str[1:3] + CBG_Info['GISJOIN'].str[4:7]
CBG_Info = CBG_Info[CBG_Info['CTFIPS'].isin(CT_Info_ls['CTFIPS'])]
CBG_Info = CBG_Info.to_crs("EPSG:4326")
CBG_Info['CBGFIPS'] = CBG_Info['GEOID'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
CBG_cen = gpd.GeoDataFrame(CBG_Info[['CBGFIPS']], geometry=gpd.points_from_xy(CBG_Info.centroid.x, CBG_Info.centroid.y))
CBG_cen = CBG_cen.set_crs('EPSG:4326')
SInCTS = gpd.sjoin(CBG_cen, CTS_Info, how='inner', op='within').reset_index(drop=True)
CBG_cen = CBG_cen.merge(SInCTS[['CBGFIPS', 'CTSFIPS']], on='CBGFIPS')
CBG_CTS = CBG_cen[['CBGFIPS', 'CTSFIPS']]

'''
# Read SG visit data and output those in BMC area
t_start = datetime.datetime(2018, 1, 1)
t_end = datetime.datetime(2020, 11, 23)
range_year = [d.strftime('%Y') for d in pd.date_range(t_start, t_end, freq='7D')]
range_month = [d.strftime('%m') for d in pd.date_range(t_start, t_end, freq='7D')]
range_date = [d.strftime('%d') for d in pd.date_range(t_start, t_end, freq='7D')]
for jj in range(0, len(range_year)):
    start = datetime.datetime.now()
    print(str(range_year[jj]) + '\\' + str(range_month[jj]) + '\\' + str(range_date[jj]))

    # change to the deepest subdir
    for dirpaths, dirnames, filenames in os.walk(
            "E:\\SafeGraph\\Weekly Places Patterns Backfill for Dec 2020 and Onward Release\\patterns_backfill\\2020\\12\\14\\21\\" + str(
                range_year[jj]) + '\\' + str(range_month[jj]) + '\\' + str(range_date[jj])):
        if not dirnames: os.chdir(dirpaths)

    print(dirpaths)
    week_visit = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "*.gz"))))
    week_visit = week_visit[week_visit['iso_country_code'] == 'US'].reset_index(drop=True)
    week_visit = week_visit[~week_visit['poi_cbg'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
    week_visit = week_visit.dropna(subset=['poi_cbg']).reset_index(drop=True)
    week_visit['poi_cbg'] = week_visit['poi_cbg'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
    week_visit['CTFIPS'] = week_visit['poi_cbg'].str[0:5]
    week_visit = week_visit.merge(CT_Info_ls, on='CTFIPS')

    # Merge with POI info
    week_visit = week_visit[
        ['safegraph_place_id', 'parent_safegraph_place_id', 'postal_code', 'date_range_start', 'date_range_end',
         'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day', 'visits_by_each_hour', 'poi_cbg',
         'visitor_home_cbgs', 'visitor_daytime_cbgs', 'visitor_country_of_origin', 'distance_from_home', 'median_dwell',
         'bucketed_dwell_times', 'CTFIPS']].merge(
        BMCPOI[['safegraph_place_id', 'top_category', 'sub_category', 'naics_code', 'latitude', 'longitude']],
        on='safegraph_place_id', how='left')

    # Output
    week_visit.to_pickle(results_path + 'SG_Raw\\POI_BMC_Raw_' + str(range_year[jj]) + '-' + str(range_month[jj])
                         + '-' + str(range_date[jj]) + '.pkl')
'''

'''
# Read and Process visit data in BMC area:
t_start = datetime.datetime(2018, 1, 1)
t_end = datetime.datetime(2020, 11, 23)
range_year = [d.strftime('%Y') for d in pd.date_range(t_start, t_end, freq='7D')]
range_month = [d.strftime('%m') for d in pd.date_range(t_start, t_end, freq='7D')]
range_date = [d.strftime('%d') for d in pd.date_range(t_start, t_end, freq='7D')]
for jj in range(0, len(range_year)):
    start = datetime.datetime.now()
    print(str(range_year[jj]) + '\\' + str(range_month[jj]) + '\\' + str(range_date[jj]))
    week_visit = pd.read_pickle(results_path + 'SG_Raw\\POI_BMC_Raw_' + str(range_year[jj]) + '-' + str(range_month[jj])
                                + '-' + str(range_date[jj]) + '.pkl')
    week_visit = week_visit.drop('top_category', axis=1)
    week_visit = week_visit.merge(BMCPOI[['safegraph_place_id', 'CTSFIPS', 'top_category']], on='safegraph_place_id')

    # Hourly visit by POI type
    hour_range = [d.strftime('%Y-%m-%d %H:00:00') for d in
                  pd.date_range(week_visit.loc[0, 'date_range_start'].split('T')[0],
                                week_visit.loc[0, 'date_range_end'].split('T')[0], freq='h')][0: -1]
    hour_visit = pd.DataFrame(week_visit['visits_by_each_hour'].str[1:-1].str.split(',').tolist()).astype(int)
    hour_visit.columns = hour_range
    hour_visit['CTSFIPS'] = week_visit['CTSFIPS']
    hour_visit['top_category'] = week_visit['top_category']

    # Agg by CTSFIPS
    visit_agg = hour_visit.groupby(['top_category', 'CTSFIPS']).sum().reset_index()
    visit_agg_m = pd.melt(visit_agg, id_vars=['top_category', 'CTSFIPS'], value_vars=hour_range)
    visit_agg_m.columns = ['top_category', 'CTSFIPS', 'Time', 'Visits']
    visit_agg_m = visit_agg_m.pivot(index=['CTSFIPS', 'Time'], columns='top_category', values='Visits').reset_index()
    # visit_agg_m.groupby(['variable']).sum().plot()

    # Time range
    F_time = pd.DataFrame({'Time': len(set(visit_agg_m['CTSFIPS'])) * list(set(visit_agg_m['Time'])),
                           'CTSFIPS': np.repeat(list(set(visit_agg_m['CTSFIPS'])), len(set(visit_agg_m['Time'])))})
    F_time = F_time.sort_values(by=['CTSFIPS', 'Time'])
    visit_agg_m = visit_agg_m.merge(F_time, on=['CTSFIPS', 'Time'], how='right')

    visit_agg_m.to_pickle(results_path + 'SG_PC\\CTS_Visit_Hourly_%s-%s-%s.pkl' % (
        str(range_year[jj]), str(range_month[jj]), str(range_date[jj])))

    # Weekly OD flow by POI type
    flows_unit = []
    for i, row in enumerate(week_visit.itertuples()):
        if row.visitor_home_cbgs == "{}":
            continue
        else:
            destination_id = row.safegraph_place_id
            destination_cbg = row.poi_cbg
            origin = eval(row.visitor_home_cbgs)
            for key, value in origin.items():
                flows_unit.append([destination_id, destination_cbg, str(key).zfill(12), value])
    cbg_flow = pd.DataFrame(flows_unit, columns=["safegraph_place_id", "cbg_d", "cbg_o", "OD_flow"])
    CBG_CTS.columns = ['cbg_d', 'CTSFIPS_D']
    cbg_flow = cbg_flow.merge(CBG_CTS, on='cbg_d', how='left')
    CBG_CTS.columns = ['cbg_o', 'CTSFIPS_O']
    cbg_flow = cbg_flow.merge(CBG_CTS, on='cbg_o', how='left')
    cbg_flow = cbg_flow.merge(BMCPOI[['safegraph_place_id', 'CTSFIPS', 'top_category']], on='safegraph_place_id')

    flow_poi = cbg_flow.groupby(['CTSFIPS_O', 'CTSFIPS_D', 'top_category']).sum()['OD_flow'].reset_index()
    flow_poi = flow_poi.pivot(index=['CTSFIPS_O', 'CTSFIPS_D'], columns='top_category', values='OD_flow').reset_index()
    flow_poi['Time'] = hour_range[0]
    flow_poi = flow_poi.fillna(0)

    # Ouput
    flow_poi.to_pickle(results_path + 'SG_PC\\CTS_OD_Weekly_%s-%s-%s.pkl' % (
        str(range_year[jj]), str(range_month[jj]), str(range_date[jj])))

    print(datetime.datetime.now() - start)
'''

# Merge all and output to lib-city format
# Dynamic
CTS_Hourly = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_PC', 'CTS_Visit_Hourly*.pkl'))))
CTS_Hourly['Time'] = pd.to_datetime(CTS_Hourly['Time'])
CTS_Hourly = CTS_Hourly.fillna(0)
cal = calendar()
holidays = cal.holidays(start=CTS_Hourly['Time'].dt.date.min(), end=CTS_Hourly['Time'].dt.date.max())
CTS_Hourly['Holiday'] = CTS_Hourly['Time'].dt.date.astype('datetime64').isin(holidays).astype(int)
CTS_Hourly = CTS_Hourly[
    (CTS_Hourly['Time'] < datetime.datetime(2020, 1, 1)) & (CTS_Hourly['Time'] >= datetime.datetime(2019, 1, 1))]
# CTS_Hourly.groupby(['Time']).sum().plot()
# print(len(CTS_Hourly) / len(set(CTS_Hourly['CTSFIPS'])) / 24)
CTS_Hourly = CTS_Hourly.sort_values(by=['CTSFIPS', 'Time']).reset_index(drop=True)
CTS_Hourly['Time'] = CTS_Hourly['Time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
CTS_Hourly = CTS_Hourly.reset_index()
CTS_Hourly['type'] = 'state'
CTS_Hourly_o = CTS_Hourly[['index', 'type', 'Time', 'CTSFIPS', 'Education', 'Others', 'Recreation',
                           'Residential', 'Restaurant', 'Retail', 'Service']]
CTS_Hourly_o.columns = ['dyna_id', 'type', 'time', 'entity_id', 'Education', 'Others', 'Recreation',
                        'Residential', 'Restaurant', 'Retail', 'Service']
CTS_Hourly_o.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly.dyna', index=0)
CTS_Hourly_s = CTS_Hourly_o.copy()
CTS_Hourly_s['Visits'] = CTS_Hourly_s[
    ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']].sum(axis=1)
CTS_Hourly_s[['dyna_id', 'type', 'time', 'entity_id', 'Visits']].to_csv(
    r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly_Single.dyna', index=0)

# Geo
CTS_Info['x'] = CTS_Info.centroid.x
CTS_Info['y'] = CTS_Info.centroid.y
CTS_Info['coordinates'] = "[" + CTS_Info['x'].astype(str) + ', ' + CTS_Info['y'].astype(str) + "]"
CTS_Info['type'] = 'Point'
CTS_Info_out = CTS_Info[['CTSFIPS', 'type', 'coordinates']]
CTS_Info_out.columns = ['geo_id', 'type', 'coordinates']
CTS_Info_out.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly.geo', index=0)
CTS_Info_out.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly_Single.geo', index=0)

# Rel: build via OD
CTS_OD = pd.concat(map(pd.read_pickle, glob.glob(os.path.join(r'D:\ST_Graph\Data\SG_PC', 'CTS_OD_Weekly_*.pkl'))))
CTS_OD['Volume'] = CTS_OD[['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']].sum(
    axis=1)
CTS_OD = CTS_OD[['CTSFIPS_O', 'CTSFIPS_D', 'Volume']]
CTS_OD = CTS_OD.groupby(['CTSFIPS_O', 'CTSFIPS_D']).sum().reset_index()
CTS_D = CTS_OD.groupby(['CTSFIPS_D'])['Volume'].sum().reset_index()
CTS_D.columns = ['CTSFIPS_D', 'Inflow']
CTS_OD = CTS_OD.merge(CTS_D, on='CTSFIPS_D')
CTS_OD['link_weight'] = CTS_OD['Volume'] / CTS_OD['Inflow']
CTS_OD['type'] = 'geo'
CTS_OD = CTS_OD.reset_index()
CTS_OD = CTS_OD[['index', 'type', 'CTSFIPS_O', 'CTSFIPS_D', 'link_weight']]
CTS_OD.columns = ['rel_id', 'type', 'origin_id', 'destination_id', 'link_weight']
CTS_OD.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly.rel', index=0)
CTS_OD.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly_Single.rel', index=0)

# Ext: add holiday
CTS_Hourly_ext = CTS_Hourly[['Time', 'Holiday']]
CTS_Hourly_ext = CTS_Hourly_ext.drop_duplicates().reset_index(drop=True).reset_index()
CTS_Hourly_ext.columns = ['ext_id', 'time', 'holiday']
CTS_Hourly_ext.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly.ext', index=0)
CTS_Hourly_ext.to_csv(r'D:\ST_Graph\Data\Lib_Data\SG_CTS_Hourly_Single.ext', index=0)

# Configure
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
config['ext'] = {'ext_id': 'num', 'time': 'other', 'holiday': 'num'}
config['info'] = dict()
config['info']['data_col'] = ['Education', 'Others', 'Recreation', 'Residential', 'Restaurant', 'Retail', 'Service']
config['info']['weight_col'] = 'link_weight'
config['info']['ext_col'] = ['holiday']
config['info']['data_files'] = ['SG_CTS_Hourly']
config['info']['geo_file'] = 'SG_CTS_Hourly'
config['info']['rel_file'] = 'SG_CTS_Hourly'
config['info']['ext_file'] = 'SG_CTS_Hourly'
config['info']['output_dim'] = 7
config['info']['time_intervals'] = 3600
config['info']['init_weight_inf_or_zero'] = 'inf'
config['info']['set_weight_link_or_dist'] = 'dist'
config['info']['calculate_weight_adj'] = False
config['info']['weight_adj_epsilon'] = 0.1
json.dump(config, open(r'D:\ST_Graph\Data\Lib_Data\\' + 'config.json', 'w', encoding='utf-8'), ensure_ascii=False)

# One output
config = dict()
config['geo'] = dict()
config['geo']['including_types'] = ['Point']
config['geo']['Point'] = {}
config['rel'] = dict()
config['rel']['including_types'] = ['geo']
config['rel']['geo'] = {'link_weight': 'num'}
config['dyna'] = dict()
config['dyna']['including_types'] = ['state']
config['dyna']['state'] = {'entity_id': 'geo_id', 'Visits': 'num'}
config['ext'] = {'ext_id': 'num', 'time': 'other', 'holiday': 'num'}
config['info'] = dict()
config['info']['data_col'] = ['Visits']
config['info']['weight_col'] = 'link_weight'
config['info']['ext_col'] = ['holiday']
config['info']['data_files'] = ['SG_CTS_Hourly_Single']
config['info']['geo_file'] = 'SG_CTS_Hourly_Single'
config['info']['rel_file'] = 'SG_CTS_Hourly_Single'
config['info']['ext_file'] = 'SG_CTS_Hourly_Single'
config['info']['output_dim'] = 1
config['info']['time_intervals'] = 3600
config['info']['init_weight_inf_or_zero'] = 'inf'
config['info']['set_weight_link_or_dist'] = 'dist'
config['info']['calculate_weight_adj'] = False
config['info']['weight_adj_epsilon'] = 0.1
json.dump(config, open(r'D:\ST_Graph\Data\Lib_Data\\' + 'config_Single.json', 'w', encoding='utf-8'),
          ensure_ascii=False)
