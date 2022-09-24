####################################
# Prepare population inflow data in BMC area from SafeGraph.
# Split the inflow by POI type.
# Prepare OD flow data for graph construction.
####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import glob
import geopandas as gpd

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
CTS_Info.to_pickle(r'D:\ST_Graph\Results\CTS_Info.pkl')
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
BMCPOI.to_pickle(r'D:\ST_Graph\Results\BMCPOI.pkl')

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
CBG_CTS.to_pickle(r'D:\ST_Graph\Results\CBG_CTS.pkl')

# SJoin: POI to CBG/Census Tract
SInCBG = gpd.sjoin(allPOI, CBG_Info, how='inner', op='within').reset_index(drop=True)
BMCPOI = BMCPOI.merge(SInCBG[['safegraph_place_id', 'CBGFIPS']], on='safegraph_place_id')
BMCPOI['CTractFIPS'] = BMCPOI['CBGFIPS'].str[0:11]
print('Len of Block Group: %s' % len(set(BMCPOI['CBGFIPS'])))
print('Len of Census Tract: %s' % len(set(BMCPOI['CTractFIPS'])))
print('Len of County Subdivision: %s' % len(set(BMCPOI['CTSFIPS'])))
BMCPOI.to_pickle(r'D:\ST_Graph\Results\BMCPOI_0922.pkl')

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

# Read and Process visit data in BMC area:
t_start = datetime.datetime(2018, 1, 1)
t_end = datetime.datetime(2020, 11, 23)
range_year = [d.strftime('%Y') for d in pd.date_range(t_start, t_end, freq='7D')]
range_month = [d.strftime('%m') for d in pd.date_range(t_start, t_end, freq='7D')]
range_date = [d.strftime('%d') for d in pd.date_range(t_start, t_end, freq='7D')]
BMCPOI = pd.read_pickle(r'D:\ST_Graph\Results\BMCPOI_0922.pkl')
for jj in range(0, len(range_year)):
    start = datetime.datetime.now()
    print(str(range_year[jj]) + '\\' + str(range_month[jj]) + '\\' + str(range_date[jj]))
    week_visit = pd.read_pickle(results_path + 'SG_Raw\\POI_BMC_Raw_' + str(range_year[jj]) + '-' + str(range_month[jj])
                                + '-' + str(range_date[jj]) + '.pkl')
    week_visit = week_visit.drop('top_category', axis=1)
    week_visit = week_visit.merge(BMCPOI[['safegraph_place_id', 'CTSFIPS', 'CBGFIPS', 'CTractFIPS', 'top_category']],
                                  on='safegraph_place_id')

    # Hourly visit by POI type
    hour_range = [d.strftime('%Y-%m-%d %H:00:00') for d in
                  pd.date_range(week_visit.loc[0, 'date_range_start'].split('T')[0],
                                week_visit.loc[0, 'date_range_end'].split('T')[0], freq='h')][0: -1]
    hour_visit = pd.DataFrame(week_visit['visits_by_each_hour'].str[1:-1].str.split(',').tolist()).astype(int)
    hour_visit.columns = hour_range
    hour_visit[['CTSFIPS', 'CBGFIPS', 'CTractFIPS', 'top_category']] = week_visit[
        ['CTSFIPS', 'CBGFIPS', 'CTractFIPS', 'top_category']]

    # Agg by CTSFIPS
    for sunit in ['CTSFIPS', 'CBGFIPS', 'CTractFIPS']:
        visit_agg = hour_visit.groupby(['top_category', sunit]).sum().reset_index()
        visit_agg_m = pd.melt(visit_agg, id_vars=['top_category', sunit], value_vars=hour_range)
        visit_agg_m.columns = ['top_category', sunit, 'Time', 'Visits']
        visit_agg_m = visit_agg_m.pivot(index=[sunit, 'Time'], columns='top_category', values='Visits').reset_index()

        # Fill to full time range
        F_time = pd.DataFrame({'Time': len(set(visit_agg_m[sunit])) * list(set(visit_agg_m['Time'])),
                               sunit: np.repeat(list(set(visit_agg_m[sunit])), len(set(visit_agg_m['Time'])))})
        F_time = F_time.sort_values(by=[sunit, 'Time'])
        visit_agg_m = visit_agg_m.merge(F_time, on=[sunit, 'Time'], how='right')

        visit_agg_m.to_pickle(results_path + 'SG_Sunit\\%s_Visit_Hourly_%s-%s-%s.pkl' % (
            sunit, str(range_year[jj]), str(range_month[jj]), str(range_date[jj])))

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
    cbg_flow = pd.DataFrame(flows_unit, columns=["safegraph_place_id", "CBGFIPS_D", "CBGFIPS_O", "OD_flow"])
    CBG_CTS.columns = ['CBGFIPS_D', 'CTSFIPS_D']
    cbg_flow = cbg_flow.merge(CBG_CTS, on='CBGFIPS_D', how='left')
    CBG_CTS.columns = ['CBGFIPS_O', 'CTSFIPS_O']
    cbg_flow = cbg_flow.merge(CBG_CTS, on='CBGFIPS_O', how='left')
    cbg_flow = cbg_flow.merge(BMCPOI[['safegraph_place_id', 'CTSFIPS', 'top_category']], on='safegraph_place_id')
    cbg_flow['CTractFIPS_O'] = cbg_flow['CBGFIPS_O'].str[0:11]
    cbg_flow['CTractFIPS_D'] = cbg_flow['CBGFIPS_D'].str[0:11]
    cbg_flow = cbg_flow.fillna(0)

    for sunit in ['CTSFIPS', 'CBGFIPS', 'CTractFIPS']:
        flow_poi = cbg_flow.groupby([sunit + '_O', sunit + '_D', 'top_category']).sum()['OD_flow'].reset_index()
        flow_poi = flow_poi.pivot(index=[sunit + '_O', sunit + '_D'], columns='top_category',
                                  values='OD_flow').reset_index()
        flow_poi['Time'] = hour_range[0]
        flow_poi = flow_poi.fillna(0)
        # Output
        flow_poi.to_pickle(results_path + 'SG_Sunit\\%s_OD_Weekly_%s-%s-%s.pkl' % (
            sunit, str(range_year[jj]), str(range_month[jj]), str(range_date[jj])))

    print(datetime.datetime.now() - start)
