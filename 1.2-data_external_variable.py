########################################################################
# Prepare external information: weather, holiday,
# Static variables (POI types, socio-economics, demographics, etc)
########################################################################
import pandas as pd
import numpy as np
import os
import datetime
import glob
import geopandas as gpd

pd.options.mode.chained_assignment = None
results_path = r'D:\\ST_Graph\\Data\\'

# Get county subdivision
CTS_Info = pd.read_pickle(r'D:\ST_Graph\Results\CTS_Info.pkl')
t_s = datetime.datetime(2019, 1, 1)
t_e = datetime.datetime(2020, 12, 1)

# Ext: add weather
# Station in BMC
g_stat = pd.read_pickle(r'E:\Weather\Daily\weather_raw_2019.pkl')
g_stat = g_stat[['STATION', 'LATITUDE', 'LONGITUDE']]
g_stat = g_stat.drop_duplicates(subset=['STATION']).dropna()
g_stat_s = gpd.GeoDataFrame(g_stat, geometry=gpd.points_from_xy(g_stat['LONGITUDE'], g_stat['LATITUDE']))
ghcnd_station_s = g_stat_s.set_crs('EPSG:4326')
SInBMC = gpd.sjoin(ghcnd_station_s, CTS_Info, how='inner', op='within').reset_index(drop=True)
SInBMC = SInBMC[['STATION', 'LATITUDE', 'LONGITUDE', 'CTSFIPS']]

# Read weather data
afiles = glob.glob(r'E:\Weather\Hourly\2019\*.csv') + glob.glob(r'E:\Weather\Hourly\2020\*.csv')
nlist = list(SInBMC['STATION'].astype(str))
nfiles = [i for e in nlist for i in afiles if e in i]
hourly_wea = pd.concat(map(pd.read_csv, nfiles)).reset_index(drop=True)
hourly_wea = hourly_wea.merge(SInBMC[['STATION', 'CTSFIPS']], on='STATION')

# Only need wnd (m/s), tmp (Celcius), vis ( horizontal distance meter), AA1 (rain millimeters), AJ1 (snow centimeters)
# https://www.visualcrossing.com/resources/documentation/weather-data/how-we-process-integrated-surface-database-historical-weather-data/
# https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# https://www.ncei.noaa.gov/data/global-hourly/doc/CSV_HELP.pdf
hourly_wea['vis'] = hourly_wea['VIS'].str.split(',').str[0].astype(float)  # m
hourly_wea['wind'] = hourly_wea['WND'].str.split(',').str[3].astype(float) * 0.1  # m/s
hourly_wea['temp'] = hourly_wea['TMP'].str.split(',').str[0].astype(float) * 0.1  # Celcius
hourly_wea['rain'] = hourly_wea['AA1'].str.split(',').str[1].astype(float) * 0.1  # millimeters
hourly_wea['snow'] = hourly_wea['AJ1'].str.split(',').str[0].astype(float) * 10  # millimeters
hourly_wea = hourly_wea[['STATION', 'CTSFIPS', 'DATE', 'LATITUDE', 'LONGITUDE', 'wind', 'temp', 'rain', 'snow', 'vis']]
hourly_wea['DATE'] = pd.to_datetime(hourly_wea['DATE']).dt.round('H')
hourly_wea.describe().T[['count', 'mean', 'min', 'max']]
# Handle outliers:
hourly_wea.loc[hourly_wea['temp'] < -25, 'temp'] = np.nan
for kk in ['wind', 'temp', 'rain', 'vis']:
    hourly_wea = hourly_wea.replace(hourly_wea[kk].max(), np.nan)
hourly_wea.describe().T[['count', 'mean', 'min', 'max']]
# Fillna
hourly_wea['rain'] = hourly_wea['rain'].fillna(0)
hourly_wea['snow'] = hourly_wea['snow'].fillna(0)
for kk in ['wind', 'temp', 'vis']:
    hourly_wea[kk] = hourly_wea[kk].fillna(hourly_wea.groupby('DATE')[kk].transform('median'))
# group mean
hourly_wea_mean = hourly_wea.groupby(['DATE']).mean().reset_index()
hourly_wea_mean = hourly_wea_mean[(hourly_wea_mean['DATE'] < t_e) & (hourly_wea_mean['DATE'] >= t_s)].reset_index(
    drop=True)
hourly_wea_mean[['DATE', 'wind', 'temp', 'rain', 'snow', 'vis']].to_pickle(r'D:\ST_Graph\Results\weather_2019_bmc.pkl')
# plt.plot(hourly_wea_mean['rain'])
# hourly_wea.groupby(['CTSFIPS', 'DATE']).mean()['rain'].plot()

# Add socio-economic data
# POT INFO
BMCPOI = pd.read_pickle(r'D:\ST_Graph\Results\BMCPOI_0922.pkl')
CBG_CTS = pd.read_pickle(r'D:\ST_Graph\Results\CBG_CTS.pkl')
for sunit in ['CTSFIPS', 'CBGFIPS', 'CTractFIPS']:  # 'CBGFIPS', 'CTractFIPS'
    BMCPOI_count = BMCPOI.groupby([sunit, 'top_category']).count()['safegraph_place_id'].reset_index()
    BMCPOI_count = BMCPOI_count.pivot(index=sunit, columns='top_category', values='safegraph_place_id').reset_index()
    BMCPOI_count = BMCPOI_count.fillna(0)

    # Income etc.
    CBG_Features = pd.read_csv(r'E:\Research\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
    CBG_Features['CBGFIPS'] = CBG_Features['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
    CBG_Features = CBG_Features.merge(CBG_CTS, on='CBGFIPS')
    CBG_Features['CTractFIPS'] = CBG_Features['CBGFIPS'].str[0:11]

    CTS_SUM_POP = CBG_Features.groupby([sunit]).sum()[['Total_Population', 'ALAND']].reset_index()
    CTS_SUM_POP.columns = [sunit, 'Total_Population_' + sunit, 'ALAND_' + sunit]
    CBG_Features = CBG_Features.merge(CTS_SUM_POP, on=sunit)

    # To abs and then covert to pct
    abslist = ['Median_income', 'Democrat_R', 'Republican_R', 'Urbanized_Areas_Population_R', 'HISPANIC_LATINO_R',
               'Black_R', 'Asian_R', 'Bt_18_44_R', 'Bt_45_64_R', 'Over_65_R', 'Male_R', 'White_Non_Hispanic_R',
               'White_Hispanic_R', 'Education_Degree_R']
    for kk in abslist: CBG_Features[kk] = CBG_Features[kk] * CBG_Features['Total_Population']
    CBG_Features_sum = CBG_Features.groupby([sunit]).sum()[abslist].reset_index()
    CBG_Features_sum = CBG_Features_sum.merge(CTS_SUM_POP, on=sunit)
    for kk in abslist: CBG_Features_sum[kk] = CBG_Features_sum[kk] / CBG_Features_sum['Total_Population_' + sunit]

    CTS_Socio = CBG_Features_sum[[sunit] + abslist]
    CTS_Socio = CTS_Socio.merge(CTS_SUM_POP, on=sunit)
    CTS_Socio = CTS_Socio.merge(BMCPOI_count, on=sunit)
    CTS_Socio = CTS_Socio.sort_values(by=[sunit]).reset_index(drop=True)
    CTS_Socio.to_pickle(r'D:\ST_Graph\Results\%s_Socio_bmc.pkl' % sunit)
    for kk in list(CTS_Socio.columns)[1:]: CTS_Socio[kk] = (CTS_Socio[kk] - CTS_Socio[kk].mean()) / CTS_Socio[kk].std()
    CTS_Socio.rename({sunit: 'geo_id'}, axis=1, inplace=True)
    CTS_Socio.to_csv(r'D:\ST_Graph\Results\%s_Hourly_GP.static' % sunit, index=0)
