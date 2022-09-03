import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from functools import reduce
import glob
import geopandas as gpd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import json
import numpy as np

pd.options.mode.chained_assignment = None
results_path = r'C:\Users\huson\PycharmProjects\Bigscity-LibCity\libcity\cache\6050\evaluate_cache\\'

Predict_R = np.load(results_path + '2022_09_03_00_33_40_FNN_SG_CTS_Hourly_Single_predictions.npz')
print(Predict_R['prediction'].shape)
Predict_Real = pd.DataFrame({'prediction': Predict_R['prediction'].flatten(), 'truth': Predict_R['truth'].flatten()})

plt.plot(Predict_R['prediction'][:, 23, 1, 0], label='prediction')
plt.plot(Predict_R['truth'][:, 23, 1, 0], label='truth')
plt.legend()
