from os import DirEntry
import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn.model_selection import ShuffleSplit

city_id = 1
district_data = pd.read_csv("data/raw/district.csv")
aq_station_data = pd.read_csv("data/raw/station.csv")
aq_data = pd.read_csv("data/raw/airquality.csv.gz")   #READ ONLY ONE COLUMN HERE 
met_data = pd.read_csv("data/raw/meteorology.csv.gz")

district_ids = np.array(district_data['district_id'])[np.where(np.array(district_data['city_id'])==1)]
district_rows = np.array([i for i in range(len(aq_station_data)) if np.array(aq_station_data['district_id'])[i] in district_ids])
aq_station_ids,aq_dis_ids = np.array(aq_station_data['station_id'])[district_rows],np.array(aq_station_data['district_id'])[district_rows]

all_stations_data = []
time_differences = []

for aqid,disid in zip(aq_station_ids,aq_dis_ids):

    station_rows = np.where(np.isin(aq_data['station_id'],aqid))[0]
    stations_data = aq_data.iloc[station_rows,:]
    met_rows = np.where(np.isin(met_data['id'],disid))[0]
    met_station= met_data.iloc[met_rows,:]
    station_one_data = pd.merge(stations_data, met_station, on='time')
    # for col in station_one_data.columns:
    #     station_one_data[col] = station_one_data[col].interpolate(method='nearest')
    dates = pd.to_datetime([datetime.fromisoformat(station_one_data['time'][i]) for i in range(len(station_one_data['time']))])
    time_differences.append((dates - dates[0]).total_seconds() / 3600)
    all_stations_data.append(station_one_data)

print(pd.concat(all_stations_data).to_csv('tmp.csv.gz'))

   


    

