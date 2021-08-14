from os import DirEntry
import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn.model_selection import ShuffleSplit

city_id = 1
district_data = pd.read_csv("district.csv")
aq_station_data = pd.read_csv("station.csv")
aq_data = pd.read_csv("airquality.csv")   #READ ONLY ONE COLUMN HERE 
met_data = pd.read_csv("meteorology.csv")

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
    for col in station_one_data.columns:
        station_one_data[col] = station_one_data[col].interpolate(method='nearest')
    dates = pd.to_datetime([datetime.fromisoformat(station_one_data['time'][i]) for i in range(len(station_one_data['time']))])
    time_differences.append((dates - dates[0]).total_seconds() / 3600)
    all_stations_data.append(station_one_data)


lstm_samples = []
lstm_sample_stationids = []
days = 0 
start_pos = np.zeros(len(all_stations_data),dtype=np.int64)
while days!=364:
    one_sample = []
    added = []
    for i in range(len(all_stations_data)):
        try:
            one_day = list(map(lambda x: x> (days+1)*24, list(time_differences[i]))).index(True)
        except:
            continue
        sample_element = all_stations_data[i].iloc[start_pos[i]:one_day,:]
        start_pos[i] = one_day
        one_sample.append(sample_element)
        added.append(i)
    lstm_samples.append(one_sample)
    lstm_sample_stationids.append(added)
    days+=1

#LSTM_SAMPLES SIZE: 364,36,M+AQ,~24 (HOURLY DATA)

indices = ShuffleSplit(n_splits=10, test_size=.33, random_state=0).split(aq_station_ids) # FOR 10 RANDOM TEST TRAIN SPLITS

for i in indices: # FOR EACH SPLIT
    train_data = []
    test_indices = i[1]
    local_data = []
    for j in range(len(lstm_samples)):
        train_data_sample = []
        local_data_sample = []
        train_indices = list(i[0])
        for l in range(len(i[0])):
            removed = train_indices.pop()
            if removed in lstm_sample_stationids[j]:
                train_data_sample_element = [lstm_samples[j][lstm_sample_stationids[j].index(train_indices[k])] for k in range(len(train_indices)) if train_indices[k] in lstm_sample_stationids[j]]
                local_data_sample_element = lstm_samples[j][lstm_sample_stationids[j].index(removed)] 
            train_data_sample.append(train_data_sample_element) # SIZE: 24 (TRAIN STATIONS), 23 (REMAINING EXCEPT ONE LOCAL) , M+AQ, ~24
            local_data_sample.append(local_data_sample_element) # SIZE: 24 (TRAIN STATIONS) , M+AQ , ~24

        local_data.append(local_data_sample) # SIZE: 364,24,23,M+AQ,~24
        train_data.append(train_data_sample) # SIZE: 364,24,M+AQ,~24
    
    local_data = np.array(local_data,dtype=object).flatten() #SIZE: 8736, M+AQ,~24
    station_lstm_data  = np.array(train_data,dtype=object).flatten() #SIZE: 8736,23,M+AQ,~24 

    local_lstm_data = np.array([local_data[k][met_data.columns] for k in range(len(local_data))]) #SIZE: 8736, M,~24
   


    

