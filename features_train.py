import pandas as pd
import numpy as np
import math
import gc


TIME_START = pd.Timestamp(year=2021, month=9, day=1, hour=0)



def gas_features(df_gas_prices, latency=1):

    if 'data_block_id' not in df_gas_prices.columns:
        df_gas_prices['data_block_id'] = (pd.to_datetime(df_gas_prices['forecast_date']) - TIME_START).dt.days + latency

    return df_gas_prices.fillna(method='ffill')


def electricity_features(df_electricity_prices, latency=1):

    if 'data_block_id' not in df_electricity_prices.columns:
        df_electricity_prices['data_block_id'] = (pd.to_datetime(df_electricity_prices['forecast_date']) - TIME_START).dt.days + latency

    df_electricity_prices['time_id'] = df_electricity_prices['data_block_id']*24 + pd.to_datetime(df_electricity_prices['forecast_date']).dt.hour
    df_electricity_prices = df_electricity_prices.drop(['data_block_id'], axis=1)
    
    return df_electricity_prices.fillna(method='ffill')


def client_features(df_client, latency=2):

    if 'data_block_id' not in df_client.columns:
        df_client['data_block_id'] = (pd.to_datetime(df_client['date']) - TIME_START).dt.days + latency
    
    return df_client.fillna(method='ffill')


def time_features(datetime):

    time = (datetime - TIME_START).dt.days*24 + datetime.dt.hour

    df = pd.DataFrame()
    df['time_id'] = time
    
    df['year_sine'] = [np.sin((x+244)*math.pi * 2 / (365*24)) for x in df['time_id']]
    df['year_cosine'] = [np.cos((x+244)*math.pi * 2 / (365*24)) for x in df['time_id']]
    df['month_sine'] = [np.sin(x*math.pi * 2 / (6*30*24)) for x in df['time_id']]
    df['month_cosine'] = [np.cos(x*math.pi * 2 / (6*30*24)) for x in df['time_id']]
    df['day_sine'] = [np.sin(x*math.pi * 2 / (24)) for x in df['time_id']]
    df['day_cosine'] = [np.cos(x*math.pi * 2 / (24)) for x in df['time_id']]
    df['day_of_week'] = datetime.dt.day_of_week
    df['is_weekend'] = (datetime.dt.day_of_week > 5).astype('int')
    df['month'] = datetime.dt.month
    df['day_of_month'] = datetime.dt.day 

    return df



def historical_weather_features(df_historical_weather, dict_county):
    
    df_historical_weather['lat_lon'] = df_historical_weather.apply(lambda row: \
                                                     f'{row["latitude"]:.1f}_{row["longitude"]:.1f}', axis=1)

    df_historical_weather = df_historical_weather.loc[df_historical_weather['lat_lon'].isin(dict_county.keys())]
    df_historical_weather['county'] = df_historical_weather['lat_lon'].map(lambda x: dict_county[x])
    
    datetime = pd.to_datetime(df_historical_weather['datetime']) 
    df_historical_weather['time_id'] = ((datetime - pd.Timedelta(hours=11) - TIME_START).dt.days + 2)*24 + datetime.dt.hour 
    
    df_features = df_historical_weather.drop(['datetime', 'latitude', 'longitude', 'data_block_id', 'lat_lon'], axis=1).groupby(['time_id', 'county']).agg('mean').fillna(method='ffill').reset_index()
    
    return df_features




def forecast_weather_features(df_forecast_weather, dict_county):
    
    datetime = pd.to_datetime(df_forecast_weather['forecast_datetime']) 
    df_forecast_weather['time_id'] = ((datetime - TIME_START).dt.days)*24 + datetime.dt.hour
    df_forecast_weather['lat_lon'] = df_forecast_weather.apply(lambda row: \
                                                     f'{row["latitude"]:.1f}_{row["longitude"]:.1f}', axis=1)
    
    df_mean = df_forecast_weather.drop(['origin_datetime', 'forecast_datetime', 'hours_ahead', 'latitude', 'longitude', 'data_block_id', 'lat_lon'], axis=1) \
                                     .groupby(['time_id']).agg('mean').fillna(method='ffill').reset_index()
  
    df_forecast_weather = df_forecast_weather.loc[df_forecast_weather['lat_lon'].isin(dict_county.keys())]
    df_forecast_weather['county'] = df_forecast_weather['lat_lon'].map(lambda x: dict_county[x])

    df_features = df_forecast_weather.drop(['origin_datetime', 'forecast_datetime', 'hours_ahead', 'latitude', 'longitude', 'data_block_id', 'lat_lon'], axis=1) \
                                     .groupby(['time_id', 'county']).agg('mean').fillna(method='ffill').reset_index()
   

    return df_features.merge(df_mean, on='time_id', how='left')



def create_forecast_lag(df, lag):
    df_res = df.copy()
    df_res['time_id'] = df['time_id'] + lag
    df_res.columns = [f'{feature}_{lag}' if not feature in ['time_id', 'county'] else feature for feature in df_res.columns]
    return df_res


def create_target_lag(df, lag):
    df_target = df[['time_id', 'is_consumption', 'is_business', 'product_type', 'county', 'target']]
    df_target['time_id'] = df['time_id'] + lag
    df_target = df_target.rename(columns={'target': f'target_{lag}'})
    return df_target


def get_features(df_train, df_client, df_gas_prices, df_electricity_prices, df_forecast_weather, df_historical_weather, dict_county, df_targets=None):

    gc.collect()
    
    df_client = client_features(df_client)
    df_gas = gas_features(df_gas_prices)
    df_el = electricity_features(df_electricity_prices)

    datetime = pd.to_datetime(df_train['datetime'])
    df_train['time_id'] = (datetime - TIME_START).dt.days*24 + datetime.dt.hour
    
    df_temp = historical_weather_features(df_historical_weather, dict_county)
    df_fore = forecast_weather_features(df_forecast_weather, dict_county)
    df_fore_lag_neg1 = create_forecast_lag(df_fore, -1)
    df_fore_lag_1 = create_forecast_lag(df_fore, 1)
    
    df_time = time_features(datetime.drop_duplicates())    

    if df_targets is not None:
        datetime_targets = pd.to_datetime(df_targets['datetime'])
        df_targets['time_id'] = (datetime_targets - TIME_START).dt.days*24 + datetime_targets.dt.hour
    else:
        df_targets = df_train

    df_target_48 = create_target_lag(df_targets, 48)
    df_target_72 = create_target_lag(df_targets, 72)
    df_target_49 = create_target_lag(df_targets, 49)
    df_target_50 = create_target_lag(df_targets, 50)
    df_target_96 = create_target_lag(df_targets, 96)
    df_target_120 = create_target_lag(df_targets, 120)
    df_target_144 = create_target_lag(df_targets, 144)
    df_target_168 = create_target_lag(df_targets, 168)
    df_target_336 = create_target_lag(df_targets, 336)
    df_target_504 = create_target_lag(df_targets, 504)

    df = df_train[['time_id', 'data_block_id', 'is_consumption', 'is_business', 'product_type', 'county']] \
                .merge(df_client[['product_type', 'county', 'is_business', 'data_block_id', 'eic_count', 'installed_capacity']], 
                       on=['product_type', 'county', 'is_business', 'data_block_id'], how='left') \
                .merge(df_target_48, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_168, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_336, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_504, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_49, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_50, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_72, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_96, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_120, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_target_144, on=['time_id', 'is_consumption', 'is_business', 'product_type', 'county'], how='left') \
                .merge(df_temp, on=['time_id', 'county'], how='left') \
                .merge(df_fore, on=['time_id', 'county'], how='left') \
                .merge(df_fore_lag_neg1, on=['time_id', 'county'], how='left') \
                .merge(df_time, on='time_id', how='left') \
                .merge(df_el[['euros_per_mwh', 'time_id']], on='time_id', how='left') \
                .merge(df_gas[['data_block_id', 'lowest_price_per_mwh', 'highest_price_per_mwh']], on='data_block_id', how='left') \
                .merge(df_fore_lag_1, on=['time_id', 'county'], how='left') \

    df['target_ratio'] = df['target_168'] / (df['target_336'] + .0001)
    
    if 'target' in df_train.columns:
        df['target'] = df_train['target']
    
    return df.loc[df['time_id'] >= 48]