# Import libraries and dependencies
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import geoglows
import numpy as np
import math
import datetime
import warnings
warnings.filterwarnings('ignore') 

# Change the work directory
user = os.getlogin()
user_dir = os.path.expanduser('~{}'.format(user))
os.chdir(user_dir)
os.chdir("tethys_apps_peru/geoglows_database_peru")

# Import enviromental variables
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')

# Generate the conection token
token = "postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER, DB_PASS, DB_NAME)




###############################################################################################################
#                                 Function to get and format the data from DB                                 #
###############################################################################################################
def get_format_data(sql_statement, conn):
    # Retrieve data from database
    data =  pd.read_sql(sql_statement, conn)
    # Datetime column as dataframe index
    data.index = data.datetime
    data = data.drop(columns=['datetime'])
    # Format the index values
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    data.index = pd.to_datetime(data.index)
    # Return result
    return(data)

def get_format_data_level(sql_statement, conn):
    # Retrieve data from database
    data =  pd.read_sql(sql_statement, conn)
    # Datetime column as dataframe index
    data.index = data.datetime
    data = data.drop(columns=['datetime'])
    # Format the index values
    data[data < 0.1] = 0.1
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    data.index = pd.to_datetime(data.index)
    # Return result
    return(data)


###############################################################################################################
#                                         Function to bias correction                                         #
###############################################################################################################
def get_bias_corrected_data(sim, obs):
    outdf = geoglows.bias.correct_historical(sim, obs)
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


###############################################################################################################
#                                   Getting return periods from data series                                   #
###############################################################################################################
def gumbel_1(std: float, xbar: float, rp: int or float) -> float:
  return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)

def get_return_periods(comid, data):
    # Stats
    max_annual_flow = data.groupby(data.index.strftime("%Y")).max()
    mean_value = np.mean(max_annual_flow.iloc[:,0].values)
    std_value = np.std(max_annual_flow.iloc[:,0].values)
    # Return periods
    return_periods = [100, 50, 25, 10, 5, 2]
    return_periods_values = []
    # Compute the corrected return periods
    for rp in return_periods:
      return_periods_values.append(gumbel_1(std_value, mean_value, rp))
    # Parse to list
    d = {'rivid': [comid], 
         'return_period_100': [return_periods_values[0]], 
         'return_period_50': [return_periods_values[1]], 
         'return_period_25': [return_periods_values[2]], 
         'return_period_10': [return_periods_values[3]], 
         'return_period_5': [return_periods_values[4]], 
         'return_period_2': [return_periods_values[5]]}
    # Parse to dataframe
    corrected_rperiods_df = pd.DataFrame(data=d)
    corrected_rperiods_df.set_index('rivid', inplace=True)
    return(corrected_rperiods_df)


###############################################################################################################
#                                         Getting ensemble statistic                                          #
###############################################################################################################
def ensemble_quantile(ensemble, quantile, label):
    df = ensemble.quantile(quantile, axis=1).to_frame()
    df.rename(columns = {quantile: label}, inplace = True)
    return(df)

def get_ensemble_stats(ensemble):
    high_res_df = ensemble['ensemble_52_m^3/s'].to_frame()
    ensemble.drop(columns=['ensemble_52_m^3/s'], inplace=True)
    ensemble.dropna(inplace= True)
    high_res_df.dropna(inplace= True)
    high_res_df.rename(columns = {'ensemble_52_m^3/s':'high_res_m^3/s'}, inplace = True)
    stats_df = pd.concat([
        ensemble_quantile(ensemble, 1.00, 'flow_max_m^3/s'),
        ensemble_quantile(ensemble, 0.75, 'flow_75%_m^3/s'),
        ensemble_quantile(ensemble, 0.50, 'flow_avg_m^3/s'),
        ensemble_quantile(ensemble, 0.25, 'flow_25%_m^3/s'),
        ensemble_quantile(ensemble, 0.00, 'flow_min_m^3/s'),
        high_res_df
    ], axis=1)
    return(stats_df)


###############################################################################################################
#                                    Warning if exceed x return period                                        #
###############################################################################################################
def is_warning(arr):
    cond = [i >= 40 for i in arr].count(True) > 0
    return(cond)

def get_excced_rp(stats: pd.DataFrame, ensem: pd.DataFrame, rperiods: pd.DataFrame):
    dates = stats.index.tolist()
    startdate = dates[0]
    enddate = dates[-1]
    span = enddate - startdate
    uniqueday = [startdate + datetime.timedelta(days=i) for i in range(span.days + 2)]
    # get the return periods for the stream reach
    rp2 = rperiods['return_period_2'].values
    rp5 = rperiods['return_period_5'].values
    rp10 = rperiods['return_period_10'].values
    rp25 = rperiods['return_period_25'].values
    rp50 = rperiods['return_period_50'].values
    rp100 = rperiods['return_period_100'].values
    # fill the lists of things used as context in rendering the template
    days = []
    r2 = []
    r5 = []
    r10 = []
    r25 = []
    r50 = []
    r100 = []
    for i in range(len(uniqueday) - 1):  # (-1) omit the extra day used for reference only
        tmp = ensem.loc[uniqueday[i]:uniqueday[i + 1]]
        days.append(uniqueday[i].strftime('%b %d'))
        num2 = 0
        num5 = 0
        num10 = 0
        num25 = 0
        num50 = 0
        num100 = 0
        for column in tmp:
            column_max = tmp[column].to_numpy().max()
            if column_max > rp100:
                num100 += 1
            if column_max > rp50:
                num50 += 1
            if column_max > rp25:
                num25 += 1
            if column_max > rp10:
                num10 += 1
            if column_max > rp5:
                num5 += 1
            if column_max > rp2:
                num2 += 1
        r2.append(round(num2 * 100 / 52))
        r5.append(round(num5 * 100 / 52))
        r10.append(round(num10 * 100 / 52))
        r25.append(round(num25 * 100 / 52))
        r50.append(round(num50 * 100 / 52))
        r100.append(round(num100 * 100 / 52))
    alarm = "R0"
    if(is_warning(r2)):
        alarm = "R2"
    if(is_warning(r5)):
        alarm = "R5"
    if(is_warning(r10)):
        alarm = "R10"
    if(is_warning(r25)):
        alarm = "R25"
    if(is_warning(r50)):
        alarm = "R50"
    if(is_warning(r100)):
        alarm = "R100"
    return(alarm)


def get_corrected_forecast(simulated_df, ensemble_df, observed_df):
    monthly_simulated = simulated_df[simulated_df.index.month == (ensemble_df.index[0]).month].dropna()
    monthly_observed = observed_df[observed_df.index.month == (ensemble_df.index[0]).month].dropna()
    min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
    max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())
    min_factor_df = ensemble_df.copy()
    max_factor_df = ensemble_df.copy()
    forecast_ens_df = ensemble_df.copy()
    for column in ensemble_df.columns:
      tmp = ensemble_df[column].dropna().to_frame()
      min_factor = tmp.copy()
      max_factor = tmp.copy()
      min_factor.loc[min_factor[column] >= min_simulated, column] = 1
      min_index_value = min_factor[min_factor[column] != 1].index.tolist()
      for element in min_index_value:
        min_factor[column].loc[min_factor.index == element] = tmp[column].loc[tmp.index == element] / min_simulated
      max_factor.loc[max_factor[column] <= max_simulated, column] = 1
      max_index_value = max_factor[max_factor[column] != 1].index.tolist()
      for element in max_index_value:
        max_factor[column].loc[max_factor.index == element] = tmp[column].loc[tmp.index == element] / max_simulated
      tmp.loc[tmp[column] <= min_simulated, column] = min_simulated
      tmp.loc[tmp[column] >= max_simulated, column] = max_simulated
      forecast_ens_df.update(pd.DataFrame(tmp[column].values, index=tmp.index, columns=[column]))
      min_factor_df.update(pd.DataFrame(min_factor[column].values, index=min_factor.index, columns=[column]))
      max_factor_df.update(pd.DataFrame(max_factor[column].values, index=max_factor.index, columns=[column]))
    corrected_ensembles = geoglows.bias.correct_forecast(forecast_ens_df, simulated_df, observed_df)
    corrected_ensembles = corrected_ensembles.multiply(min_factor_df, axis=0)
    corrected_ensembles = corrected_ensembles.multiply(max_factor_df, axis=0)
    return(corrected_ensembles)



###############################################################################################################
#                                                 Streamflow                                                  #
###############################################################################################################

# Setting the connetion to db
db = create_engine(token)

# Establish connection
conn = db.connect()

# Getting stations
stations = pd.read_sql("select * from streamflow_station;", conn)

# Number of stations 
n = len(stations)

# For loop
for i in range(n):
    print("{0} - {1}".format(i, stations.code[i]))
    # State variables  
    station_code = stations.code[i]
    station_comid = stations.comid[i]
    try:
        # Query to database
        observed_data = get_format_data("select distinct * from sf_{0} order by datetime;".format(station_code), conn)
        simulated_data = get_format_data("select * from r_{0};".format(station_comid), conn)
        ensemble_forecast = get_format_data("select * from f_{0};".format(station_comid), conn)
        # Corect the historical simulation
        corrected_data = get_bias_corrected_data(simulated_data, observed_data)
        # Return period
        return_periods = get_return_periods(station_comid, corrected_data)
        # Corrected Forecast
        ensemble_forecast = get_corrected_forecast(simulated_data, ensemble_forecast, observed_data)
        # Forecast stats
        ensemble_stats = get_ensemble_stats(ensemble_forecast)
        # Warning if excced a given return period in 10% of emsemble
        stations.loc[i, ['alert']] = get_excced_rp(ensemble_stats, ensemble_forecast, return_periods)
    except Exception as err:
        print(err)


# Insert to database
stations.to_sql('streamflow_station', con=conn, if_exists='replace', index=False)

# Close connection
conn.close()





###############################################################################################################
#                                                Water Level                                                  #
###############################################################################################################

# Setting the connetion to db
db = create_engine(token)

# Establish connection
conn = db.connect()

# Getting stations
stations = pd.read_sql("select * from waterlevel_station;", conn)

# Number of stations
n = len(stations)

# For loop
for i in range(n):
    print("{0} - {1}".format(i, stations.code[i]))
    # State variables
    station_code = stations.code[i].lower()
    station_comid = stations.comid[i]
    try:
        # Query to database
        observed_data = get_format_data_level("select distinct * from wl_{0} order by datetime;".format(station_code), conn)
        simulated_data = get_format_data_level("select * from r_{0};".format(station_comid), conn)
        ensemble_forecast = get_format_data_level("select * from f_{0};".format(station_comid), conn)
        # Corect the historical simulation
        corrected_data = get_bias_corrected_data(simulated_data, observed_data)
        # Return period
        return_periods = get_return_periods(station_comid, corrected_data)
        # Corrected Forecast
        ensemble_forecast = get_corrected_forecast(simulated_data, ensemble_forecast, observed_data)
        # Forecast stats
        ensemble_stats = get_ensemble_stats(ensemble_forecast)
        # Warning if excced a given return period in 10% of emsemble
        stations.loc[i, ['alert']] = get_excced_rp(ensemble_stats, ensemble_forecast, return_periods)
    except Exception as err:
        print(err)


# Insert to database
stations.to_sql('waterlevel_station', con=conn, if_exists='replace', index=False)

# Close connection
conn.close()


