from hs_restclient import HydroShare, HydroShareAuthBasic
from dotenv import load_dotenv
import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import numpy as np


# Change the work directory
user = os.getlogin()
user_dir = os.path.expanduser('~{}'.format(user))
os.chdir(user_dir)
os.chdir("tethys_apps_peru/geoglows_database_peru")


# Import enviromental variables
load_dotenv()
HS_USER = os.getenv('HS_USER') # Hydroshare username
HS_PASS = os.getenv('HS_PASS') # Hydroshare password
HS_REID = os.getenv('HS_REID') # Hydroshare resource ID
DB_USER = os.getenv('DB_USER') # Database username
DB_PASS = os.getenv('DB_PASS') # Database password
DB_NAME = os.getenv('DB_NAME') # Database name


# Authentication on hydroshare
auth = HydroShareAuthBasic(username = HS_USER, password = HS_PASS)
hs = HydroShare(auth=auth)


# Function to retrieve the observed data from station
def get_observed_data(code):
    # Open the resource
    hs.setAccessRules(HS_REID, public=True)
    # Download observed data
    url = 'https://www.hydroshare.org/resource/{0}/data/contents/Discharge_Data/{1}.csv'.format(HS_REID, code)
    df = pd.read_csv(url, index_col=0)
    # Close the resource
    hs.setAccessRules(HS_REID, public=False)
    # Formating the data
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    df = df.rename(columns={"Streamflow (m3/s)":"streamflow"})
    # Returning
    return(df)


# Generate the conection token for database
token = "postgresql+psycopg2://{0}:{1}@localhost:5432/{2}".format(DB_USER, DB_PASS, DB_NAME)
  
# Establish connection to database
db = create_engine(token)
conn = db.connect()

# Retrieve data from database
stations =  pd.read_sql("select code from streamflow_station;", conn)

# Error list
error_list = []

# Loop routine
n = len(stations.code)
for i in range(n):
    code = stations.code[i]
    # Progress
    prog = round(100 * i/n, 3)
    print("Progress: {0} %. Station: {1}".format(prog, code))
    try:
        # Download and insert into database
        observed_df = get_observed_data(code)
        table = 'sf_{0}'.format(code)
        conn.execute("DROP TABLE IF EXISTS {0};".format(table))
        observed_df.to_sql(table, con=conn, if_exists='replace', index=True)
    except:
        print("Error downloading data in station {0}".format(code))
        error_list = np.append(error_list, code)


# Close connection
conn.close()