# Import libraries and dependencies
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

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


# Function to retrieve data from GESS API
def get_data(comid):
    url = 'https://geoglows.ecmwf.int/api/HistoricSimulation/?reach_id={0}&return_format=csv'.format(comid)
    status = False
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0) 
        if(outdf.shape[0]>0):
           print(outdf.shape)
           status = True
        else:
           raise ValueError("Dataframe has not data.")
      except:
        print("Trying to retrieve data...")
    # Filter and correct data
    outdf[outdf < 0] = 0
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    print("Returning data...")
    return(outdf)


# Function to insert data to database
def insert_data(db, comid):
    # Get historical data
    historical = get_data(comid)
    # Establish connection
    conn = db.connect()
    # Define the table and delete if exist
    table = 'r_{0}'.format(comid)
    conn.execute("DROP TABLE IF EXISTS {0};".format(table))
    # Insert to database
    try:
        historical.to_sql(table, con=conn, if_exists='replace', index=True)
        print("Successfully inserted data...")
    except:
       print("Error to insert data in comid={0}".format(comid))
    # Close conection
    conn.close()   


# Read comids
data = pd.read_excel('peru_geoglows_drainage.xlsx', index_col=0)

# Setting the connetion to db
db= create_engine(token)


n = len(data.comid) + 1

for i in range(1340,n):
    # State variable
    comid = data.comid[i]
    # Progress
    prog = round(100 * i/n, 3)
    print("Progress: {0} %. Comid: {1}".format(prog, comid))
    try:
        insert_data(db, comid)
    except:
        insert_data(db, comid)







