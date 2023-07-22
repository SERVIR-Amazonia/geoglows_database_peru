# Import libraries and dependencies
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import datetime as dt

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
    date = dt.datetime.now().strftime('%Y%m%d')
    idate = dt.datetime.strptime(date, '%Y%m%d') - dt.timedelta(days=60)
    idate = idate.strftime('%Y%m%d')
    url = 'https://geoglows.ecmwf.int/api/ForecastRecords/?reach_id={0}&start_date={1}&end_date={2}&return_format=csv'.format(comid, idate, date)
    status = False
    while not status:
      try:
        outdf = pd.read_csv(url, index_col=0)
        status = True
      except:
        print("Trying to retrieve data...")
    # Filter and correct data
    outdf[outdf < 0] = 0
    outdf.index = pd.to_datetime(outdf.index)
    outdf.index = outdf.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
    outdf.index = pd.to_datetime(outdf.index)
    return(outdf)


# Function to insert data to database
def insert_data(db, comid):
    # Get historical data
    historical = get_data(comid)
    # Establish connection
    conn = db.connect()
    # Insert to database
    table = 'fr_{0}'.format(comid)
    try:
        historical.to_sql(table, con=conn, if_exists='replace', index=True)
    except:
       print("Error to insert data in comid={0}".format(comid))
    # Close conection
    conn.close()   



# Read comids
data = pd.read_excel('peru_geoglows_drainage.xlsx', index_col=0)

# Setting the connetion to db 
db= create_engine(token)


n = len(data.comid) + 1

for i in range(1,n):
    # State variable
    comid = data.comid[i]
    # Progress
    prog = round(100 * i/n, 3)
    print("Progress: {0} %. Comid: {1}".format(prog, comid))
    try:
        insert_data(db, comid)
    except:
        insert_data(db, comid)

