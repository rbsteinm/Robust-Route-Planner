from math import sqrt, cos, radians, asin
import pyspark.sql.functions as functions
from datetime import datetime

def distance(long1, lat1, long2, lat2):
    """
    Compute the distance in kms between two locations
    given their coordinates (longitude, latitude)
    """
    # convert decimal degrees to radians 
    long1, long2, lat1, lat2 = [radians(x) for x in [long1, long2, lat1, lat2]]
    
    r = 6371 # earth radius
    # haversine formula
    return 2*r*asin(sqrt(hav(lat2-lat1)+cos(lat1)*cos(lat2)*hav(long2-long1)))

def hav(x):
    """haversine function """
    return (1-cos(x))/2

# models the network for the day given in parameter
# returns a dict of edge like: stopA -> (stopB,departure_time,arrival_time)
def model_network(df, date):
    df2 = df.filter(df.date == date)
    df2 = df2.select('stop_id', 'next_sid', 'schedule_dep', 'next_sched_arr')
    df2 = df2.withColumnRenamed('next_schedule_arr', 'schedule_arr')
    rows = df2.collect()
    edges = dict()
    for row in rows:
        if not row[0] in edges:
            edges[row[0]] = []
        edges[row[0]].append((row[1], row[2], row[3]))
    return edges
    print(df2.count())
    

    
#######################################################################
#__________________________ SPARK FUNCTIONS ___________________________
#######################################################################


# returns from the date the weekday as an integer
# monday = 0, sunday = 6
@functions.udf
def get_weekday(date):
    return datetime.strptime(date, '%Y-%m-%d %H:%M:%S').weekday()

@functions.udf
def date_choice(arr_time, dep_time):
    '''
    the departure time is null if it's the last stop of the trip
    returns a column that contains the departure time, or the arrival time if it's null
    this will serve to reconstruct the network from the trip's schedules
    '''
    return arr_time if dep_time is None else dep_time

@functions.udf
def stop_type(schedule_departure, schedule_arrival):
    '''
    create a column that tells if a stop is the first/last one of its trip or in the middle
    '''
    if schedule_departure is None:
        return 'last'
    elif schedule_arrival is None:
        return 'first'
    else:
        return 'mid'
    
@functions.udf
def edge_is_valid(tid, time, sid, stop_type, next_tid, next_time, next_sid, next_stop_type):
    return (time <= next_time and tid == next_tid and stop_type!='last' and next_stop_type!='first')