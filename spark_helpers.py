import pyspark.sql.functions as functions
from datetime import datetime
    
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
    returns a column that contains an average between the arrival and departure times at each stop
    this will serve to order the stops by time, in order toreconstruct the network from the trip's schedules
    '''
    if arr_time is None:
        return dep_time
    elif dep_time is None:
        return arr_time
    else:
        arr_ts = datetime.strptime(arr_time, '%Y-%m-%d %H:%M:%S').timestamp()
        dep_ts = datetime.strptime(dep_time, '%Y-%m-%d %H:%M:%S').timestamp()
        mean = (dep_ts+arr_ts)/2
        return datetime.fromtimestamp(mean).strftime('%Y-%m-%d %H:%M:%S')

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
def edge_is_valid(tid, time, sid, stop_type, next_tid, next_time, next_sid, next_stop_type, dep, next_arr):
    # sometimes, the last and first stop types are not correct due to malformated data
    # to fix this issue, we consider that if a trip is longer than 10 hours, it can be discarded
    duration = (datetime.strptime(next_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(time, '%Y-%m-%d %H:%M:%S')).total_seconds()/3600
    return (time <= next_time and tid == next_tid and stop_type!='last' and next_stop_type!='first' and duration < 10 and dep <= next_arr)
