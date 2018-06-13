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

@functions.udf
def keep_time(date):
    """
    Keep only the time and not the day for a date
    """
    return date.split(' ')[1]

def rush_inter(date):
    """
    Create time intervals, in order to group the trip in the same schedule.
    We decided to create two buckets of time, during the rush hours 
    (at the start and end of the work day : 6h/9h and 17h/19h)
    and the rest of the day.
    """
    if (date >= '06:00:00' and date <= '09:00:00') or (date >= '17:00:00' and date <= '19:00:00'):
        return 0
    else:
        return 1
    
create_rush = udf(rush_inter)
    
@functions.udf
def create_interval(date):
    """
    Create more specific time intervals, separate the day in 6 time buckets.
    """
    if (date >= '00:00:00' and date < '06:00:00'):
        return 0
    elif (date >= '06:00:00' and date < '09:00:00'):
        return 1
    elif (date >= '09:00:00' and date < '13:00:00'):
        return 2
    elif (date >= '13:00:00' and date < '16:00:00'):
        return 3
    elif (date >= '16:00:00' and date < '19:00:00'):
        return 4
    else:
        return 5
    
def group_weekday_py(weekday):
    """
    Separate the day of the week in 4 buckets:
    - Wednesnay, Saturday and Sunday separatly
    - Monday, Tuesday, Thursday and Friday together
    """
    if weekday == 2:
        return 1
    elif weekday == 5:
        return 2
    elif weekday == 6:
        return 3
    else:
        return 0
    
group_weekday = functions.udf(group_weekday_py)
    
@functions.udf
def delete_neg(distri):
    """
    Remplace negative values by 0
    """
    return [0 if d < 0 else d for d in distri]

@functions.udf
def iqm(distri):
    """
    Compute the interquartile mean (IQM)
    """
    s = len(distri)
    lp = s/4 + 1
    lm = 3*s/4
    f = 0
    for i,x in enumerate(sorted(distri)):
        if lp <= i <= lm:
            f += x
    return 2/s * f

@functions.udf
def interquartile(distri):
    """
    Compute the interquartile range (IQR)
    """
    q1, q3 = np.percentile(distri, [25, 75])
    return float(q3) - float(q1)