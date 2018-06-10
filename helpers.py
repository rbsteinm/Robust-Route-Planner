from math import sqrt, cos, radians, asin
import pyspark.sql.functions as functions
from datetime import datetime, timedelta
import numpy as np

#######################################################################
#______________________________ HELPERS _______________________________
#######################################################################

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
    
# get the duration in seconds from t2 to t1
# t2 > t1
def get_duration(t1, t2):
    return (datetime.strptime(t2, '%Y-%m-%d %H:%M:%S') - datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')).total_seconds()

def str_to_date(string):
    return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

# converts the date strings to datetime objects
def network_to_datetime(network):
    for i in network.keys():
        for j in network[i].keys():
            network[i][j] = [(str_to_date(dep),str_to_date(arr),tid,line) for (dep,arr,tid,line) in network[i][j]]
            
            
#######################################################################
#___________________________ MODEL NETWORK ____________________________
#######################################################################

# models the network for the day given in parameter
# returns a dict of edge like: stopA -> stopB -> [(dep_time1,arr_time1, tid1, line1), (dep_time2,arr_time2, tid2, line2), ...]
def model_network(df, date):
    df2 = df.filter(df.date == date)
    df2 = df2.select('trip_id','line', 'stop_id', 'next_sid', 'schedule_dep', 'next_sched_arr')
    df2 = df2.withColumnRenamed('next_schedule_arr', 'schedule_arr')
    rows = df2.collect()
    edges = dict()
    for row in rows:
        if not row[2] in edges:
            edges[row[2]] = dict()
        if not row[3] in edges[row[2]]:
            edges[row[2]][row[3]] = []
        edges[row[2]][row[3]].append((row[4], row[5], row[0], row[1]))
        
    # sort the list according to departure times
    for stopA in edges:
        for stopB in edges[stopA]:
            edges[stopA][stopB].sort(key=lambda x: x[0])
    
    return edges

def compute_walking_time(distance, walking_speed=5.04):
    """distance in km, speed in km/h, returns time in datetime format"""
    return timedelta(hours=(distance / walking_speed))

def compute_walking_network(stations, max_walk_dist=1):
    """
    computes the time you need to walk between any two nodes in the network
    in the form {A -> B -> time} and {B -> A -> time}
    the data structure contains None if the distance to walk is > max_walk_dist
    """
    walking_network = {}
    # TODO instead of Nones just dont put B as a neigh of A if it's not reachable
    for i in stations.keys():
        walking_network[i] = {}
        for j in stations.keys():
            dist = distance(stations[i]['long'], stations[i]['lat'], stations[j]['long'], stations[j]['lat'])    
            if(i!=j and dist < max_walk_dist):
                time = compute_walking_time(dist)
                walking_network[i][j] = time
    return walking_network


#######################################################################
#___________________________ SHORTEST PATH ____________________________
#######################################################################
    
def get_next_correspondance(edges, current_time, walking_network, source, dest):
    """
    returns departure/arrival times of the first ride departing after the current time
    assumes that the list current_time is sorted according to departure times!
    """
    tid, line = None,None
    # if the edge exists for the rides
    if source in edges and dest in edges[source]:
        times = edges[source][dest]
        # index of the fist ride departing after the current time
        index = np.searchsorted([x[0] for x in times], current_time)
        # None if there is no more ride at this time
        (dep,arr,tid,line) = times[index] if index < len(times) else (None,None,None,None)
    else:
        (dep,arr) = (None,None) # None if there is no edge by vehicule
        
    # determine if you can reach dest from source by foot
    if source in walking_network and dest in walking_network[source]:
        walk_time = walking_network[source][dest]
        (dep_walk,arr_walk) = (current_time,current_time+walk_time)
    else:
        (dep_walk,arr_walk) = (None,None)
    
    if dep is None and dep_walk is None:
        return (None,None,None,None) # if you can neither walk nor take a transport
    elif dep is None:
        return (dep_walk,arr_walk,'walk','walk') # if you can only walk
    elif dep_walk is None:
        return (dep,arr,tid,line) # if you can only take a transport
    else:
        return (dep,arr,tid,line) if (arr<arr_walk) else (dep_walk,arr_walk,'walk','walk') # if you can walk or ride, do the fastest
    
def shortest_path(models, walking_network, stations, source, destination, departure_time):
    '''
    Compute the shortest path between source and destination using Dijksta's algorithm
    models: contains 7 networks, one for each day of the week. They can be generated using model_network()
    source, destination: station IDs
    '''
    edges = models[departure_time.weekday()] # get the network for the correct day of the week
    Q = set(stations.keys()) # deep copy
    dist = dict.fromkeys(Q, datetime.max) # distances to the source (= arrival time at each node)
    prev = dict.fromkeys(Q, (None, None, None, None, None)) # (previous node, dep/arr times, trip_id, line) in the shortest path
    dist[source] = departure_time
    
    while Q:
        unvisited_dist = {key: dist[key] for key in Q} # distances of unvisited nodes
        u = min(unvisited_dist, key=unvisited_dist.get) # u <- vertex in Q with minimum dist[u]
        
        if dist[u] == datetime.max:
            raise Exception('Only nodes with infinity distance in the queue. The graph is disconected')
        
        Q.remove(u) #remove u from Q
        
        # if this is the destination node, we can terminate
        if u == destination:
            path = []
            while prev[u][0] != source:
                assert(prev[u][0] is not None), 'Could not find a path from ' + str(stations[source].name) + ' to ' + str(stations[destination].name)
                assert(len(path) < 300), 'Path has more than 300 hops too long, something is wrong ...'
                current_edge = (prev[u][0],u,prev[u][1],prev[u][2],prev[u][3],prev[u][4])
                path.insert(0,current_edge)
                u = prev[u][0] # get previous node
            current_edge = (prev[u][0],u,prev[u][1],prev[u][2],prev[u][3],prev[u][4])
            path.insert(0,current_edge) # push the source at the beginning of the path
            return path
        
        current_time = dist[u]
        neighbors = set(edges[u].keys()) if u in edges else set() # u's neighbors by vehicule
        walk_neighbors = set(walking_network[u].keys())
        for v in neighbors.union(walk_neighbors):
            # take the first correspondance
            (dep_time, arr_time,tid,line) = get_next_correspondance(edges, current_time, walking_network, u, v)
            # there is no more correspondance for this edge
            if dep_time is None:
                continue
            dist_u_v = arr_time - dep_time
            waiting_time = dep_time - current_time
            arr_v = current_time + waiting_time + dist_u_v # determine at what time you'll arrive to v
            # a shorter path to v has been found
            if arr_v < dist[v]:
                dist[v] = arr_v
                prev[v] = (u,dep_time,arr_time,tid,line)
            
    
    raise Exception('No path was found from source to destination')
    
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
