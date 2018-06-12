from math import sqrt, cos, radians, asin
from datetime import datetime, timedelta
import numpy as np
import pickle

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

def load_metadata():
    with open('./data/metadata.pickle', 'rb') as handle:
        stations = pickle.load(handle)
    return stations

def reduce_path(path):
    res = []
    previous_edge = path[0]
    first_edge = path[0]
    #first_stop, first_stop_dep = path[0][0], path[0][2]
    length = 0 # number of hops of this subpath
    for (i, edge) in enumerate(path):
        length += 1
        # if we change bus or it's the last hop of the path
        if edge[5] != previous_edge[5] or i+1 == len(path):
            res.append((first_edge[0], previous_edge[1], first_edge[2],previous_edge[3], previous_edge[5], length))
            length = 0
            first_edge = edge
        previous_edge = edge
    return res

def reduced_path_tostring(red_path, stations):
    for path in red_path:
        stop_a = stations[path[0]]['name']
        stop_b = stations[path[1]]['name']
        time_a = path[2].strftime('%H:%M')
        time_b = path[3].strftime('%H:%M')
        line = path[4]
        text = 'line ' + str(line) + ' from ' + stop_a + ' to ' + stop_b + ' '
        text += time_a + ' -> ' + time_b + '(' + str(path[5]) + ' stops)'
        print(text)


def reduced_path_to_json(l):
    d = {}
    for i in range(len(l)):
        subpath = {
            'source': str(l[i][0]),
            'dest': str(l[i][1]),
            'dep_time': l[i][2].strftime("%Y-%m-%d %H:%M"),
            'arr_time': l[i][3].strftime("%Y-%m-%d %H:%M"),
            'line': l[i][4],
            'nhops': str(l[i][5]),
        }
        d[i] = subpath
    return d
            
            
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

def load_networks():
    '''
    loads the train network for each day of the week
    '''
    models = []
    days_names = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    for day in days_names:
        with open('./data/'+ day +'.pickle', 'rb') as handle:
            network = pickle.load(handle)
        models.append(network)
    return models

def compute_walking_time(distance, walking_speed=5.04):
    """distance in km, speed in km/h, returns time in datetime format"""
    return timedelta(hours=(distance / walking_speed))

def compute_walking_network(stations, max_walk_dist=0.5):
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
    
def get_next_correspondance(edges, current_time, walking_network, source, dest, prev_tid):
    """
    returns departure/arrival times of the first ride departing after the current time
    assumes that the list current_time is sorted according to departure times!
    prev_tid: trip_id of the edge that led to source. Allows us to check if there is a change of bus at dest,
    in which case we will add one minute to the current time to take changing time into account
    """
    tid, line = None,None
    # if the edge exists for the rides
    if source in edges and dest in edges[source]:
        times = edges[source][dest] # list of schedules form source to destinations
        # index of the fist ride departing after the current time
        index = np.searchsorted([x[0] for x in times], current_time)
        # None if there is no more ride at this time
        (dep,arr,tid,line) = times[index] if index < len(times) else (None,None,None,None)
        # If you have less than one minute for a change of vehicule, that's not acceptable
        # add 60 second to the current time and seach again for the next correspondance
        if tid is not None and prev_tid!=tid and current_time==dep:
            index = np.searchsorted([x[0] for x in times], current_time+timedelta(seconds=60))
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
            # reconstruct the shortest path
            while prev[u][0] != source:
                assert(prev[u][0] is not None),'no path from ' + stations[source].name + ' to ' + stations[destination].name
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
            (dep_time, arr_time,tid,line) = get_next_correspondance(edges, current_time, walking_network, u, v, prev[u][3])
            # there is no more correspondance for this edge
            if dep_time is None:
                continue
            dist_u_v = arr_time - dep_time # travelling time
            waiting_time = dep_time - current_time # waiting time at the station
            arr_v = current_time + waiting_time + dist_u_v # determine at what time you'll arrive to v
            # a shorter path to v has been found
            if arr_v < dist[v]:
                dist[v] = arr_v
                prev[v] = (u,dep_time,arr_time,tid,line)
            
    raise Exception('No path was found from source to destination')
    

#######################################################################
#______________________ BFS (REACHABLE STATIONS) ______________________
#######################################################################

    
def get_neighbors(network, walking_network, station_id):
    '''
    return all the direct neighbors of station_id, either by bus or foot
    does not take the schedules into account!
    '''
    neighbors = set()
    if station_id in network:
        for neigh in network[station_id].keys():
            neighbors.add(neigh)
    if station_id in walking_network:
        for neigh in walking_network[station_id].keys():
            neighbors.add(neigh)
    return neighbors

def get_reachable_stations(network, walking_network, source):
    '''
    returns all the stations reachable from source, either by foot or bus
    does not take the schedules into account!
    '''
    visited, queue = set(), [source]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            neighbors = get_neighbors(network, walking_network, node)
            queue.extend(neighbors - visited)
    return visited