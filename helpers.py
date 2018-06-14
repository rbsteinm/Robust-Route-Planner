from math import sqrt, cos, radians, asin
from datetime import datetime, timedelta
from pyproj import Proj, transform
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models.markers import Square
from bokeh.models.glyphs import Patches
from bokeh.palettes import Set3
from bokeh.models import HoverTool
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import time
import numpy as np
import pickle
import functools
import copy

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
    all_rides = []
    current_ride = []
    i = 0
    prev_line = None
    while(i < len(path)):
        current_line = path[i][5]
        if current_line != prev_line:
            all_rides.append(current_ride)
            current_ride = [path[i]]
        else:
            current_ride.append(path[i])
        prev_line = current_line
        i += 1
    # append last ride
    all_rides.append(current_ride)
    return [(ride[0][0],ride[-1][1], ride[0][2], ride[-1][3], ride[0][5], len(ride)) for ride in all_rides if len(ride) > 0]
            

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

def to_typical_day(date):
    '''
    puts any date in the range 15.01.18-21.01.18 according to its weekday
    this allows to run the shortest path alg. with correct schedules
    '''
    typical_week = ['2018-01-' + str(x) + ' ' + date.strftime("%H:%M:%S") for x in range(15, 22)]
    typical_week = [str_to_date(x) for x in typical_week]
    return typical_week[date.weekday()]

def edge_to_typical_day(e):
    return (e[0],e[1], to_typical_day(e[2]), to_typical_day(e[3]),e[4],e[5])

def back_to_original_date(path, date):
    '''
    changes the day and month of each datetime object in the path
    they were changed previously for the shortest path, this function puts them back
    to their original values
    date: the original date
    '''
    month = date.month
    day = date.day
    res_path = []
    for p in path:
        res_path.append((p[0],p[1],p[2].replace(month=month,day=day),p[3].replace(month=month,day=day),p[4],p[5]))
    return res_path

def compute_path_time(path):
    t1 = path[0][2]
    t2 = path[-1][3]
    return t2 - t1
            
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
    original_date = departure_time
    departure_time = to_typical_day(departure_time)
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
            return back_to_original_date(path, original_date)
        
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
#_______________________ REVERSE SHORTEST PATH ________________________
#######################################################################

def get_next_correspondance_reverse(edges, current_time, walking_network, source, dest, prev_tid):
    """
    returns departure/arrival times of the first ride departing after the current time
    assumes that the list current_time is sorted according to departure times!
    prev_tid: trip_id of the edge that led to source. Allows us to check if there is a change of bus at dest,
    in which case we will add one minute to the current time to take changing time into account
    
    Here since we are in reverse:
    - we follow the reverse edge from u to v
    - original edges goes v -> u, or dest -> source
    - source==u and dest==v
    - dep is the time at which you leave v and arr is the time at which you arrive at u
    """
    tid, line = None,None
    # if the edge exists for the rides (here source is v and dest is u)
    if source in edges and dest in edges[source]:
        times = edges[source][dest] # list of schedules form source to destinations
        # index of the fist ride arriving before the current time
        index = np.searchsorted([x[1] for x in times], current_time) - 1
        # None if there is no ride that can make you arrive at this time
        (dep,arr,tid,line) = times[index] if index >= 0 else (None,None,None,None)
        # If you have less than one minute for a change of vehicule, that's not acceptable
        # add 60 second to the current time and seach again for the next correspondance
        if tid is not None and prev_tid!=tid and current_time==arr:
            index = np.searchsorted([x[1] for x in times], current_time-timedelta(seconds=60)) - 1
            (dep,arr,tid,line) = times[index] if index >= 0 else (None,None,None,None)
    else:
        (dep,arr) = (None,None) # None if there is no edge by vehicule
        
    # determine if you can reach dest from source by foot
    if source in walking_network and dest in walking_network[source]:
        walk_time = walking_network[source][dest]
        (dep_walk,arr_walk) = (current_time-walk_time,current_time)
    else:
        (dep_walk,arr_walk) = (None,None)
    
    if dep is None and dep_walk is None:
        return (None,None,None,None) # if you can neither walk nor take a transport
    elif dep is None:
        return (dep_walk,arr_walk,'walk','walk') # if you can only walk
    elif dep_walk is None:
        return (dep,arr,tid,line) # if you can only take a transport
    else:
        return (dep,arr,tid,line) if (dep>=dep_walk) else (dep_walk,arr_walk,'walk','walk') # if you can walk or ride, do the fastest
    
def shortest_path_reverse(models, walking_network, stations, source, destination, arrival_time):
    '''
    Compute the shortest path between destination and source using Dijksta's algorithm
    models: contains 7 networks, one for each day of the week. They can be generated using model_network()
    source, destination: station IDs
    '''
    original_date = arrival_time
    arrival_time = to_typical_day(arrival_time)
    edges = build_reverse_network(models[arrival_time.weekday()]) # get the network for the correct day of the week
    Q = set(stations.keys()) # deep copy
    dist = dict.fromkeys(Q, datetime.min) # distances to the source (= departure time at each node)
    prev = dict.fromkeys(Q, (None, None, None, None, None)) # (previous node, dep/arr times, trip_id, line) in the shortest path
    dist[destination] = arrival_time
    
    while Q:
        unvisited_dist = {key: dist[key] for key in Q} # distances of unvisited nodes
        u = max(unvisited_dist, key=unvisited_dist.get) # u <- vertex in Q with maximum dist[u]
        #print('current node ',u)
        
        if dist[u] == datetime.min:
            raise Exception('Only nodes with infinity distance in the queue. The graph is disconected')
        
        Q.remove(u) #remove u from Q
        
        # if this is the source node, we can terminate
        if u == source:
            path = []
            # reconstruct the shortest path
            while prev[u][0] != destination:
                assert(prev[u][0] is not None),'no path from ' + stations[source].name + ' to ' + stations[destination].name
                assert(len(path) < 300), 'Path has more than 300 hops, something is wrong ...'
                current_edge = (u,prev[u][0],prev[u][1],prev[u][2],prev[u][3],prev[u][4])
                path.append(current_edge)
                u = prev[u][0] # get previous node
            current_edge = (u,prev[u][0],prev[u][1],prev[u][2],prev[u][3],prev[u][4])
            path.append(current_edge)
            return back_to_original_date(path, original_date)
        
        current_time = dist[u]
        neighbors = set(edges[u].keys()) if u in edges else set() # u's neighbors by vehicule
        walk_neighbors = set(walking_network[u].keys())
        for v in neighbors.union(walk_neighbors):
            # take the first correspondance. dep is the departure time from v and arr the arrival time at u
            (dep_time,arr_time,tid,line) = get_next_correspondance_reverse(edges, current_time, walking_network, u, v, prev[u][3])
            # there is no more correspondance for this edge
            if dep_time is None:
                continue
            dist_u_v = arr_time - dep_time # travelling time from v to u
            waiting_time = current_time -  arr_time # waiting time after your arrival at u
            dep_v = current_time - waiting_time - dist_u_v # determine at what time you left v for u
            # a shorter path to v has been found
            if dep_v > dist[v]:
                dist[v] = dep_v
                prev[v] = (u,dep_time,arr_time,tid,line)
            
    raise Exception('No path was found from source to destination')
    
def build_reverse_network(network):
    '''
    reverts all the edges in the network
    '''
    reverse_net = dict()
    for source in network.keys():
        for dest in network[source].keys():
            if not dest in reverse_net.keys():
                reverse_net[dest] = dict()
            reverse_net[dest][source] = network[source][dest]
    return reverse_net
    

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


#######################################################################
#______________________ UNCERTAINTY PROBABILITY  ______________________
#######################################################################

def search_inter(trip_id, weekday, time_interval, delay_distribution_pd):
    """
    search the worst case delay for a given trip_id / weekday / time interval
    """
    interquartile = delay_distribution_pd[(delay_distribution_pd.trip_id == str(trip_id)) & 
                            (delay_distribution_pd.arrival_interval == str(time_interval)) & 
                            (delay_distribution_pd.weekday == str(weekday))].worst_case
    return float(interquartile) if interquartile.size != 0 else float(0)


def routing_algo(path, delay_distribution_pd):
    """
    Take a path and determine the rate of missing each transport change according to the predictive model
    we built earlier. It returns also the combining probability for the whole path.
    """
    prev_edge = path[0]
    certainty = {}
    for i, edge in enumerate(path):
        
        if (edge[4] != prev_edge[4]) and (edge[4] != 'walk') and ((prev_edge[4] != 'walk') or (i-1 != 0)):
            
            time_for_change = edge[2] - prev_edge[3]
            
            if (prev_edge[4] == 'walk') and (i-1 != 0):      
                time_inter = rush_inter(path[i-2][3].time().isoformat())
                inter = search_inter(path[i-2][4], group_weekday_py(path[i-2][3].weekday()), time_inter, delay_distribution_pd)
            else:
                time_inter = rush_inter(prev_edge[3].time().isoformat())
                inter = search_inter(prev_edge[4], group_weekday_py(edge[3].weekday()), time_inter, delay_distribution_pd)
            
            if inter:
                certainty[i] = min(time_for_change.total_seconds() / inter, 1)
            else:
                certainty[i] = 1
        prev_edge = edge
        
        if certainty:
            certainty_tot = functools.reduce((lambda v1,v2: v1*v2), certainty.values())
        else:
            certainty_tot = 1
            
    return certainty, certainty_tot


#######################################################################
#_____________________ GENERATE ALTERNATIVE PATHS  ____________________
#######################################################################


def remove_worst_edge(network, uncertainies, path):
    '''
    removes the worst edge from the network and returns it
    returns None if the network is already good enough (total uncertainty >= threshold)
    '''
    if uncertainies[1] == 1:
        return network
    index = min(uncertainies[0], key=uncertainies[0].get)
    edge = path[index]
    return remove_edge(edge, network)
    
    
def remove_edge(e, net):
    '''
    returns a deep copy of the network and removes the edge e.
    Warning, this methods removes an edge INPLACE
    '''
    e = edge_to_typical_day(e)
    net[e[0]][e[1]].remove(e[2:])
    return net


def safest_paths(models, walking_network, stations, source, dest, date, delay_distribution, n_iters=100, threshold=0.8, n_paths=4):
    '''
    tries to compute new shortest paths by iteratively removing the most risky edge of the network
    and recomputing a shortest path from the reduced network.
    Note that we do not remove an entire edge, but just a single row of this edge's schedule.
    '''
    # do a deepcopy of the network we need
    my_net = copy.deepcopy(models[date.weekday()])
    my_models = [None]*7
    my_models[date.weekday()] = my_net
    all_paths = []
    i = 0
    while(i < n_iters):
        print(i+1, '/', n_iters)
        sp = shortest_path(my_models, walking_network, stations, source, dest, date)
        uncertainies = routing_algo(sp, delay_distribution)
        path_safety = uncertainies[1] # safety of the entire path, [0,1]
        all_paths.append((sp, uncertainies))
        # stop iterating if the path is safe enough
        if path_safety >= threshold:
            break
        my_net = remove_worst_edge(my_net, uncertainies, sp)
        my_models[date.weekday()] = my_net
        i += 1
    # get the shortest path and the n_paths-1 safest paths
    results = [all_paths.pop(0)]
    
    for i in range(n_paths-1):
        if len(all_paths) == 0:
            break
        safest_path = max(all_paths, key=lambda item: item[1][1])
        results.append(safest_path)
        all_paths.remove(safest_path)

    return results
    

#######################################################################
#______________________ BOKEH MAP VISUALIZATION  ______________________
#######################################################################

def to_merc(lat, long):
    """
    Come from homework, change latitude longitude into mercator coordinates
    """
    return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), long, lat)

def create_line(a, index):
    """
    Create different lines according to the different trips of the journey
    """
    result = []
    if len(index) == 1:
        result.append(a[:index[0]+1])
        result.append(a[index[0]:])
    else:
        for i,j in enumerate(index):
            if i+1 == len(index):
                result.append(a[index[i-1]:j+1])
                result.append(a[j:])
            elif i == 0:
                result.append(a[:j+1])
            else:
                result.append(a[index[i-1]:j+1])
    return result

def create_data_for_visu(pandas_df, path):
    """
    Create all the required data for the visualization for the given path.
    """
    x = []
    y = []
    transport = []
    labels = []
    chang = []

    for i, road in enumerate(path):
        long1 = pandas_df[pandas_df.station_ID == road[0]].long.values[0]
        long2 = pandas_df[pandas_df.station_ID == road[1]].long.values[0]

        lat1 = pandas_df[pandas_df.station_ID == road[0]].lat.values[0]
        lat2 = pandas_df[pandas_df.station_ID == road[1]].lat.values[0]

        lab1 = pandas_df[pandas_df.station_ID == road[0]].name.values[0]
        lab2 = pandas_df[pandas_df.station_ID == road[1]].name.values[0]

        merc1 = to_merc(lat1, long1)
        merc2 = to_merc(lat2,long2)

        x.append(merc1[0])
        x.append(merc2[0])

        y.append(merc1[1])
        y.append(merc2[1])

        labels.append(lab1)
        labels.append(lab2)

        transport.append(road[5])

        if transport[i-1] != transport[i]:
            chang.append(i)


    labels = list(dict.fromkeys(labels))
    x = list(dict.fromkeys(x))
    y = list(dict.fromkeys(y))

    lines_x = create_line(x,chang)
    lines_y = create_line(y,chang)
    
    labels_show = ['' if (i not in chang) and (i!=0) and (i!=len(labels)-1) else lab for i,lab in enumerate(labels)]
    
    transport_array = []
    transport_array.append(transport[0])
    [transport_array.append(t) for i,t in enumerate(transport[1:]) if t != transport[i]]

    colormap = {t:Set3[12][i] for i,t in enumerate(set(transport))}
    colors = [colormap[x] for x in transport_array]

    return labels, labels_show, x, y, lines_x, lines_y, transport_array, colors

def plot_trip(pandas_df, path):
    """
    Plot the map of the path
    """
    output_notebook()

    labels, labels_show, x, y, lines_x, lines_y, transport_array, colors = create_data_for_visu(pandas_df, path)

    source = ColumnDataSource(data=dict(x=x, y=y, names=labels_show, desc=labels))

    label_set = LabelSet(x='x', y='y', text='names', level='glyph',
                  x_offset=-10, y_offset=10, source=source, render_mode='canvas')

    zurich_coord = to_merc(47.378177, 8.540192)
    x_y_offset = 6000
    
    hover = HoverTool(tooltips=[("stop name", "@desc")])
    
    p = figure(x_range=(-x_y_offset+zurich_coord[0], x_y_offset+zurich_coord[0]), 
               y_range=(-x_y_offset+zurich_coord[1], x_y_offset+zurich_coord[1]),
               x_axis_type="mercator", 
               y_axis_type="mercator", tools=[hover, "pan","wheel_zoom","box_zoom","reset"])

    p.add_tile(CARTODBPOSITRON)

    for (colr, leg, x, y ) in zip(colors, transport_array, lines_x, lines_y):
        my_plot = p.line(x, y, color = colr, legend = leg, line_width=4)

    p.circle(x="x", y="y", size=15, fill_color="#000000", source=source)

    p.add_layout(label_set)

    show(p)