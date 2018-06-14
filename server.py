#!flask/bin/python

import sys
import random, json
import pandas as pd

from flask import Flask, render_template, jsonify, request, redirect, Response
import helpers
from datetime import datetime
app = Flask(__name__)


stations = helpers.load_metadata()
models = helpers.load_networks()
walking_network = helpers.compute_walking_network(stations)
ZH_HB = 8503000
delay_distribution_pd = pd.read_pickle('./data/full_delay_distri.pickle')

reachable_stations_ids = helpers.get_reachable_stations(models[0], walking_network, ZH_HB)
reachable_stations = {sid: stations[sid] for sid in reachable_stations_ids}

@app.route('/')
def index():
    return render_template('index.html',name=None)

@app.route('/get_stations')
def get_stations():
    return jsonify({str(key): reachable_stations[key] for key in reachable_stations.keys()})

@app.route('/receiver', methods = ['POST'])
def worker():
    # read json + reply
    data = request.get_json()
    date = data['date']
    reverse = (data['dep'] == 'arr')
    destination = int(data['station'])
    if not reverse:
        #x = helpers.safest_paths(models, walking_network, reachable_stations, ZH_HB, 8591436, datetime(2018, 1, 15, 14), delay_distribution_pd)
        paths = helpers.safest_paths(models,walking_network,reachable_stations,ZH_HB, destination, datetime(int(date['year']), int(date['month']), int(date['day']), int(date['hour']),int(date['minute'])), delay_distribution_pd)
    else:
        paths = helpers.safest_paths(models,walking_network,reachable_stations,ZH_HB, destination, datetime(int(date['year']), int(date['month']), int(date['day']), int(date['hour']),int(date['minute'])), delay_distribution_pd, reverse=True)
    res = dict()
    for i,x in enumerate(paths):
        res[str(i)] = {'path': helpers.reduced_path_to_json(helpers.reduce_path(x[0])), 'safety': round(100*x[1][1]), 'time':str(helpers.compute_path_time(x[0]))}
    return jsonify(result=res)

    # y = [helpers.reduce_path(x[0]), x[1][1] for x in paths]
    #print(y)
    #return render_template('test.html', name=None)
    #return jsonify(result={'data':helpers.reduced_path_to_json(y)})




@app.route('/eval/')              
def eval():
        return render_template("test.html")




if __name__ == '__main__':
    # run!
    app.run()