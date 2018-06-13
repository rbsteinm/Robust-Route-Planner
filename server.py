#!flask/bin/python

import sys
import random, json

from flask import Flask, render_template, jsonify, request, redirect, Response
import helpers
from datetime import datetime
app = Flask(__name__)


stations = helpers.load_metadata()
models = helpers.load_networks()
walking_network = helpers.compute_walking_network(stations)
ZH_HB = 8503000

reachable_stations_ids = helpers.get_reachable_stations(models[0], walking_network, ZH_HB)
reachable_stations = {sid: stations[sid] for sid in reachable_stations_ids}

@app.route('/')
def index():
    test = helpers.shortest_path(models,walking_network,stations,ZH_HB, 8502559, datetime(2018, 1, 15, 14, 0))
    print(test)
    return render_template('index.html',name=None)
    return page
    for path in y:
        stop_a = stations[path[0]]['name']
        stop_b = stations[path[1]]['name']
        time_a = path[2].strftime('%H:%M')
        time_b = path[3].strftime('%H:%M')
        line = path[4]
        text += 'line ' + str(line) + '<br> from ' + stop_a + ' to ' + stop_b
        text += '<br>' + time_a + ' -> ' + time_b + '<br><br>' + '</p>'
    return '<h2>Path</h2><p>'+text+'</p>'

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
        x = helpers.shortest_path(models,walking_network,stations,ZH_HB, destination, datetime(int(date['year']), int(date['month']), int(date['day']), int(date['hour']),int(date['minute'])))
    else:
        x = helpers.shortest_path_reverse(models,walking_network,stations,ZH_HB, destination, datetime(int(date['year']), int(date['month']), int(date['day']), int(date['hour']),int(date['minute'])))
    y = helpers.reduce_path(x)
    #print(y)
    #return render_template('test.html', name=None)
    return jsonify(result={'data':helpers.reduced_path_to_json(y)})




@app.route('/eval/')              
def eval():
        return render_template("test.html")




if __name__ == '__main__':
    # run!
    app.run()