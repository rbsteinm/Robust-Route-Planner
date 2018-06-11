from flask import Flask
import helpers
from datetime import datetime
app = Flask(__name__)

@app.route('/')
def hello_world():
    stations = helpers.load_metadata()
    models = helpers.load_networks()
    walking_network = helpers.compute_walking_network(stations)
    x = helpers.shortest_path(models,walking_network,stations,8576218, 8590727, datetime(2018, 1, 15, 14))
    #x = helpers.hav(2)
    y = helpers.reduce_path(x)
    text = '<p>'
    for path in y:
        stop_a = stations[path[0]]['name']
        stop_b = stations[path[1]]['name']
        time_a = path[2].strftime('%H:%M')
        time_b = path[3].strftime('%H:%M')
        line = path[4]
        text += 'line ' + str(line) + '<br> from ' + stop_a + ' to ' + stop_b
        text += '<br>' + time_a + ' -> ' + time_b + '<br><br>' + '</p>'
    return '<h2>Path</h2><p>'+text+'</p>'