from flask import Flask, render_template, jsonify
import helpers
from datetime import datetime
app = Flask(__name__)


stations = helpers.load_metadata()
models = helpers.load_networks()
walking_network = helpers.compute_walking_network(stations)

@app.route('/')
def index():
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
    return jsonify({str(key): stations[key] for key in stations.keys()})

@app.route('/receiver', methods = ['POST'])
def worker():
    # read json + reply
    data = request.get_json()
    result = ''

    for item in data:
        # loop over every row
        result += str(item['make']) + '\n'

    return result