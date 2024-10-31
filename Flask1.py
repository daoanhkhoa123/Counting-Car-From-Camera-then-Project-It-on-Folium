from flask import Flask, render_template, url_for
import folium
from numpy.random import choice
import time
import random 

app = Flask(__name__)

posts = [
    {      
        'author': 'Corey',
        'title' : 'Blog',
        'content': 'first',
        'date':'nothing'   
    },
        {      
        'author': 'Nupe',
        'title' : 'Blog',
        'content': 'second',
        'date':'nothing'   
    }
]


@app.route("/")
def home():
    return render_template('layout.html', posts=posts)

@app.route("/about")
def about():
    return render_template('about.html')

def gen_dummies_data(num):
    lst = []
    for i in range(num):
        car_count = random.choice([5, 10, 25, 30, 7, 8])
        status = 'light' if car_count < 10 else 'medium' if car_count < 25 else 'heavy'
        lst.append({'time': time.strftime("%Y-%m-%dT%H:%M:%S"), 'car_count': car_count, 'status': status})
    return lst
@app.route("/generate_map")
def generate_map():
    gen_map()  # This will update the map with new data
    return {"status": "Map updated"}

# def gen_map():
#     road1_coords = [[37.7749, -122.4194], [37.7750, -122.4195]]  # Road 1
#     road2_coords = [[37.7745, -122.4189], [37.7746, -122.4190]]  # Road 2
   
#     m = folium.Map(location=[37.7749, -122.4194])
#     m.fit_bounds([[37.7740, -122.4200], [37.7760, -122.4180]])
    
#     traffic_data = gen_dummies_data(20)
#     for data in traffic_data:
#         color = 'green' if data['status'] == 'light' else 'orange' if data['status'] == 'medium' else 'red'
#         folium.PolyLine(road1_coords, color=color, weight=15, opacity=0.7).add_to(m)
#         folium.PolyLine(road2_coords, color=color, weight=15, opacity=0.7).add_to(m)    
#     # Save map to HTML
#     m.save('static/map.html')
#     return 


# FOR TESTING
def gen_map():
    market_street_coords = [
        [37.774929, -122.419416],
        [37.776276, -122.417543],
        [37.778618, -122.414797],
        [37.779903, -122.413396],
        [37.781357, -122.411145],
        [37.783084, -122.408658],
        [37.784679, -122.406312],
        [37.786188, -122.404083],
        [37.787994, -122.401395],
        [37.790198, -122.399113],
        [37.791955, -122.396637],
        [37.793731, -122.394242]
    ]

    m = folium.Map(location=[37.7749, -122.4194], zoom_start=13)
    traffic_data = gen_dummies_data(5)
    for data in traffic_data:
        color = 'green' if data['status'] == 'light' else 'orange' if data['status'] == 'medium' else 'red'
        folium.PolyLine(market_street_coords, color=color, weight=8, opacity=0.6).add_to(m)
    m.save('static/map.html')

# MAIN RUN
if __name__ == '__main__':
    gen_map()
    app.run(debug=1)

