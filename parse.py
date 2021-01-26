# author: Paul Galatic
# author: Sarthak Gupte
#
# Program to turn a GPS file into a KML file.
#
# Three TASKS:
#  - Develop a means of finding when two routes are the same (e.g. "two routes
#    are the same if 90% of their points have at least 1 'match'")
#       - It would be a good idea to "compress" the route to increase speed of
#           data processing
#  - Develop a function that will return True if two points are "the same place"
#    and False otherwise
#  - Organize all examples of a given route into sets and then return the average
#    stats (#stops, distance, and time) for all examples in the set
#

# STD LIB
import os
import pdb
import sys
import math
import pickle
import argparse
from datetime import datetime, date

# EXTERNAL LIB
import pynmea2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# PROJECT LIB
from extern import *


def parse_args():
    '''Parses arguments'''
    ap = argparse.ArgumentParser()

    ap.add_argument('--target', nargs='?', const=None, default=None,
                    help='dir containing gps files to parse')
    ap.add_argument('--datafile', nargs='?', const=None, default=None,
                    help='pre-computed data')

    return ap.parse_args()


def parse_all(fname):
    '''
    Tries to parse a NMEA file using PYNMEA2. Only lines that can be parsed are
    returned.
    '''
    # IDEA: Include preprocessing here (e.g. for anomales)? We can toss any
    # messages from a route that we feel aren't helpful. We could also have a
    # 'fail gracefully' case where we discard an entire route if necessary.
    messages = []

    with open(fname, 'r', encoding='iso-8859-1') as gps:
        for line in gps:
            # Only parse valid NMEA lines
            if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                try:
                    msg = pynmea2.parse(line)
                except pynmea2.nmea.ChecksumError:
                    continue
                except pynmea2.nmea.ParseError:
                    continue

                if getattr(msg, 'lon', '') == '':
                    continue  # avoid messages with no longitude value

                if getattr(msg, 'spd_over_grnd', -1) != -1:
                    continue  # Skip debugging lines

                messages.append(msg)

    return messages


def msg_latlon(msg):
    # Longitude is in the form dddmm.mmmm, so we extract the first
    # three digits as degrees and the rest as minutes.
    lon = float(msg.lon[:3]) + (float(msg.lon[3:]) / 60)
    lon = round(lon, 6)
    # Latitude is the same, but in the form ddmm.mmmm.
    lat = float(msg.lat[:2]) + (float(msg.lat[2:]) / 60)
    lat = round(lat, 6)
    # West and South mean we need to multiply the longitude and
    # latitude, respecitvely, by -1.
    if msg.lon_dir == 'W':
        lon = -lon
    if msg.lat_dir == 'S':
        lat = -lat
    return f'{lon},{lat},{0}'


def header():
    '''Write the header of the KML file.'''
    return '\n'.join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<Style id="yellowPoly">',
        '  <LineStyle>',
        '    <color>Af00ffff</color>',
        '    <width>6</width>',
        '  </LineStyle>',
        '  <PolyStyle>',
        '    <color>7f00ff00</color>',
        '  </PolyStyle>',
        '</Style>',
        '<Style id="redPushpin">',
        '  <Icon>',
        '    <href>https://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png</href>',
        '  </Icon>',
        '</Style>',
        '<Placemark><styleUrl>#yellowPoly</styleUrl>',
        '  <LineString>',
        '    <Description>Speed in Knots, instead of altitude.</Description>',
        '  <extrude>1</extrude>',
        '  <tesselate>1</tesselate>',
        '  <altitudeMode>absolute</altitudeMode>',
        '  <coordinates>',
        '',
    ])


def body(messages, stops):
    '''Write the body of the KML file.'''
    lines = []
    # Add the linestring that denotes the route
    for msg in messages:
        lines.append(msg_latlon(msg))

    # Switch from using linestrings to using pushpins.
    lines += [
        '    </coordinates>',
        '   </LineString>',
        '  </Placemark>',
    ]

    # Add the stops in the route.
    for stop in stops:
        lines += [
            '<Placemark><styleUrl>#redPushpin</styleUrl>',
            '  <Point>',
            '  <altitudeMode>absolute</altitudeMode>',
        ]
        lines.append(f'    <coordinates>{msg_latlon(stop)}</coordinates>')
        lines += [
            '  </Point>',
            '</Placemark>',
        ]

    return '\n'.join(lines)


def footer():
    '''Write the tail-end of the KML file.'''
    return '\n'.join([
        '',
        ' </Document>',
        '</kml>',
    ])


def build_kml(fname, messages, stops):
    '''Write all the stages of the KML file.'''
    oname = str(KML_PATH / (os.path.splitext(os.path.basename(fname))[0] + '.kml'))
    with open(oname, 'w') as out:
        out.write(header())
        out.write(body(messages, stops))
        out.write(footer())


def convert_to_degree(attribute, limit):
    '''Convert radians to degrees.'''
    degree = float(attribute[:limit]) + (float(attribute[limit:]) / 60)
    degree = round(degree, 6)
    return math.radians(degree)


def point_dist(msg1, msg2):
    '''
    Calculates the distance between two messages. We will find the latitude and
    longitude for the messages. Then using haversine formula we will calculate
    the distance(kms) between two GPS points.
    '''
    earth_radius = 6373.0

    msg1_lon = convert_to_degree(msg1.lon, 3)
    msg1_lat = convert_to_degree(msg1.lat, 2)
    msg2_lon = convert_to_degree(msg2.lon, 3)
    msg2_lat = convert_to_degree(msg2.lat, 2)

    difference_lon = msg1_lon - msg2_lon
    difference_lat = msg1_lat - msg2_lat

    a = math.sin(difference_lat / 2) ** 2 \
        + math.cos(msg1_lat) * math.cos(msg2_lat) \
        * math.sin(difference_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dist = earth_radius * c
    kms = round(dist, 4)
    return kms * 1000


def get_total_time(messages):
    '''
    The total time elapsed in a route is given by the starting and ending
    messages.
    '''
    # IDEA: Sometimes we get negative total duration, and that msut be
    # investigated. There may be other errors. We should make sure the GPS
    # files look okay.
    first = messages[0].timestamp
    last = messages[-1].timestamp
    elapsed = datetime.combine(date.min, last) - datetime.combine(date.min, first)
    return elapsed.total_seconds()


def get_total_dist(messages):
    '''
    Calculates the total amount of distance in a route, given a series of
    messages describing that route.
    '''
    total_dist = 0

    prev = None
    for msg in messages:
        if prev:
            curr_dist = point_dist(msg, prev)
            # IDEA: We will probably have to compensate for the "resting noise"
            # movement as the GPS signal jitters around.
            if curr_dist > JITTER_THRESH:
                total_dist += curr_dist
            # Some routes have gigantic jumps which are anomalous and must be
            # discarded. Returning MAX_DIST + 1 filters this route out.
            if curr_dist > MAX_STEP:
                return MAX_DIST + 1

        prev = msg

    return round(total_dist, 2)


def get_total_stops(messages):
    '''
    Calculates the total number of stops in a route, given a series of messages
    describing that route.
    '''
    stops = set()
    stop_time = 0

    prev = None
    moving = False
    for msg in messages:
        if prev:
            if moving:
                # If we slow down past the stopping theshold, but we have moved
                # recently, add a stop. In either case, reset the stop time.
                # This keeps us from recording really long stops (e.g. to get
                # groceries) as stops.
                if point_dist(prev, msg) < STOPPED_THRESH:
                    moving = False
                    if stop_time <= STOP_TIME_THRESHOLD:
                        stops.add(msg)
                    stop_time = 0
            else:
                # If we start moving again, record it. Otherwise, increase the
                # amount of time that we've been stopped.
                if point_dist(prev, msg) > MOVING_THRESH:
                    moving = True
                else:
                    stop_time += 1

        prev = msg

    return stops

def filter_routes(route):
    '''
    Filters the messages of the route and saves only the index with a
    multiple of 6, and checks if these indexes are not  stops.
    '''
    prev_index = None
    updated_route = []

    for msg_index in range(len(route)):
        if msg_index % 6 == 0:

            # if first msg add to route
            if not prev_index:
                updated_route.append(route[msg_index])
                prev_index = msg_index

            # if not the first msg check if it is not a stop.
            elif point_dist(route[msg_index], route[prev_index]) > STOPPED_THRESH:
                updated_route.append(route[msg_index])
                prev_index = msg_index

    return updated_route


def match_point(msg1, route2):
    '''
    Checks for matching msg in the route if found return true
    else false.
    '''
    for msg in route2:
        if point_dist(msg1, msg) < MATCH_POINT_THRESH:
            return True

    return False


def match_routes(route1, route2):
    '''
    Checks for similarity between two routes if the difference in data point
    is nearly 1200 (four minutes) then similar. Also if both the route has more than
    60% common data points.
    '''
    points_matched = 0

    if route1 and route2:

        # check if the difference between the routes have is 5 mins. If routes 
        # have very different durations, don't consider them similar.
        if abs(len(route1) - len(route2)) <= MATCH_TIME_THRESH:
            return True

        else:
            # Apply a second layer of resolution reduction
            filter1 = filter_routes(route1)
            filter2 = filter_routes(route2)
        
            # finding matching number of points.
            for msg in filter1:
                if match_point(msg, filter2):
                    points_matched += 1

        if points_matched / len(filter1) >= MATCH_ROUTE_THRESH:
            # print(points_matched / len(filter1))
            return True

    return False


def extract(messages):
    '''
    Provided a set of messages, this function will return a dictionary in the
    form:
    {
        'time': total_time,
        'dist': total_dist,
        'stps': total_stps
    }

    Where:
        time -> amount of time elapsed over a route
        dist -> amount of distance traveled over a route
        stps -> number of stops in a route
    '''
    #filtered = filter_routes(messages)
    
    total_time = get_total_time(messages)
    total_dist = get_total_dist(messages)
    total_stps = get_total_stops(messages)

    return {
        'time': total_time,
        'dist': total_dist,
        'stps': total_stps
    }


def check_extremities(start, end):
    '''
    Checks that the start and end points of a given route matches with the
    expected end and start point.
    '''
    # expected start coordinates are professor's home's lat and long.
    expected_start_lat = 4308.3045
    expected_start_lon = 07726.2645

    # expected end coordinates are RIT's lat and long.
    expected_end_lat = 4305.1650
    expected_end_lon = 07740.8100
    
    # check to see if the start and end points are anywhere near the expected
    # points
    if (abs(float(start.lat) - expected_start_lat) < EX_THRESH and
        abs(float(start.lon) - expected_start_lon) < EX_THRESH and
        abs(float(end.lat) - expected_end_lat) < EX_THRESH*10 and
        abs(float(end.lon) - expected_end_lon) < EX_THRESH*10):
        return True
    else:
        return False


def pareto_visualize(target, data_list):
    '''
    Given a list of routes, visualizes the ones that meet the threshold of
    potential pareto optimality. INCOMPLETE
    '''
    CSV_OUT = f'{target}.csv'
    PARETO_OUT = f'{target}_pareto.csv'

    dataframe = pd.DataFrame(data_list)
    headers = dataframe.columns[:3]
    # Calculate normalized dataset
    array = dataframe[['time', 'stps', 'dist']]
    scalar = preprocessing.MinMaxScaler()
    scaled = scalar.fit_transform(array)
    normalized = pd.DataFrame(scaled, columns=headers)
    # Apply regularization
    dataframe['time'] = dataframe['time'] * ((1 - ALPHA) + normalized['dist'] * ALPHA)
    dataframe['stps'] = dataframe['stps'] * ((1 - ALPHA) + normalized['dist'] * ALPHA)
    log(dataframe, nostamp=True)
    dataframe.to_csv(CSV_OUT, sep=' ', columns=['time', 'stps', 'fname'], index=False, header=False)
    # Run open source code to find pareto optimals
    os.system(f'python pareto/pareto.py {CSV_OUT} -o 0 1 --output {PARETO_OUT}')
    pareto = pd.read_csv(PARETO_OUT, sep=' ', header=None, names=['time', 'stps', 'fname'])
    log(pareto, nostamp=True)

    data_plt = plt.scatter(dataframe['time'], dataframe['stps'], c='b', marker='o')
    pare_plt = plt.scatter(pareto['time'], pareto['stps'], c='r', marker='o')

    plt.title('All routes and pareto optimal routes')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of stops')
    plt.legend([data_plt, pare_plt], ['Full dataset', 'Pareto optimal routes'])

    plt.show()


def process_gps_files(target):
    data_list = []
    fnames = [os.path.join(target, fil) for fil in os.listdir(target)]
    common_route = None
    similar_routes = []

    for fname in fnames:
        if fname.endswith('.txt'):  # This is a GPS file
            log(f'Converting {fname}...')
            messages = parse_all(fname)
            data = extract(messages)

            # Use constants defined in extern.py to remove obviously anomalous
            # or unreasonable data (negative driving time, driving farther than
            # 100km, et cetera).
            if data['time'] < MIN_TIME:
                log('ANOMALY: Duration too short ({t} < {l})'.format(
                    t=data['time'], l=MIN_TIME))
                continue
            if data['time'] > MAX_TIME:
                log('ANOMALY: Duration too long ({t} > {l})'.format(
                    t=data['time'], l=MAX_TIME))
                continue
            if data['dist'] < MIN_DIST:
                log('ANOMALY: Distance too short ({t} < {l})'.format(
                    t=data['dist'], l=MIN_DIST))
                continue
            if data['dist'] > MAX_DIST:
                log('ANOMALY: Distance too long ({t} > {l})'.format(
                    t=data['dist'], l=MAX_DIST))
                continue
            if len(data['stps']) < MIN_STOP:
                log('ANOMALY: Too few stops ({t} < {l})'.format(
                    t=len(data['stps']), l=MIN_STOP))
                continue
            if len(data['stps']) > MAX_STOP:
                log('ANOMALY: Too many stops ({t} > {l})'.format(
                    t=len(data['stps']), l=MAX_STOP))
                continue

            # We will need to build the KML after we extract the data so that
            # we can render stops, turns, etc. We should construct our messages
            # from parse_all() so that build_kml() can use them.
            build_kml(fname, messages, data['stps'])

            data['stps'] = len(data['stps'])

            # Check if start and end point is RIT and Prof's home. If valid, 
            # add it.
            if check_extremities(messages[0], messages[-1]):
                if len(similar_routes) == 0:
                    common_route = messages
                    similar_routes.append(messages)
                else:
                    if match_routes(common_route, messages):
                        similar_routes.append(messages)
            else:
                log('ANOMALY: Extremities do not match')
                continue

            data['fname'] = fname
            data_list.append(data)
            log('Time: {t}\tDist: {d}\tStops: {s}'.format(
                t=data['time'], d=data['dist'], s=data['stps']))
            # log('Similar routes: {s}'.format(s=len(similar_routes)))

    # Store the data file so that it doesn't need to be computed over and over
    # unnecessarily.
    with open(target + '.pkl', 'wb') as outfile:
        pickle.dump(data_list, outfile)

    return data_list


def main():
    '''Driver program'''
    args = parse_args()
    log('Starting...')

    if not (args.target or args.datafile):
        log('Please specify target or datafile (e.g. python parse.py --target=gps)')
        sys.exit(1)

    if not os.path.exists('kml'):
        os.mkdir('kml')

    # If we don't need to recompute the data, then load from a pickle file.
    if args.datafile:
        with open(args.datafile, 'rb') as infile:
            data_list = pickle.load(infile)
        target = os.path.splitext(args.datafile)[0]
    # Otherwise, compute from scratch.
    else:
        data_list = process_gps_files(args.target)
        target = args.target

    pareto_visualize(target, data_list)

    log('...finished.')
    return 0


if __name__ == '__main__':
    main()
