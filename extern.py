#
# author: Paul Galatic
#
# Program to store common utilities.
#

import time
import pathlib

# CONSTANTS
KML_PATH = pathlib.Path('kml/')
TIME_FORMAT = '%H:%M:%S'
ALPHA = 0.1

# These thresholds serve as informal decision boundaries for our program, 
# arrived at through manual experimentation and educated assumptions.
STOP_TIME_THRESHOLD = 480
MATCH_TIME_THRESH = 1200
MATCH_POINT_THRESH = 100
MATCH_ROUTE_THRESH = 0.6
MOVING_THRESH = 10
STOPPED_THRESH = 5
JITTER_THRESH = 2
EX_THRESH = 0.01

# These constants help us control anomalies and set bounds on what we can 
# expect from a reasonable route.
MIN_DIST = 100      # We expect to drive at least 100 meters.
MIN_TIME = 60       # We expect a trip to take at least 60 seconds.
MIN_STOP = 0        # We expect to stop no fewer than 0 times.
MAX_DIST = 300000   # We expect to drive less than 300 kilometers.
MAX_TIME = 10800    # We expect a trip to take less than 3 hours.
MAX_STOP = 1000     # We expect to stop less than 1000 times.
MAX_STEP = 1000     # We expect to move no more than 1 kilometer in a GPS step.

def log(*args, nostamp=False):
    '''More informative print debugging'''
    t = time.strftime(TIME_FORMAT, time.localtime())
    s = '\t'.join([str(arg) for arg in args])
    if not nostamp: s = f'[{t}]: ' + s
    print(s)