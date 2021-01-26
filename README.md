# gps
For organizing code related to RIT CSCI-720 Group Project 1.

# Installation
```
pip install -r requirements.txt
```

# Running the program
Input files are expected to be in the NMEA format.

```
python parse.py --target=<FOLDER_WITH_GPS_TEXT_FILES>
```

The program parses all files before performing visualization. It genreates a
PKL file once this parsing is finished. In order to skip the parsing step and 
just repeat visualization, use this command:

```
python parse.py --datafile=<FOLDER_WITH_GPS_TEXT_FILES>.pkl
```

