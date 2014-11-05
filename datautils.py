import numpy as np
import sys, glob, datetime, time, pytz, json, math
sys.path.insert(0,'../weather')
from weather_wunderland import *
import unicodedata
from collections import Counter
from itertools import *


################################################################################
################### Calculation helper functions         #######################
################################################################################  
"""
  Calculate distance between two coordinates
  @John D. Cook
  www.johndcook.com/python_longitude_latitude.html
"""
def distance(lat1, long1, lat2, long2):

  # Convert latitude and longitude to 
  # spherical coordinates in radians.
  degrees_to_radians = math.pi/180.0
        
  # phi = 90 - latitude
  phi1 = (90.0 - lat1)*degrees_to_radians
  phi2 = (90.0 - lat2)*degrees_to_radians
        
  # theta = longitude
  theta1 = long1*degrees_to_radians
  theta2 = long2*degrees_to_radians

  # Compute spherical distance from spherical coordinates.
        
  # For two locations in spherical coordinates 
  # (1, theta, phi) and (1, theta, phi)
  # cosine( arc length ) = 
  #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
  # distance = rho * arc length
    
  cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + math.cos(phi1)*math.cos(phi2))
  arc = math.acos( cos )

  # Remember to multiply arc by the radius of the earth 
  # in your favorite set of units to get length.
  return arc * 6373
  
def get_month_from_epoch(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return dt.month

def get_day_of_month(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return dt.day

def get_time_of_day_minutes(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return ((dt.hour *60) + dt.minute)

def get_time_of_day_hour(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return dt.hour if dt.minute < 30 else dt.hour + 1

def get_day_of_week(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return dt.isoweekday()

def get_is_weekend(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return 0 if dt.isoweekday() > 5 else 1

def get_hour_minutes(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return dt.hour, dt.minute

def get_hour(epoch):
  return datetime.datetime.fromtimestamp(epoch).hour

def get_minutes_from_weekstart(epoch):
  dt = datetime.datetime.fromtimestamp(epoch)
  return ((dt.isoweekday()-1) * (24*60)) + dt.hour*60 + dt.minute


################################################################################
################### Time Qualifier checks                #######################
################################################################################
def check_epoch_between_hours(start_hour, end_hour, epoch, tz=None):
  hour = (int) (datetime.datetime.fromtimestamp(epoch).hour)
  return start_hour <= hour < end_hour

def check_morning_rush(epoch,tz=None):
  return check_epoch_between_hours(6,9,epoch,tz)
  
def check_morning_rush2(epoch,tz=None):
  return check_epoch_between_hours(8,9,epoch,tz)
  
def check_evening_rush(epoch,tz=None):
  return check_epoch_between_hours(16,19,epoch,tz)
  
def check_evening_rush2(epoch,tz=None):
  return check_epoch_between_hours(16,17,epoch,tz)

def check_evening_rush(epoch,tz=None):
  return check_epoch_between_hours(15,18,epoch,tz)

def check_if_before_12(epoch):
  return check_epoch_between_hours(0,12, epoch)

def check_if_weekend(epoch):
  t = time.ctime(epoch)
  return 'Sun' in t or 'Sat' in t

def check_if_monday(epoch):
  t = time.ctime(epoch)
  return 'Mon' in t

def check_if_monday_tuesday(epoch):
  t = time.ctime(epoch)
  return 'Mon' in t or 'Tue' in t

def check_if_weekday(epoch):
  return not check_if_weekend(epoch)


################################################################################
################### Rack            Qualifier checks     #######################
################################################################################
def filter_rack_by_ID(rack_ID):
  return rack_ID == 4

################################################################################
################### Bike percentage Qualifier checks     #######################
################################################################################
def check_shortages(Z):
  return 0 <= Z <= 30

def check_overflows(Z):
  return 70 <= Z

def check_to_remove_medium_percentages(Z):
  return not (30 <= Z <= 70)

def check_empty(Z):
  return Z <= 1.0


################################################################################
################### Filter functions                     #######################
################################################################################
"""
  Cleanse the data, remove off-season data.
"""
def cleanse_XYZ(XYZ):
  #Save the invalid data points in this list, remove them later
  #You cannot remove them directly while traversing, otherwise race conditions give wrong results
  invalid_dataset = []

  for X,YZ in XYZ:
    invalid = True
    
    for Y,Z in YZ:
      if Z > 0:
        #print "found one non-zero point in this file (time: %s)"%(X)
        invalid = False
        break
    if invalid:
      invalid_dataset.append((X,YZ))

  for X,YZ in invalid_dataset:
    XYZ.remove((X,YZ))

  return XYZ


"""
  X: epoch
  Y: coordinates
  Z: free bikes percentage

  Every datapoint in XYZ is a tuple (X,YZ)
  If the tuple passes our X,Y,Z func tests, add it to a valid data set
  Then return the valid data set
"""
def filter_data_XYZ(XYZ, Xchecktest=None, Ychecktest=None, Zchecktest=None):
  valid_data = []
  for X,YZ in XYZ:
    if (Xchecktest is None) or Xchecktest(X):
      valid_YZ = []
      for Y,Z in YZ:
        if (Ychecktest is None) or Ychecktest(Y):
          if (Zchecktest is None) or Zchecktest(Z):
            valid_YZ.append([Y,Z])
      valid_data.append([X,valid_YZ])
  return valid_data


################################################################################
################### Data formatting functions for XYZ style data    ############
################################################################################

"""
  Create a 3D-matrix from data
  X: epoch
  Y: distance_from_central_rack
  Z: free bikes percentage
"""
def create_distance_matrix(data_files, racks, central_rack):
  XYZ = []
  clat, clon = central_rack['latitude'], central_rack['longitude']
  
  for f in data_files:
    fo = open(f, "r")
    data = json.loads(fo.read())
    fo.close()    
    X = data['time']
    city_status = data['stations']
    YZ = []
    for station_status in city_status:
      id = station_status['id']
      if racks[id]['capacity'] > 0:
        id = station_status['id']
        Y = distance(clat, clon, racks[id]['latitude'], racks[id]['longitude'])
        Z = (station_status['bikes']*100.0) / (racks[id]['capacity'])
        YZ.append((Y,Z))    
    XYZ.append((X,YZ))
  return XYZ

def create_obs(time_features, station_features, weather_features, content, racks, regression_mode=False, filter_rack_by_ID=None):  
  time_funcs = {"time_of_day_minutes" : get_time_of_day_minutes, "time_of_day_hours": get_time_of_day_hour,
                "day_of_month": get_day_of_month, "day_of_week" : get_day_of_week,
                "minutes_since_weekstart": get_minutes_from_weekstart,"month": get_month_from_epoch,
                "is_weekend": get_is_weekend, "epoch": int}

  okta_conv = {"CLR": 0, "FEW": 1.5, "SCT": 3.5, "BKN": 6, "OVC": 8, "-RA": 5,
   "RA": 6, "+RA": 7, "DZ": 3, "-DZ": 2, "+DZ": 4, "": 4}
  obs = []
  weather = content['weather']

  for station in content['stations']:
    invalid = False

    ob = []
    bikes = station['bikes']
    free = station['free'] 
    station_id = station['id']
    if filter_rack_by_ID is not None:
      if filter_rack_by_ID != station_id:
        continue

    for time_feature in time_features:
      func = time_funcs[time_feature]
      ob.append(func(content['time']))

    for station_feature in station_features:
      if station_id not in racks:
        print "Station not found"
        if station_feature not in racks[station_id]:
          print "Station feature not found:", station_feature
      ob.append(racks[station_id][station_feature])

    for weather_feature in weather_features:
      value = weather[unicode(weather_feature)]
      value = value.rstrip(' ')
      value = value.lstrip(' ')
      if 'Conditions' in weather_feature:
        if value not in okta_conv:
          print "Value: -%s-" %value
          invalid = True
          break

        value = (okta_conv[value])
      value = (float)(value)
      ob.append(value)

    if bikes == 0 and free == 0:
      percentage = 0
    else:
      percentage = (bikes*100.0) / (bikes + free)
    
    if regression_mode:
      ob.append(bikes)
    else:
      ob.append(percentage)

    if not invalid:
      obs.append(ob)
    else:
      raise Exception('Invalid data')
  return obs

def rename_features(features):
  time_params = ['epoch','time_of_day_hours', 'day_of_week', 'is_weekend']
  station_params = ['id', 'latitude', 'longitude', 'altitude']
  weather_params = ['TemperatureC', 'HourlyPrecipMM', 'Conditions']
  
  rename_dict = {"epoch" : "Timestamp (UNIX)", "time_of_day_hours": "Time_of_day (hours)",
   "day_of_week": "Day_of_week", "is_weekend": "Is_weekend", "id": "Station_id", "latitude": "Latitude",
   "longitude": "Longitude", "altitude": "Altitude", "TemperatureC": "Temperature (degrees Celcius)",
   "HourlyPrecipMM": "Hourly Precipitation (mm)", "Conditions": "Cloud cover"}

  renamed_features = [(rename_dict[feature] if feature in rename_dict else feature) for feature in features]
  return renamed_features

def get_bikeshare_data(time_features, station_features, weather_features, directory, regression_mode=False, filter_rack_by_ID=None):
 
  obs = []
  stations_info = eval(open(directory+ "racks_dict").read())
  bikeshare_files = glob.glob(directory + "20*.json")
  bikeshare_files.sort()
  part_size = len(bikeshare_files) / 10
  i = 0
  features = time_features + station_features+ weather_features + ["percentage"]
  if 'HourlyPrecipMM' in features:
    rain_col = features.index('HourlyPrecipMM')
  rainy_obs = 0
  non_rainy_obs = 0
  for bikeshare_file in bikeshare_files:
    fo = open(bikeshare_file, "r")
    content = json.loads(fo.read())
    fo.close()
    try:
      observation = create_obs(time_features, station_features, weather_features, content, stations_info, regression_mode, filter_rack_by_ID)
      rain_level = observation[0][rain_col]
      if rain_level > 0:
        #print rain_level
        rainy_obs += 1
      else:
        non_rainy_obs += 1
      
      obs += observation
    except Exception as e:
      print "Exception in file:", bikeshare_file
      print "Exception:" , type(e), "-", e

    i += 1

    if i%part_size == 0:
      print "Read %s files out of %s" %(i, len(bikeshare_files))
  
  print "non_rainy_obs: %s, rainy_obs: %s" %(non_rainy_obs, rainy_obs)
  return np.array(obs), features



"""
  Create a 3D-matrix from data
  X: epoch
  Y: rack ID
  Z: free bikes percentage
"""
def create_XYZmatrix(data_files, racks):
  XYZ = {}
  
  i = 0

  invalid_stations = {}

  for f in data_files:
    if "racks" in f:
      continue
    fo = open(f, "r")
    data = json.loads(fo.read())
    fo.close()    
    X = data['time']
    city_status = data['stations']
    YZ = {}
    
    i += 1

    if i%10000 == 0:
      print "Read: %d files" %i

    #Set this data to be invalid
    non_zero_racks = 0

    for station_status in city_status:
      id = station_status['id']
      if racks[id]['capacity'] > 0:
        id = station_status['id']
        Y = id
        if station_status['bikes'] == 0 and station_status['free'] == 0:
          continue
        Z = (station_status['bikes']*100.0) / (station_status['bikes'] + station_status['free'])
        
        if Z > 100.0:
          if id not in invalid_stations:
            invalid_stations[id] = {"capacity": racks[id]['capacity'], 'recorded': [station_status['bikes']]}
          else:
            invalid_stations[id]['recorded'].append(station_status['bikes'])
       
        #Data set is in season only if the recording contains at least one non-zero rack 
        if Z > 0.0:
          non_zero_racks += 1
        YZ[Y] = Z    
    
    #Only append in_season data
    in_season = (non_zero_racks > 30)
    if in_season:
      XYZ[X] = YZ
  
  for val in invalid_stations.values():
    recorded = val['recorded']
    c = Counter(recorded)
    val['recorded'] = c

  print "Found %s invalid recordings" % len(invalid_stations.keys())

  #out = open("invalid.txt", "w")
  #out.write(json.dumps(invalid_stations, sort_keys=True, indent=4))
  #out.close()
  #print invalid_stations
  return XYZ


def create_observation(obs_data, rack_features, time_features, weather_features, racks, weather_data = None):
  """
  X is the epoch mark
  Y is the rack ID
  Z is the percentage of free bikes
  """

  X,Y,Z = obs_data
  ob = []

  time_funcs = {"time_of_day" : get_time_of_day_minutes, "time_of_day_hours": get_time_of_day_hour,
                "day_of_week" : get_day_of_week, "minutes_since_weekstart": get_minutes_from_weekstart,
                "month": get_month_from_epoch, "is_weekend": get_is_weekend, "epoch": int}
  
  #Required to insert observation features in the correct order
  for time_feature in time_features:
    func = time_funcs[time_feature]
    ob.append(func(X))

  """
  #Append values for all time related features
  if "time_of_day" in time_features:
    ob.append(get_time_of_day_minutes(X))
  if "time_of_day_hours" in time_features:
    ob.append(get_time_of_day_hour(X))
  if "day_of_week" in time_features:
    ob.append(get_day_of_week(X))
  if "minutes_since_weekstart" in time_features:
    ob.append(get_minutes_from_weekstart(X))
  if "month" in time_features:
    ob.append(get_month_from_epoch(X))
  if "is_weekend" in time_features:
    ob.append(get_is_weekend(X))
  if "epoch" in time_features:
    ob.append(X)
  """

  #Append values for all rack related features
  for rack_feature in rack_features:
    ob.append(racks[Y][rack_feature])

  #Add weather data
  if weather_data is not None:
    okta_conv = {"CLR": 0, "FEW": 1.5, "SCT": 3.5, "BKN": 6, "OVC": 8}

    weather_report = weather_data.get_closest_weather_report(X)

    #if weather_report['Clouds'] == "BKN" and weather_report['Conditions'] == "CLR":
    #print type(weather_report)
    if weather_report is None:
      #print "something bad happened with the weather system at ", X
      return None
    
    for feature in weather_features:
      value = weather_report[unicode(feature)]
      if 'Conditions' in feature:
        if value not in okta_conv:
          return None

        value = (okta_conv[value])

      value = (float)(value)
      ob.append(value)
        

  #Finally return the percentage (result) with the observation
  ob.append(Z)
  if  Z > 100:
    print "Error: invalid percentage value"
    print X, Y, Z
    return None
    
  return ob

   
def XYZ_to_multidimensional_data(XYZ, racks,
                                 rack_variables=['latitude', 'longitude', 'altitude', 'capacity'],
                                 time_variables = ["time_of_day", "day_of_week", "minutes_since_weekstart"],
                                 weather_variables = ['TA', 'NN', 'RR'],
                                 verbose = False):
 
  w = None
  if weather_variables is not None:
    try:
      w = WeatherUtil("Oslo", "oslo-weather.json", "oslo_parsedURLs.txt")
    except:
      print "Weather file not found"

  if verbose:
    print time_variables + rack_variables + weather_variables + ["percentage"]

  obs = []
  for X, YZ in XYZ:
    for Y, Z in YZ:
      ob = create_observation((X,Y,Z), rack_variables, time_variables, weather_variables, racks,w)
      if ob is not None:
        obs.append(ob)
  
  
  
  return np.array(obs), time_variables + rack_variables + weather_variables + ["percentage"] 


################################################################################
################### Helper functions for classifications #######################
################################################################################
def balance(X_train, y_train):
    bias_counter = Counter(list(y_train))
    #print bias_counter
    N = min(bias_counter.values())
    #print N    
    X_new = []
    y_new = []    
    occurrences = {}
    for i in range(len(X_train)):
        target_class = y_train[i]        
        insert = (target_class in occurrences and occurrences[target_class] < N) or (target_class not in occurrences)
        if insert:
            if target_class not in occurrences:
                occurrences[target_class] = 1
            else:
                occurrences[target_class] += 1
            X_new.append(X_train[i])
            y_new.append(y_train[i])    
    return np.array(X_new), np.array(y_new)
 
def get_sorted_features (features, importances):
  importances_dict = {}
  for i in range(len(importances)):
    importances_dict[features[i]] = importances[i]
  return sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
  
def get_learning_data(XYZ, racks, rack_params, time_params, weather_params, target_name , target_classifier, pivot=None): 
  """
  Create training and test data sets
  """
  XYZ2,features =  XYZ_to_multidimensional_data(XYZ, racks, rack_params, time_params, weather_params, verbose=True)

  np.random.shuffle(XYZ2)  
  target_col_index = features.index(target_name)

  #print "Targets are of type: %s" % features[target_col_index]
  #print "Features of observations are: ", features[:target_col_index] + features[(target_col_index + 1):]

  #Remove the target column
  target = np.copy(XYZ2[:,target_col_index])
  target_name = features[target_col_index]
  empty_row = np.array([0 for i in range(len(XYZ2[:,target_col_index]))])

  XYZ2[:,target_col_index] = empty_row
  y = np.array([target_classifier(t) for t in target])
  #Divide the data into a training and test set

  if pivot is None or pivot >= len(XYZ2):
    pivot = len(XYZ2) / 2  

  X_test = XYZ2[:-pivot]
  y_test = y[:-pivot]  
  X_train = XYZ2[-pivot:]
  y_train = y[-pivot:]  

  #print "Balancing data"
  X_train_new, y_train_new = balance(X_train, y_train)
  return X_train_new, y_train_new, X_test, y_test, features

def get_learning_data2(time_params, station_params, weather_params, target_name,
 target_classifier, data_path, balance_data=False): 

  """
  Create training and test data sets
  """
  print "Weather params: %s" %weather_params
  XYZ2,features =  get_bikeshare_data(time_params, station_params, weather_params, data_path)

  np.random.shuffle(XYZ2)  
  target_col_index = features.index(target_name)

  print "Targets are of type: %s" % features[target_col_index]
  print "Features of observations are: ", features[:target_col_index] + features[(target_col_index + 1):]

  #Remove the target column
  target = np.copy(XYZ2[:,target_col_index])
  target_name = features[target_col_index]
  empty_row = np.array([0 for i in range(len(XYZ2[:,target_col_index]))])

  XYZ2[:,target_col_index] = empty_row
  y = np.array([target_classifier(t) for t in target])
  #Divide the data into a training and test set

  pivot = len(XYZ2) / 2

  X_test = XYZ2[:-pivot]
  y_test = y[:-pivot]  
  X_train = XYZ2[-pivot:]
  y_train = y[-pivot:]  

  #print "Balancing data"
  #X_train_new, y_train_new = balance(X_train, y_train)
  X_train_new, y_train_new = X_train, y_train

  if balance_data:
    X_train_new, y_train_new = balance(X_train, y_train)
    print "Balancing data"
  else:
    print "No balancing requested."
  return X_train_new, y_train_new, X_test, y_test, features

def get_X_y_features(time_params, station_params, weather_params, target_name, target_classifier, data_path, balance_data):
  X, features = get_bikeshare_data(time_params, station_params, weather_params, data_path)
  np.random.shuffle(X)
  target_col_index = features.index(target_name)
  target = np.copy(X[:,target_col_index])
  target_name = features[target_col_index]
  empty_row = np.array([0 for i in range(len(X[:,target_col_index]))])

  X[:,target_col_index] = empty_row
  y = np.array([target_classifier(t) for t in target])

  if balance_data:
    X_mod, y_mod = balance(X,y)
    return X_mod, y_mod, features
  
  return X,y, features

def get_learning_data3(time_params, station_params, weather_params, target_name,
 target_classifier, data_path, balance_data=False):
  """
  Create training and test data sets
  """
  training_folder_path = data_path+"training_set/"
  test_folder_path = data_path + "test_set/"

  if not (os.path.isdir(training_folder_path) and os.path.isdir(test_folder_path)):
    return get_learning_data2(time_params, station_params, weather_params, target_name, target_classifier, data_path, balance_data)
  else:
    print "Training and test folders detected."
    X_train, y_train, features = get_X_y_features(time_params, station_params, weather_params, target_name, target_classifier, training_folder_path, balance_data)
    X_test, y_test, features = get_X_y_features(time_params, station_params, weather_params,target_name, target_classifier, test_folder_path, balance_data)
    return X_train, y_train, X_test, y_test, features

def powerset(iterable):
  """
  Create a powerset from iterable.
  powerset([1,2,3]) --> [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
  """
  s = list(iterable)
  powerset = chain.from_iterable(combinations(s,r) for r in range(len(s)+1))
  powersetList = map(list,powerset)
  if [] in powersetList:
    powersetList.remove([])
  return powersetList
  
################################################################################
################### Data formatting functions for racks  #######################
################################################################################
"""
Create a 3D matrix with latitude, longitude and <dimension_name> as the three axes
"""
def racks_3D(racks, dimension_name):
  racks3d = []
  for rack in racks.values():
    racks3d.append([rack['longitude'], rack['latitude'], rack[dimension_name]])
  return np.array(racks3d)


def racks_4D(racks):
  racks4D = []
  for rack in racks.values():
    racks4D.append([rack['longitude'], rack['latitude'], rack['altitude'], rack['capacity']])
  return np.array(racks4D)

################################################################################
############## Init function to create dataset from data files    ##############
################################################################################
def init_racks(data_folder_path, central_rackID=None):
  racks = {}
  try:
    fo = open(data_folder_path + "racks_dict")
    racks = eval(fo.read())
    fo.close()
  except:
    print "no rack data found"
 
  return racks

def init(data_folder_path, central_rackID=None):
  print "Init called with %s" %(data_folder_path)
  data_files = glob.glob(data_folder_path +"*.json")
  racks = init_racks(data_folder_path, central_rackID)
  XYZ = create_XYZmatrix(data_files, racks)
  return XYZ, racks

