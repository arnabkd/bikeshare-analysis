import sys, glob, random, time, datetime, os
from datautils import *
import matplotlib.pyplot as plt
from classifier_functions import *
from sklearn import tree, ensemble
from sklearn.neighbors import KNeighborsRegressor
import datetime, dateutil.relativedelta
from sklearn.metrics import mean_squared_error
from math import sqrt
from send_email import *
import matplotlib.cm as cm
import csv
import time
from itertools import izip


img_names = []


def visualize(data_path, station_IDs=None):
  """
  visualize the traffic of a bike-share station.
  """
  data,features = get_bikeshare_data(['epoch'], ['id'], [], data_path, regression_mode=True)
  if station_IDs is None or len(station_IDs) < 1:
    station_IDs = [random.randint(0, max(data[:,1])) for x in range(10)]
    station_IDs = random.sample(station_IDs, 2)
  
  station_IDs = [int(station_ID) for station_ID in station_IDs]
  
  print "Displaying data for %s" %station_IDs
  for station_ID in station_IDs:
    timestamp = [d[0] for d in data if d[1] == station_ID]
    bikes = [d[2] for d in data if d[1] == station_ID]

    plt.plot(timestamp,bikes,label= ("Bikeshare traffic for station_ID: %s" % station_ID)) 


  plt.xlabel("Time", fontsize=40)
  ticks = np.arange(min(data[:,0]), max(data[:,0]), 24*60*60)
  labels = ["Day %s"% (x+1) for x in range(len(ticks))]
  plt.xticks(ticks, labels, fontsize=30)
  plt.yticks(fontsize=30)
  plt.ylabel("Number of free bikes at station", fontsize=40)
  plt.legend(prop={'size': 30})
  plt.show()

def visualize_all(data_path):

  data,features = get_bikeshare_data(['epoch'], ['id'], [], data_path, regression_mode=True)
  plt.plot(data[:,0],data[:,2])
  plt.xlabel("Time")
  plt.ylabel("Number of free bikes")
  ticks = np.arange(min(data[:,0]), max(data[:,0]), 24*60*60)
  labels = ["Day %s"% (x+1) for x in range(len(ticks))]
  plt.xticks(ticks, labels)
  plt.show()



def filter_station(X, id_index, station_ID, y=None):
  """
  Filter a dataset to only contain data from a certain station
  """
  station_ytrain = []
  station_Xtrain = []

  for i in range(len(X)):

    s_ID = int(X[i][id_index])
    if s_ID== station_ID:
      station_Xtrain.append(X[i])
      if y is not None:
        station_ytrain.append(y[i])

  if y is not None:
    return np.array(station_Xtrain), np.array(station_ytrain)

  return np.array(station_Xtrain)

def filter_similar(X, id_index, y, similar_stations):
  """
  Filter a dataset to only contain data from the stations in the similar_stations list
  """
  similar_Xtrain = []
  similar_ytrain = []
  
  for i in range(len(X)):
    s_ID = int(X[i][id_index])
    if s_ID in similar_stations:
      similar_Xtrain.append(X[i])
      similar_ytrain.append(y[i])

  return np.array(similar_Xtrain), np.array(similar_ytrain)

def visualize_prediction(estimator, est_label, X_train, y_train, X_test, y_test, features):
  fig = plt.figure(1)
  plot = fig.add_subplot(111)
  print "Training estimator: %s" % est_label 
  estimator.fit(X_train, y_train)
  print "Running predictions"
  y_pred = estimator.predict(X_test)
  y_train_pred = estimator.predict(X_train)

  #print min(y_train), max(y_train)

  #Calculate tick sizes for the X axis
  num_observations_per_day = (24*60*60)/(5*60)
  test_sample_size = len(X_test)
  test_sample_range = range(test_sample_size)
  
  #Set zoom for figure
  plt.xlim(0, test_sample_size)
  plt.ylim(0,20)
  
  #Plot prediction and test data
  plt.plot(test_sample_range, y_test, label="Actual test data", color="green")
  plt.plot(test_sample_range, y_pred, label="Prediction", color="red")

  num_days_in_test_sample = (int) (test_sample_size / num_observations_per_day) + 1
  x = np.arange(0, test_sample_size, num_observations_per_day)
  plt.xticks(x, ["day #%s"% (i+1) for i in range(num_days_in_test_sample)])
  plt.grid()

  #print "Max-xtest: %s, max-ytest: %s, max-ypred: %s" % (test_sample_size, max(y_test), max(y_pred))
  
  test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
  

  #Add axis and graph legends, title
  title = "Estimator: %s, RMSE: %s" %(est_label, test_rmse)
  plt.legend(loc=1,prop={'size':20})
  plt.xlabel('Time', fontsize=20)
  plt.ylabel('Number of bikes available', fontsize=20)
  plt.title(title, fontsize=30)

  #Change font size for tick labels
  plot.tick_params(axis='both', which='major', labelsize=15)
  plot.tick_params(axis='both', which='minor', labelsize=15)

  #Export feature importances to a .csv file in the bin folder
  try:
    feature_importances = get_sorted_features(features, estimator.feature_importances_)
    feature_importances = [(f_name, importance*100) for (f_name,importance) in feature_importances]
    features_sorted = [f[0] for f in feature_importances]
    if "percentage" in features_sorted:
      del feature_importances[features_sorted.index("percentage")]

    features = [feature for feature,_ in feature_importances]
    importances = [importance for _,importance in feature_importances]

    if not os.path.exists("bin/"):
      os.mkdir("bin/")

    with open(("bin/feature_importances_%s.csv"%est_label), "wb") as f:
      writer = csv.writer(f)
      #print "created writer object"
      writer.writerows(izip(features, importances))
      print "Feature importances saved to file: %s" %str(f)
  except:
    print "Something went wrong while showing feature importances"
  plt.show()
  
  return train_rmse, test_rmse


def prune_target_col(data, target_col):
  """
  Remove the target column of a dataset
  """
  #print "Removing index %s of %s " %(target_col, len(data[0])-1)
  y = np.copy(data[:,target_col])
  empty_col = np.array([0 for i in range(len(data[:,target_col]))])
  data[:,target_col] = empty_col
  return data

def get_target(data, target_col):
  """
  Returns a numpy array containing the target column of a dataset
  """
  return np.copy(data[:,target_col])


def create_train_test_split(X, y, split):
  """
  Create a train-test split in the data using the split variable as the ratio
  """
  if split > 99 or split < 1: 
    print "Error: The split must be between 1 and 100"
    return

  print "Creating a %s - %s split" %(split, 100-split)
  sep = (len(data)*split)/100.0

  X_train = np.array(data[:sep])
  y_train = np.array(y[:sep])

  X_test = np.array(data[sep:])
  y_test = np.array(y[sep:])

  return X_train, y_train, X_test, y_test

def output_RMSE_dict(stations, X_train, y_train, X_test, y_test, estimator, id_index, full_dataset=True):
  print "Using full dataset: %s" %full_dataset
  rmse_test = {}
  rmse_train = {}
  
  #Fit estimator to full dataset if required, otherwise fit individually
  if full_dataset:
    print "Fitting to full dataset"
    estimator.fit(X_train, y_train)

  for s in stations.keys():
    try:
      if not full_dataset:
        print "Fitting individually for station %s" %s
        similar_stations = stations[s]['similar']
        similar_Xtrain, similar_ytrain = filter_similar(X_train, id_index, y_train, similar_stations)
        #Fit estimators to each individual station's similar stations
        estimator.fit(similar_Xtrain, similar_ytrain)
        print "Prediction for station #%s" %s

      station_Xtrain, station_ytrain = filter_station(X_train, id_index, s, y_train)
      station_predict_train = estimator.predict(station_Xtrain)
      rmse_train[s] = sqrt(mean_squared_error(station_ytrain, station_predict_train))

      station_Xtest, station_ytest = filter_station(X_test, id_index, s, y_test)
      station_predict_test = estimator.predict(station_Xtest)
      rmse_test[s] = sqrt(mean_squared_error(station_ytest, station_predict_test))
    except:
      print "Something went wrong for station #%s" %s
      print stations[s]

  return rmse_train, rmse_test

def visualize_RMSE_dict(title, plot_ID, rmse_dict, c=None):
  X = rmse_dict.keys()
  y = rmse_dict.values()
  ax = plt.subplot(plot_ID)  
  plt.legend()
  plt.scatter(X,y, color="blue" if c is None else c)
  plt.title(title)
  ax.set_xlim(padding([X]))
  ax.set_ylim(padding([y]))

def run_prediction(data_path, estimator, estimator_name, dt):
  time_params = ['time_of_day_hours', 'day_of_week', 'is_weekend']
  station_params = ['id', 'latitude', 'longitude', 'altitude']
  weather_params = ['TemperatureC', 'HourlyPrecipMM', 'Conditions', 'WindSpeedKMH', 'WindSpeedGustKMH', 'Humidity']
  data, features = get_bikeshare_data(time_params, station_params, weather_params, data_path, regression_mode=True)

  stations = init_racks(data_path)

  X_train, y_train, X_test, y_test = create_train_test_split(data, len(features)-1, 95)

  rmse_train1, rmse_test1 = output_RMSE_dict(stations, X_train, y_train, X_test, y_test, estimator, features.index('id'), True)
  rmse_train2, rmse_test2 = output_RMSE_dict(stations, X_train, y_train, X_test, y_test, estimator, features.index('id'), False)
  
  
  try:
    visualize_RMSE_dict("With full dataset", 211, rmse_test1,c="blue")
    visualize_RMSE_dict("With only similar subsets", 212, rmse_test2, c="green")
  except:
    pass
  global img_names
  img_names.append('bin/estimator_%s_%s.png' %(estimator_name, dt))
  plt.savefig(img_names[-1],pad_inches=0.1)
  
  #for station in stations.keys():
    #if rmse_test1[station] < rmse_test2[station]:
      #print "Station #%s predicted better with only a \"similar stations\" subset as the training set" %station
      #print "RMSE with full training set: %s, with subset %s " % (rmse_test1[station], rmse_test2[station])


  
  return rmse_test1, rmse_test2

def find_max(val_lists):
  return np.max([np.max(val_list) for val_list in val_lists])  

def find_min(val_lists):
  return np.min([np.min(val_list) for val_list in val_lists])
 
def padding(val_lists, p=None):
  """
  Used to find the optimal padding for a prediction plot
  """
  if p is None:
    p = 0.1
  min_val = find_min(val_lists)
  max_val = find_max(val_lists)
  padding = np.abs(max_val - min_val) * p
  return (min_val - padding, max_val + padding)

def plot_stats(rmse_dicts, labels, estimator_name, dt):
  if len(labels) is not len(rmse_dicts):
    print "Error: Number of labels must be equal to the number of RMSE matrices"
    return -1

  ax = plt.subplot(211)
  colors = "bgrcmykw"

  for i in range(len(rmse_dicts)):
    print "Plotting scatter for rmse_dicts[%s]" %i
    plt.scatter(rmse_dicts[i].keys(), rmse_dicts[i].values(), label=labels[i], color=colors[i] if i < len(colors) else "blue")
  plt.legend()
  plt.title("Scatter plot to show RMSE of predictions across all stations")
  ax.set_xlim(padding([rmse_dict.keys() for rmse_dict in rmse_dicts]))
  ax.set_ylim(padding([rmse_dict.values() for rmse_dict in rmse_dicts],0.5))
  
  ax = plt.subplot(212)
  y_means = []
  for i in range(len(rmse_dicts)):
    print "Plotting mean for rmse_dicts[%s]" %i
    y_mean = [np.mean(rmse_dicts[i].values()) for key in rmse_dicts[i].keys()]
    y_means.append(y_mean)
    plt.plot(rmse_dicts[i].keys(), y_mean, label= labels[i], linestyle="--")
  plt.legend()
  ax.set_xlim(padding([rmse_dict.keys() for rmse_dict in rmse_dicts]))
  ax.set_ylim(padding(y_means))
  plt.title("Mean performance when using different training sets")
  

  global img_names
  img_names.append('bin/stats_estimator_%s_%s.png' %(estimator_name, dt))
  plt.savefig(img_names[-1], pad_inches=0.1)
  
  return 0


def run_prediction_for_station(est, est_name, X,y,  features, stations ,station_ID, ratio):
  X_train, y_train, X_test, y_test = create_train_test_split(X,y, ratio)
  print X_train[0], y_train[0]
  print X_test[0], y_test[0]
  #print "Size of total training data %s" %len(X_train)
  #similar_Xtrain, similar_ytrain = filter_similar(X_train, len(time_params),y_train, stations[station_ID]['similar'])

  #print "Size of filtered training data %s" %len(similar_Xtrain)
  #print "Length of test set %s hours" %(len(X_test[:,0]) / (12.0*320)) 

  #print "Using learning data from stations"
  #print stations[station_ID]['similar']

  station_Xtrain, station_ytrain = filter_station(X_train, len(time_params), station_ID, y_train)
  station_Xtest, station_ytest = filter_station(X_test, len(time_params), station_ID, y_test)
  """
  for (est, est_name) in estimators:
    print (est, est_name)
    visualize_prediction(est, est_name +"\nTrained with only similar stations", similar_Xtrain, similar_ytrain, station_Xtest, station_ytest, features)
    plt.clf()
  """
  start_time = time.time()
  train_rmse, test_rmse = visualize_prediction(est, est_name, X_train, y_train, station_Xtest, station_ytest, features)
  lapsed_time = time.time() - start_time

  print "Training error: %s , test error: %s" %(train_rmse, test_rmse)
  print "Time elapsed: %s" % lapsed_time

def show_prediction_errors(training_set, test_set, estimators, labels):
  X_train, y_train = training_set
  X_test, y_test = test_set

  errors=[]
  
  #Plot RMSE for each estimator
  for estimator in estimators:
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    error = np.sqrt(np.mean(np.square(y_test - y_pred)))
    errors.append(error)

  indexes = [x for x in range(len(estimators))]
  width = 0.1
  
  plt.bar(indexes, errors)
  plt.xticks(np.array(indexes)+width/2., labels)
  plt.show()

def run_tests(data_path, estimators, estimator_labels):
  data, features = get_bikeshare_data(['epoch'], ['id'], [], data_path, regression_mode=True)
  
  sep = (len(data)*90)/100
  target_col = len(data[0])-1
  target = data[:,target_col]
  target = np.copy(data[:,target_col])

  empty_col = np.array([0 for i in range(len(data[:,target_col]))])
  data[:,target_col] = empty_col

  X_train = np.array(data[:sep])
  y_train = target[:sep]

  X_test = np.array(data[sep:])
  y_test = target[sep:]

  show_prediction_errors((X_train,y_train), (X_test, y_test), estimators, estimator_labels)

def run_regressors(data_path):
  regressors = [tree.DecisionTreeRegressor(), ensemble.RandomForestRegressor(n_estimators=50, n_jobs=4, verbose=2)]#, ensemble.ExtraTreesRegressor(n_estimators=50, n_jobs=4, verbose=2), KNeighborsRegressor()]
  labels = ["DecisionTreeRegressor", "RandomForestRegressor (n_estimators=50)", "ExtraTreesRegressor(n_estimators=50)"]# "KNeighborsRegressor" ]

  run_tests(data_path, regressors, labels)


def send_regression_report(data_path, estimator, estimator_name, email_to, username, password):
  email_to, username, password = sys.argv[2], sys.argv[3], sys.argv[4]
  rmse1, rmse2 = run_prediction(data_path, estimator, estimator_name, dt)
  plt.clf()
  plot_stats([rmse1, rmse2],["Training set = full", "Training set = only_similar"], estimator_name, dt)
  send_email(email_to, username, password, "smtp.gmail.com", 587, report , subject="Regression report", files=img_names)

def print_usage_and_exit():
  """
  Called when the script is called with the wrong arguments. Just print the desired usage and exit with -1.
  """
  print "Usage: python visualize_station <data_path> <station_ID>"
  print "The data_path variable should be the relative path to the folder containing your dataset"
  print "The station ID must be an integer describing the station ID, making sure that there is such a station in the dataset provided"
  sys.exit(-1)


if __name__ == "__main__":
  if len(sys.argv) is not 3:
    print_usage_and_exit()

  try:
    data_path = sys.argv[1]
    station_ID = (int) (sys.argv[2])
    time_params = ['epoch', 'time_of_day_hours', 'day_of_week', 'day_of_month','month','time_of_day_minutes']
    station_params = ['id', 'latitude', 'longitude', 'altitude']
    weather_params = ['TemperatureC', 'HourlyPrecipMM', 'Conditions']
    data, features = get_bikeshare_data(time_params, station_params, weather_params, data_path, regression_mode=True)
  except:
    print_usage_and_exit()


  y = get_target(data, len(features) - 1)
  X = prune_target_col(data, len(features) - 1)

  features = rename_features(features)
  stations = init_racks(data_path)

  estimators = [(tree.DecisionTreeRegressor(), "DecisionTreeRegressor")]
  
  #Uncomment for deeper analysis
  #estimators = [(tree.DecisionTreeRegressor(), "DecisionTreeRegressor"), (ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=30), "Random Forest Regressor (n=30)"), (ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(),n_estimators=30), "Ada Boost Regressor (n=30)")]

  for est, est_name in estimators:
    #Tweak the split_ratio variable to change the train-test split
    split_ratio = 80
    run_prediction_for_station(est, est_name, X,y, features, stations, 59, split_ratio)
