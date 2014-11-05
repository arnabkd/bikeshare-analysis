from datautils import *
from classifier_functions import *
from sklearn.cluster import KMeans, DBSCAN
import pylab as plt

def get_station_similarity_matrix(data_path):
  data , stations = init(data_path)
  station_IDs = stations.keys()

  #Similarity matrix
  #similarity_m[X][Y] will tell you how many times stations X and Y have experienced the same 
  #classification at the same time.
  similarity_matrix =  [[0 for x in station_IDs] for y in station_IDs]

  print "Computing affinity (similarity) matrix"

  size = len(data.keys())
  c = 0
  for epoch in data.keys():
    system_status = data[epoch]
    c += 1
    if c % (size/10) == 0:
      print "Processed %s/%s files" % (c,size)

    for x in station_IDs:
      for y in station_IDs:
        if (y not in system_status) or (x not in system_status):
          continue

        if classify_percentage2(system_status[x]) == classify_percentage2(system_status[y]):
          similarity_matrix[x][y] += 1


  #Affinity Matrix, used for clustering
  sm = np.array(similarity_matrix)
  return sm, stations

def KMeans_Cluster(sm, stations):
  print "Fitting KMeans"
  #KMeans
  kmeans = KMeans(n_clusters=2)
  kmeans.fit(sm)

  labels = kmeans.labels_

  return labels, stations

def affinity_matrix(similarity_matrix, stations):
  #Threshold is atleast 95% similarity
  max_matches = max(similarity_matrix[0])
  cluster_match_matrix = [[0 for station_ID in stations.keys()] for key in stations.keys()]
  for x in range(len(similarity_matrix)):
    for y in range(len(similarity_matrix[0])):
      if similarity_matrix[x][y] > (0.9*max_matches):
        cluster_match_matrix[x][y] = 1

  return cluster_match_matrix, stations


def set_similar(affinity_matrix, stations):
  for i in range(len(affinity_matrix)):
    similar = []
    for j in range(len(affinity_matrix[i])):
      if (affinity_matrix[i][j] == 1):
        print "Appending station #%s to similar stations list for station #%s" %(j, i)
        similar.append(j)

    stations[i]['similar'] = similar
  return stations


def plot_clusters (labels, stations):
  labels = [int(label) for label in labels]
  print "Plotting result"
  colors = plt.cm.Spectral(np.linspace(0,1,len(set(labels))))
  #print similarity_m
  for stationID in stations.keys():
    station = stations[stationID]
    lat,lon = station['latitude'], station['longitude']
    plt.scatter(lon,lat, color=colors[labels[stationID]])
  plt.legend()
  plt.show() 

if __name__ == "__main__":
  try:
    data_path = sys.argv[1]
    sm,stations = get_station_similarity_matrix(data_path)
    am, stations = affinity_matrix(sm, stations)
    stations = set_similar(am, stations)

    if len(sys.argv) == 3 and "update" in sys.argv[2]:
      stations = set_similar(am, stations)
      fo = open(data_path+"racks_dict", "w")
      fo.write(str(stations))
      fo.close()
  except:
    print "Error"



  
	  
