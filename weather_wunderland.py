import urllib2, re, json, os, pytz, datetime, calendar, time
from datetime import date, timedelta as td


class WeatherUtil:
  stations = {"London": "ILONDONT2", "Oslo" : "IOSLOOSL31", "NYC": "KNYNEWYO85", 
  "Chicago": "KILCHICA130", "New York": "KNYNEWYO71", "Washington, DC": "KDCWASHI18"}
  weather = {}
  parsed_URLs = []

  def __init__(self, city, weather_filename, parsed_URLs_file):
    self.city = city
    self.weather_filename = weather_filename
    self.parsed_URLs_file = parsed_URLs_file    
    if os.path.isfile(self.weather_filename) and os.path.isfile(self.parsed_URLs_file):
      print "loading weather from file"
      self.load_weather_from_file()
      #print self.weather.keys()

  def get_weather_for_day(self, station_id, day, month, year):
    url = "http://www.wunderground.com/weatherstation/WXDailyHistory.asp?ID=%s&day=%s&month=%s&year=%s&graphspan=day&format=1" % (station_id, day, month, year)
    if url in self.parsed_URLs:
      return None
    print url
    self.parsed_URLs.append(url)
    content = urllib2.urlopen(url).read()

    #Remove line breaks
    content = re.sub(r"<br>|<br />", "", content)

    #Split into lines
    content = content.split("\n")

    #Remove empty lines
    content = filter(lambda a: len(a) > 0, content)

    return content

  def UTC_time_to_epoch(self,timestamp):
    timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    epoch = calendar.timegm(timestamp.utctimetuple())
    return epoch
  
  def get_closest_epoch(self, epoch):
    #print epoch
    d = time.gmtime(epoch)
    day, month, year = d.tm_mday, d.tm_mon, d.tm_year

    #Parse the URL for this day if this hasn't been done already
    self.save_weather(day, month, year)

    keys = self.weather.keys()

    #Find the closest weather report to this epoch
    min_diff = abs(epoch - (int) (keys[0]))
    closestKey = None
    for key in keys:
      diff = abs((int) (key) - epoch)
      if diff < min_diff:
        min_diff = diff
        closestKey = key

    return closestKey

  def get_closest_weather_report(self, epoch):
    closestKey = self.get_closest_epoch(epoch)
    if closestKey is not None:
      return self.weather[closestKey]

  def save_weather(self, day, month, year):
    #Save in the format: weather[epoch]['weather_feature']
    content = self.get_weather_for_day(self.stations[self.city], day ,month, year)

    if content is None:
      return

    #Example of an header
    #['Time', 'TemperatureC', 'DewpointC', 'PressurehPa', 'WindDirection',
    # 'WindDirectionDegrees', 'WindSpeedKMH', 'WindSpeedGustKMH', 'Humidity',
    # 'HourlyPrecipMM', 'Conditions', 'Clouds', 'dailyrainMM', 'SoftwareType', 'DateUTC']
    header = content[0].split(",")
    utc_recording_index = header.index('DateUTC')

    #Separate header from weather recordings
    content = content[1:]

    #Remove trailing comma from the recording
    content = [line.rstrip(",") for line in content]

    for line in content:
      weather_features = line.split(",")
      epoch = self.UTC_time_to_epoch(weather_features[utc_recording_index])

      if epoch in self.weather:
        continue

      self.weather[epoch] = {}

      #Ignore the last column (UTC time), as that is already the key
      for i in range(len(header) - 1):
        self.weather[epoch][header[i]] = weather_features[i]

    self.save_weather_to_file()

  def save_weather_in_timespan(self, start_date, end_date):
    dates = self.find_all_dates_in_between(start_date, end_date)
    for date in dates:
      self.save_weather(unicode(date.day), unicode(date.month), unicode(date.year))

    self.save_weather_to_file()
    #print self.weather.keys()

  def find_all_dates_in_between(self, start_date, end_date):
    start_y, start_m, start_d = start_date
    end_y, end_m, end_d = end_date

    d1 = date(start_y,start_m,start_d)
    d2 = date(end_y,end_m, end_d)

    delta = d2 - d1

    dates = []

    for i in range(delta.days + 1):
      dates.append(d1 + td(days=i))

    return dates

  def save_weather_to_file(self):
    fo = open(self.weather_filename, "w")
    s = json.dumps(self.weather, sort_keys=True, indent=4)
    fo.write(s)
    fo.close()

    fo = open(self.parsed_URLs_file, "w")
    fo.write(str(self.parsed_URLs))
    fo.close()

  def load_weather_from_file(self):
    fo = open(self.weather_filename, "r")
    self.weather = json.loads(fo.read())
    fo.close()

    fo = open(self.parsed_URLs_file, "r")
    self.parsed_URLs = eval(fo.read())
    fo.close()

  def print_weather(self):
    s = json.dumps(self.weather, sort_keys=True, indent=4)
    print s

