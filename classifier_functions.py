################################################################################
################### Classifier functions                 #######################
################################################################################  
def classify_percentage(percentage):
  """
  Classify into 10 buckets (0-10,10-20,20-30,40-50
                            50-60,60-70,70-80,80-90,90-100)
  """
  return round(percentage, -1)
  
def classify_percentage2(percentage):
  """
  Classify into 3 buckets (0-34, 34-68, 68-100)
  """
  return (int) (percentage/34.0)

def classify_int(percentage):
  """
  Classify by returning the int value of the percentage value.
  """
  return int(percentage)

def classify_percentage3(percentage):
  """
  Classify into 3 buckets (0-30, 30-70, 70-100)
  """
  if 0.0 <= percentage <= 30.0:
    return 0
  elif 70.0 <= percentage <= 100.0:
    return 2
  else:
    return 1

def classify_percentage4(percentage):
  """
  Classify into 3 buckets (0-15, 15-85, 85-100)
  """
  if 0.0 <= percentage <= 15.0:
    return 0
  elif 15.0 <= percentage <= 85.0:
    return 1
  else:
    return 2
  
def classify_percentage5(percentage):
  """
  Classify as 0% => 0, 100% => 2, and 1 otherwise
  """
  if percentage < 1.0:
    return 0
  elif percentage > 99.0:
    return 2
  return 1
    
def classify_shortage_binary(percentage):
  """
  Classify into 2 buckets (0-10, 10-100)
  """
  return 0 if percentage < 10.0 else 1

def classify_overflow_binary(percentage):
  """
  Classify into 2 buckets (0-90, 90-100)
  """
  return 0 if percentage < 90.0 else 1

def classify_distance (distance):
  if 0.0 <= distance <= 1.0:
    return 0
  elif 1.0 <= distance <= 2.0:  
    return 1
  elif 2.0 <= distance <= 3.0:
    return 2
  elif 3.0 <= distance <= 4.0:
    return 3
  else:
    return 4