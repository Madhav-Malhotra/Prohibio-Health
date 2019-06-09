# Prohibio-Health
This is a predictive model of insect breeding and migration patterns in Manatee County, Florida from 2012 to 2015.

Our open-source insect data came from: https://www.vectorbase.org/popbio/map/#
Our open-source climate data came from: https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/
The Tensorflow tutorial used can be found here: https://www.tensorflow.org/beta/tutorials/load_data/csv

# Instructions for use
The raw data files Temp.csv (Temperature minimum and maximum records), PRCP.csv (Precipitation records), and Insects.csv (mosquito population records) can be found in the repository.
The data_packager.py file converts these raw data files to usable test and training forms. Run this python script with the raw data files downloaded to get the processed files. 
The main.py file converts the processed csv data to a Tensorflow dataset and then creates a neural network model based on that.
I have attached a copy of what the final test and train data looks like before being used in the neural network (testLPO.csv and trainLPO.csv).

# Other resources
Prohibio Health Website: https://paige-gugeler.github.io/Prohibio-Health-Website/
Project Article: https://medium.com/@miaisakovic/mitigating-the-impact-of-parasitic-outbreaks-e14a4212ba68
Prohibio Health Business Pitch: https://www.youtube.com/watch?v=BNuZ9z9iZHM&feature=youtu.be
