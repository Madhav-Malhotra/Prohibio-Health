### Prohibio Health - Climate Mosquito Model Data Packager
### Madhav Malhotra
### Under Guidance of Yacov Iland, Jim Fare, and Ashu Syal
### This project takes open source research data on 
### climate and mosquito populations in Manatee County, 
### Florida. It then creates a predictive model for
### mosquito populations based on the climate data.

import csv

#Processing raw precipitation data
with open('PRCP.csv', 'r') as PRCPInput:
  with open('PRCP-processed1.csv', 'w') as PRCPOutput:
    #Iterable 2D array of csv file
    PRCPReader = csv.reader(
        PRCPInput, delimiter = ',');
    PRCPWriter = csv.writer(
      PRCPOutput, delimiter = ",", quotechar = '"', quoting = csv.QUOTE_ALL);

    stations = []; #Holds station IDs
    latitudes = []; #Checking for new station locations
    stationTag = 0; #Stations from 1 - 16
    outputPRCP = []; #Holds rows for output file

    headings = next(PRCPReader, None); #header row
    outputPRCP.append(headings);

    for row in PRCPReader: #Checking all PRCP records
      #If new location found
      if row[1] not in latitudes:
        #Update locations and ID tag
        latitudes.append(row[1]);
        stationTag += 1;
        #Add new station to list
        stations.append((stationTag, row[1], row[2]));
      
      row[0] = stationTag;
      outputPRCP.append(row);

    PRCPWriter.writerows(outputPRCP);
    PRCPOutput.close();
  PRCPInput.close(); 

#Processing raw temperature data
with open('Temp.csv', 'r') as tempInput:
  with open('Temp-processed1.csv', 'w') as tempOutput:
    #Iterable 2D array of csv file
    tempReader = csv.reader(
        tempInput, delimiter = ',');
    tempWriter = csv.writer(
      tempOutput, delimiter = ",", quotechar = '"', quoting = csv.QUOTE_ALL);

    outputTemp = []; #Holds rows for output file

    headings = next(tempReader, None); #header row
    outputTemp.append(headings);
    
    for row in tempReader: #Checking all temp. records
      for station in stations:
        if row[1] == station[1]: 
          row[0] = station[0];
          break;
      outputTemp.append(row);

    #Output processed data
    tempWriter.writerows(outputTemp);
    tempOutput.close();
  tempInput.close(); 

print("Classified weather stations")

#Processing insect data
with open('Insects.csv', 'r') as insectInput:
  with open('Insects-processed1.csv', 'w') as insectOutput:
    insectsReader = csv.reader( #Reading raw data
      insectInput, delimiter = ',');
    insectsWriter = csv.writer( #To output processed data
      insectOutput, delimiter = ",", quotechar = '"', quoting = csv.QUOTE_ALL);
    outputInsects = []; #Stores output 2D array
    columnsToKeep = [0, 6, 9, 10]; #Features to keep

    headings = next(insectsReader, None); #header row
    #Adding header row to output.
    headingsCopy = [];
    for col in columnsToKeep:
      headingsCopy.append(headings[col]) 
    headingsCopy.append('Closest Station');
    outputInsects.append(headingsCopy);

    for row in insectsReader:
      shortestDist = (1, 999999999999999999);
      #Assumes no insect is ^^^^^ degrees away from 
      #a weather station.

      for station in stations:  
        #Calculating proximity of current mosquito 
        #to every weather station
        DistX = ( float(row[9]) - float(station[1]) ) **2;
        DistY = ( float(row[10]) - float(station[2]) ) **2;
        currentDist = ( DistX + DistY ) **0.5;

        if currentDist < shortestDist[1]:
          #Deciding which weather station is closest 
          #to current mosquito
          shortestDist = (station[0], currentDist);
      
      #Creating copy of each row with processed data
      rowcopy = [];
      for col in columnsToKeep:
        rowcopy.append(row[col]);
      rowcopy.append(shortestDist[0]);
      outputInsects.append(rowcopy);
    #Outputting processed insect data
    insectsWriter.writerows(outputInsects);
    insectOutput.close();
  insectInput.close();
  
print("Calculated Proximity");

#Read Processed Insects file
with open('Insects-processed1.csv', 'r') as inputIP:
  IPReader = csv.reader(
    inputIP, delimiter = ',');
  #Check if mosquito from new month
  months = [];
  numMosquitoesMonthly = {}; #Maps months to insect numbers

  next(IPReader, None);
  for row in IPReader:
    currentDate = row[1];
    if currentDate[:7] not in months:
      #Add to list of months with data
      numMosquitoesMonthly[currentDate[:7]] = 1;
      months.append(currentDate[:7]);
    else:
      #Map each month to number of mosquitoes that month
      for month in numMosquitoesMonthly.keys():
        if currentDate[:7] == month:
          count = numMosquitoesMonthly[currentDate[:7]];
          numMosquitoesMonthly[currentDate[:7]] = count + 1;
  #Note: Mosquito count categories are 0-100, 101-500,
  #501-2000, 2001-3500
  inputIP.close();

#Read Processed PRCP file 
with open('PRCP-processed1.csv', 'r') as inputPP:
  PPReader = csv.reader(inputPP);
  #Preparing final output (csv for Tensorflow)
  outputSemifinal = [];
  outputSemifinal.append(["Date", "Latitude", "Longtitude", "Weather Station", "PRCP", "Monthly Mosquito Count"]);
  
  next(PPReader, None);
  for row in PPReader:
    #If record from one of months with insect data 
    currentDate = row[4];
    if currentDate[:7] in numMosquitoesMonthly.keys():
      #Save date, latitude, longtitude, station ID, 
      #PRCP, and monthly mosquitoes in 2D array
      monthlyMosquitoCount = numMosquitoesMonthly[currentDate[:7]];
      outputRow = [row[4], row[1], row[2], row[0], row[5], monthlyMosquitoCount];
      outputSemifinal.append(outputRow);
  
  inputPP.close();

#Read Temp. file
with open("Temp-processed1.csv", 'r') as inputTP:
  TPReader = csv.reader(inputTP);
  outputFinal = [];
  outputFinal.append(["Date", "Latitude", "Longtitude", "Weather Station", "PRCP", "Monthly Mosquito Count", "TMAX", "TMIN"]);

  next(TPReader, None);
  count = 1;
#If record matches date and ID from PRCP processed data
  for row in TPReader:
    if row[4] == outputSemifinal[count][0] and row[0] == outputSemifinal[count][3]:
      #Append temp data to saved 2D Array
      rowcopy = outputSemifinal[count];
      rowcopy.append(row[5]);
      rowcopy.append(row[6]);
      outputFinal.append(rowcopy);
      count += 1;
  
  inputTP.close();

for row in range(1, len(outputFinal)):
  currentRow = outputFinal[row];
  if currentRow[5] < 101:
    currentRow[5] = '0-100';
  elif currentRow[5] > 100 and currentRow[5] < 501:
    currentRow[5] = '101-500';
  elif currentRow[5] > 500 and currentRow[5] < 2001:
    currentRow[5] = '501-2000';
  elif currentRow[5] > 2000 and currentRow[5] < 3501:
    currentRow[5] = '2001-3500';
  
  #Note: Mosquito count categories are 0-100, 101-500,
  #501-2000, 2001-3500

#Outputting final processed data for Tensorflow dataset
with open('Final-processed.csv', 'w') as outputFP:
  FPWriter = csv.writer(outputFP, delimiter = ",",
   quotechar = '"', quoting = csv.QUOTE_ALL);
  
  FPWriter.writerows(outputFinal);

  #As Mr. Fare said, ALWAYS close your files...
  outputFP.close();

#Splitting processed data into train:test (8:2)
with open('Final-processed.csv', 'r') as inputFP:
  with open('Final-train.csv', 'w') as outputFTrain:
    with open('Final-test.csv', 'w') as outputFTest:
      FPReader = csv.reader(inputFP); #All data
      FTrainWriter = csv.writer( #To write train file
        outputFTrain,
        quoting = csv.QUOTE_ALL);
      FTestWriter = csv.writer( #To write test file
        outputFTest,
        quoting = csv.QUOTE_ALL);
      
      headerRow = next(FPReader, None); #Gets csv headings
      outputRaw = []; #Copy of all data

      for row in FPReader:
        outputRaw.append(row); 
      
      #80% of all data
      outputTest = outputRaw[:6401];
      #Adds csv header
      outputTest.insert(0, headerRow);
      #Outputs test data
      FTrainWriter.writerows(outputTest);

      #20% of all data
      outputTrain = outputRaw[6401:];
      #Adds csv header
      outputTrain.insert(0, headerRow);
      #Output train data
      FTestWriter.writerows(outputTrain);

      outputFTest.close();
    outputFTrain.close();
  inputFP.close();