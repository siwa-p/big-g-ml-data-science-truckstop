# Big G Express: Predicting Derates

This is part of a team project which I worked with [Michael Tatar](https://github.com/michaeltatarjr) and [Alex Balli](https://github.com/ballir27) as a part of NSS Data Science project coursework.

In this project, we are working with fault code data and vehicle onboard diagnostic data to try and predict an upcoming full derate. These are indicated by an SPN 5246. 

We have two files containing the data you will use to make these predictions (J1939Faults.csv and VehicleDiagnosticOnboardData.csv) as well as two files describing some of the contents (DataInfo.docx and Service Fault Codes_1_0_0_167.xlsx) 

Note that in its raw form the data does not have "labels", so you must define what labels you are going to use and create those labels in your dataset. Also, you will likely need to perform some significant feature engineering in order to build an accurate predictor.

__To define labels__
In order to identify labels to use as target variable for our model, we have built a new column (boolean) that is true if there are any fault codes 7 days earlier and leading up to a derate.


__Predictors__
step 1 : Choosing a limited number of equipments based on the proportion of missing values

step 2: Choosing Features with at least 50 percent or more non-missing values

step 3 : Introduction of new features based on the datetime index of the fault code

step 4 : Filtering for incidences within a radius of 5 miles around on of the following service locations:

There are service locations at (36.0666667, -86.4347222), (35.5883333, -86.4438888), and (36.1950, -83.174722), so you should remove any records in the vicinity of these locations, as fault codes may be tripped when working on the vehicles.

step 5: For the fault codes, we noticed that "spn" and "fmi" fault codes show up in pairs. Each such pairs were identified and a dummy column for each pair was created to be used as predictors of future derate.

step 6: A logistic regression model and a Random forest decision tree were explored. Due to the nature of imbalance in the data, we employed "balanced-class" in our model.

step 7: The performance of your model were evaluated based on the cost associated with a missed full derate (approximately $4000 in towing and repairs) and the cost of a false positive prediction ($500 due to having the truck off the road and serviced unnecessarily).