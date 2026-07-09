Survival stacking converts time-to-event data into a binary classification problem by restructuring the dataset so that each row represents a data point's status at a specific time index
. This transformation allows you to use standard off-the-shelf probabilistic binary classifiers, such as logistic regression or random forests, to predict the probability of an event happening (the discrete time hazard)
.
The conversion process follows these steps:
Enumerate unique event times: Identify all the unique times (τ(1)<τ(2)<⋯<τ(L)) at which the critical event (e.g., death) actually occurred in your training data
.
Isolate data points "at risk" per time index: For each unique event time, identify all data points that are still "at risk" (meaning they are still alive and have not been censored prior to this specific time)
.
Expand features and create binary labels: For each point at risk at a given time index, create a new feature row. Append new columns to this row that explicitly encode the current time index you are evaluating (e.g., using a one-hot encoded vector)
. Then, create a binary label indicating whether the data point actually experienced the event at that exact time index (1 if it happened, 0 if it did not)
.
Stack the data vertically: Vertically stack all the matrices and label vectors generated from every unique event time into one massive feature matrix (X) and one corresponding target vector (y)
.
In this newly stacked dataset, the binary classifier takes the original features plus the time index to predict the probability of the event occurring at that specific time, given that the subject has survived up to that point
.
A major drawback of this approach is the steep computational cost: stacking expands the size of the training dataset significantly, potentially generating up to O(n 
2
 ) rows from a dataset of n original points
. To mitigate this issue and control the matrix size, practitioners often manually discretize time into a coarser grid with fewer time steps rather than using every unique time of death