%%%------------- K-Means Clustering
% Step 1: Choose the number k of clusters

% Step 2: Select k point at random called centroids

% Step 3: Assign each datapoint to the nearest centroid which lead to K
% clusters

% Step 4: Compute new centroids  of each cluster based on the datapoints it
% contains

% Step 5: Reassign each datapoint to the new closest centroid

% Choosing the k value, search for WCSS. "The Elbow Method"


data = readtable('Datasets\Mall_Customers.csv');
