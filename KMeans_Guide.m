%%%------------- K-Means Clustering
% Step 1: Choose the number k of clusters

% Step 2: Select k point at random called centroids

% Step 3: Assign each datapoint to the nearest centroid which lead to K
% clusters

% Step 4: Compute new centroids  of each cluster based on the datapoints it
% contains

% Step 5: Reassign each datapoint to the new closest centroid

% Choosing the k value, search for WCSS. "The Elbow Method"

% Import the dataset
data = readtable('Datasets\Mall_Customers.csv');

%Check for missing values
missings = sum(ismissing(data));

%Plot variables to check for outliers
IncomePlot = plot(data.AnnualIncome);
SpendingPlot = plot(data.SpendingScore);

% Perform Feature Scaling (Standardization Method)
stand_income = (data.AnnualIncome - mean(data.AnnualIncome)) / std(data.AnnualIncome);
data.AnnualIncome = stand_income; 

stand_spending = (data.SpendingScore - mean(data.SpendingScore)) / std(data.SpendingScore);
data.SpendingScore = stand_spending; 

% Select columns for clustering
selected_data = data(:,4:5);

%Data must be an array to be used in clustering algorithm
arrayed_data = table2array(selected_data);
