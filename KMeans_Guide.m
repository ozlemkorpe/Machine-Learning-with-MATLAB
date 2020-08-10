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

% Apply Elbow method to find right number of clusters
WCSS = [];
for k=1:10
    sumd=0;
    [idx,C,sumd] = kmeans(arrayed_data,k);
    WCSS(k) = sum(sumd);
end

plot(1:10, WCSS);

%------- Perform k means
%idx tell us which object belongs to which cluster, c is centroid point of
%each cluster
[idx,C] = kmeans(arrayed_data,6);

%------- Visualization
data = arrayed_data;
figure,

gscatter(data(:,1),data(:,2),idx);
hold on

for i=1:6
    scatter(C(i,1),C(i,2),96,'black','filled');
end

legend({'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5','Cluster 6' })
xlabel('Annual Income');
ylabel('Spending Score');
hold off
