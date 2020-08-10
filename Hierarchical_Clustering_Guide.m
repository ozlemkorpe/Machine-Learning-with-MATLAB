%%%------------- Hierarchical Clustering
% Step 1: Consider each data point as a single cluster.
% Step 2: Combine the two closest clusters and make them one cluster.
% Step 3: Repeatedly combine clusters until there is only one cluster.

% Approaches for finding closest clusters
% Single Link: Min distance
% Complete Link: Max distance
% Average: Average distance

% Types of Hierachical Clustering : Agglomerative, Divisive

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