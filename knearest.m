%---------------------K-Nearest Neighbour Algorithm------------------------

% 1- Choose the number of k neighbours.

% 2- Compute the k neighbors of the new data point according to some
% distance measure such as Euclidean.

% 3- Count the number of data points from each category among the
% neighbours computed in step 2.

% 4- The new datapoint is assigned to the category whit most neighbours.

data = readtable('C:\Users\Asus\Desktop\necessary\K-Nearest Neighbor\Social_Network_Ads.csv');

%--------Preprocess
sum(ismissing(data)); % Count of missing values in columns

% Check for the outliers with plotting
% plot(data.Age);
% plot(data.EstimatedSalary)

% Feature Scaling with Standardization
stand_age = (data.Age - mean(data.Age)) / std(data.Age);
data.Age = stand_age;

stand_estimatedsalary = (data.EstimatedSalary - mean(data.EstimatedSalary)) / std(data.EstimatedSalary);
data.EstimatedSalary = stand_estimatedsalary;

%-----------Classifying data
classification_model = fitcknn(data, 'Purchased~Age+EstimatedSalary'); %Classification Model

%-----------Divide data into training and testing sets
%Numberof observations, classification models, percentage. Randomly choose
%0.2 percentage for testing.

cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

%%-----------Make predictions for the testing set
predict(cross_validated_model.Trained{1}, test(cv));

