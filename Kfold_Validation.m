%---------------------K-Nearest Neighbour Algorithm------------------------

% 1- Choose the number of k neighbours.

% 2- Compute the k neighbors of the new data point according to some
% distance measure such as Euclidean.

% 3- Count the number of data points from each category among the
% neighbours computed in step 2.

% 4- The new datapoint is assigned to the category whit most neighbours.

data = readtable('Datasets\Social_Network_Ads.csv');

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
classification_model.NumNeighbors = 5; %Change number of neighbours.
%classification_model.NumNeighbors = 3; % Change the neighbor number to get better result
%classification_model = fitcknn(standardized_data, 'Survived~Age+Fare+Parch+SibSp+female+male+Pclass','NumNeighbors', 3);
%classification_model = fitcknn(standardized_data, 'Survived~Age+Fare+Parch+SibSp+female+male+Pclass','Distance', 'seuclidean');


%-----------Divide data into training and testing sets
%Numberof observations, classification models, percentage. Randomly choose
%0.2 percentage for testing.

%Validation with Holdout and Kfold
%cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
cv = cvpartition(classification_model.NumObservations,'KFold', 5); %Produce 5 classifiers
cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

%-----------Make predictions for the each testing set
Predictions_K1 = predict(cross_validated_model.Trained{1}, data(test(cv,1),1:end-1));
Predictions_K2 = predict(cross_validated_model.Trained{1}, data(test(cv,2),1:end-1));
Predictions_K3 = predict(cross_validated_model.Trained{1}, data(test(cv,3),1:end-1));
Predictions_K4 = predict(cross_validated_model.Trained{1}, data(test(cv,4),1:end-1));
Predictions_K5 = predict(cross_validated_model.Trained{1}, data(test(cv,5),1:end-1));

%-----------Analyzing the predictions
    %Confusion Matrix for each test results
Results_K1 = confusionmat(cross_validated_model.Y(test(cv,1)),Predictions_K1); 
Results_K2 = confusionmat(cross_validated_model.Y(test(cv,2)),Predictions_K2); 
Results_K3 = confusionmat(cross_validated_model.Y(test(cv,3)),Predictions_K3); 
Results_K4 = confusionmat(cross_validated_model.Y(test(cv,4)),Predictions_K4); 
Results_K5 = confusionmat(cross_validated_model.Y(test(cv,5)),Predictions_K5); 
%data.Purchased(test(cv));

%Combine results in  one confusion matrix
Results = Results_K1 + Results_K2 + Results_K3 + Results_K4 + Results_K5
