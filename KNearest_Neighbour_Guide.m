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
classification_model.NumNeighbours = 5; %Change number of neighbours.
%classification_model.NumNeighbors = 3; % Change the neighbor number to get better result
%classification_model = fitcknn(standardized_data, 'Survived~Age+Fare+Parch+SibSp+female+male+Pclass','NumNeighbors', 3);
%classification_model = fitcknn(standardized_data, 'Survived~Age+Fare+Parch+SibSp+female+male+Pclass','Distance', 'seuclidean');


%-----------Divide data into training and testing sets
%Numberof observations, classification models, percentage. Randomly choose
%0.2 percentage for testing.

cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

%-----------Make predictions for the testing set
Predictions = predict(cross_validated_model.Trained{1}, data(test(cv),1:end-1));

%-----------Analyzing the predictions
    %Confusion Matrix: / diagonal will give the false predictions, \ will
    %be the rigth predictions.
Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions); 
%data.Purchased(test(cv));

%-----------Showing Results in Graphical View

labels = unique(data.Purchased); %labels contains unique classes of Purchased
classifier_name = 'K-Nearest Neigbor (Testing Results)';

%Set ranges for age and EstimatedSalary
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1; 
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;

[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range); %2d grid for age range and estimated salary range
XGrid = [xx1(:) xx2(:)]; %Create grid

predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);

gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb'); %group values

hold on

testing_data =  data(test(cv),:);
Y = ismember(testing_data.Purchased,labels{1});
 
scatter(testing_data.Age(Y),testing_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(testing_data.Age(~Y),testing_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');


xlabel('Age');
ylabel('Estimated Salary');

title(classifier_name);
legend off, axis tight

legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');




