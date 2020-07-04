%Ensembles Guide

% Generate different datasets from one dataset. Use seperated datasets with
% different classifiers. Combine all classifiers to obtain a super
% classifier.

%---------------Importing Dataset
data = readtable('C:\Users\Asus\Desktop\necessary\Naive Bayes\Social_Network_Ads.csv');

%---------------Feature Scaling (Standardization Method)
stand_age = (data.Age - mean(data.Age))/std(data.Age);
data.Age = stand_age; 

stand_estimted_salary = (data.EstimatedSalary - mean(data.EstimatedSalary))/std(data.EstimatedSalary);
data.EstimatedSalary = stand_estimted_salary; 

%---------------Classifying Data  
classification_model = fitcensemble(data,'Purchased~Age+EstimatedSalary');

%--------------- Customization for classifier

%---------------Partitioning
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);
cross_validated_model = crossval(classification_model,'cvpartition',cv); 

%---------------Predictions
Predictions = predict(cross_validated_model.Trained{1},data(test(cv),1:end-1));

%---------------Analyzing the Results
Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions);

%---------------Visualizing Training Results
labels = unique(data.Purchased);
classifier_name = 'Ensemble (Training Results)';

Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;

[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
 
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
 
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
 
hold on
 
training_data = data(training(cv),:);
Y = ismember(training_data.Purchased,labels{1});
 
scatter(training_data.Age(Y),training_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Age(~Y),training_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
xlabel('Age');
ylabel('Estimated Salary');
 
title(classifier_name);
legend off, axis tight
 
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
 
%---------------Visualizing Test Results 
labels = unique(data.Purchased);
classifier_name = 'Ensemble (Testing Results)';

Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;

[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];

predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);

figure

gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');

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
 
