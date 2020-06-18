%This version produces the right result around %72 and wrong result around %28

%----------------Import Data
titanic_train = readtable('C:\Users\Asus\Desktop\Kaggle_Titanic_Competition\titanic_train.csv');
% titanic_test = readtable('C:\Users\Asus\Desktop\Kaggle_Titanic_Competition\titanic_test.csv');

%----------------Missing data check
titanic_train_missing = sum(ismissing(titanic_train));
% titanic_test_missing = sum(ismissing(titanic_test));

%----------------Handling Missing Values
%For age column which has 6 missing values use mean value and replace with missing value
        mean_age = cast(mean(titanic_train.Age, 'omitnan'),'uint8') ; %Cast double to integer
        filled_age = fillmissing(titanic_train.Age, 'constant', mean_age);
        filled_data = titanic_train ;
        filled_data.Age = filled_age;
       
        % Test if the any missing value left
        test_for_filled_data =  sum(ismissing(filled_data));
        
%------ Dealing with Categorical Data without order relation
        filled_data = categorical_data_to_dummy_variables(filled_data, filled_data.Sex); %Seperate genders into different columns
        filled_data.Sex = [];
        

        
%----------------Check for Outliers
%plot(filled_data.Age) %Age varies between 0-80 which can be accepted as normal

%----------------Feature Scaling with Standardization Method
standardized_data = filled_data ;
%Feature scaling for the Age
stand_age = (filled_data.Age - mean(filled_data.Age)) / std(filled_data.Age);
standardized_data.Age = stand_age;
%Feature scaling for the Fare
stand_fare = (filled_data.Fare - mean(filled_data.Fare)) / std(filled_data.Fare);
standardized_data.Fare = stand_fare;
%Feature scaling for the SibSp
stand_sibspb = (filled_data.SibSp - mean(filled_data.SibSp)) / std(filled_data.SibSp);
standardized_data.SibSp = stand_sibspb;
%Feature scaling for the Parch
stand_parch = (filled_data.Parch - mean(filled_data.Parch)) / std(filled_data.Parch);
standardized_data.Parch = stand_parch;

%----------------Classification with K-Nearest Neighbours Algorithm
classification_model = fitcknn(standardized_data, 'Survived~Age+Fare+Parch+SibSp+female+male'); %Classification Model

%----------------Partitioning
cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

%----------------Prediction
Predictions = predict(cross_validated_model.Trained{1}, standardized_data(test(cv),1:end-1));

%----------------Analyzing the Result
%Confusion Matrix: / diagonal will give the false predictions, \ will
    %be the rigth predictions.
Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SCRIPTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function to handle categorical data which does not have order relation:
    function data = categorical_data_to_dummy_variables(data,variable)
unique_values = unique(variable);
for i=1:length(unique_values)
    dummy_variable(:,i) = double(ismember(variable,unique_values{i})) ;
end 
T = table;
[rows, col] = size(dummy_variable);
for i=1:col
    T1 = table(dummy_variable(:,i));
    T1.Properties.VariableNames = unique_values(i);
    T = [T T1];
end 
    data = [T data]; 
    end

