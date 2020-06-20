%The version with normalization method

clear
%----------------Import Data
titanic_train = readtable('C:\Users\Asus\Desktop\Kaggle_Titanic_Competition\titanic_train.csv');
% titanic_test = readtable('C:\Users\Asus\Desktop\Kaggle_Titanic_Competition\titanic_test.csv');

%----------------Missing data check
titanic_train_missing = sum(ismissing(titanic_train));
% titanic_test_missing = sum(ismissing(titanic_test));

%----------------Handling Missing Values
%For age column which has 6 missing values use mean value and replace with missing value
%filled_data = rmmissing(titanic_train);
        mean_age = cast(mean(titanic_train.Age, 'omitnan'),'uint8') ; %Cast double to integer
        filled_age = fillmissing(titanic_train.Age, 'constant', mean_age);
        filled_data = titanic_train ;
        filled_data.Age = filled_age;
 %For Fare column which has 6 missing values use mean value and replace with missing value        
        mean_fare = cast(mean(titanic_train.Fare, 'omitnan'),'uint8') ; %Cast double to integer
        filled_fare = fillmissing(titanic_train.Fare, 'constant', mean_fare);
        filled_data.Fare = filled_fare;
       
        % Test if the any missing value left
        test_for_filled_data =  sum(ismissing(filled_data));
        
%------ Dealing with Categorical Data without order relation
        filled_data = categorical_data_to_dummy_variables(filled_data, filled_data.Sex); %Seperate genders into different columns
        filled_data.Sex = [];
        

        
%----------------Check for Outliers
%plot(filled_data.Age) %Age varies between 0-80 which can be accepted as normal

%----------------Feature Scaling with Standardization Method
normalized_data = filled_data ;
%Feature scaling for the Age
normalized_age = (filled_data.Age - min(filled_data.Age)) / (max(filled_data.Age) - min(filled_data.Age));
normalized_data.Age = normalized_age;
%Feature scaling for the Fare
normalized_fare = (filled_data.Fare - min(filled_data.Fare)) / (max(filled_data.Fare) - min(filled_data.Fare));
normalized_data.Fare= normalized_fare;
%Feature scaling for the SibSp
normalized_sibsp = (filled_data.SibSp - min(filled_data.SibSp)) / (max(filled_data.SibSp) - min(filled_data.SibSp));
normalized_data.SibSp = normalized_sibsp;
%Feature scaling for the Parch
normalized_parch = (filled_data.Parch - min(filled_data.Parch)) / (max(filled_data.Parch) - min(filled_data.Parch));
normalized_data.Parch = normalized_parch;
%Feature scaling fot the Pclass
normalized_pclass = (filled_data.Pclass - min(filled_data.Pclass)) / (max(filled_data.Pclass) - min(filled_data.Pclass));
normalized_data.Pclass = normalized_pclass;

%----------------Classification with K-Nearest Neighbours Algorithm
classification_model = fitcknn(normalized_data, 'Survived~Age+Fare+Parch+SibSp+female+male+Pclass'); %Classification Model

%----------------FOR LOOP FOR CALCULATING ACCURACY IN A NUMBER OF EXECUTION
general_accuracy = 0;
for a = 1:100
 %----------------Partitioning
cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

%----------------Prediction
Predictions = predict(cross_validated_model.Trained{1}, normalized_data(test(cv),1:end-1));

%----------------Analyzing the Result
%Confusion Matrix: / diagonal will give the false predictions, \ will
    %be the rigth predictions.
Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions); 
right_results = Results(1,1) + Results(2,2);
wrong_results = Results(1,2) + Results(2,1);

truth_score = right_results /(right_results + wrong_results);

general_accuracy = general_accuracy + truth_score;
end

general_accuracy = general_accuracy / a; 


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
