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
        
%----------------Check for Outliers
plot(filled_data.Age) %Age varies between 0-80 which can be accepted as normal

%----------------Feature Scaling with Standardization Method
%Feature scaling for the Age
stand_age = (filled_data.Age - mean(filled_data.Age)) / std(filled_data.Age);
%Feature scaling for the Fare
stand_fare = (filled_data.Fare - mean(filled_data.Fare)) / std(filled_data.Fare);

%----------------Classification with K-Nearest Neighbours Algorithm
classification_model = fitcknn(filled_data, 'Survived~Age+Fare'); %Classification Model

%----------------Partitioning
cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

%----------------Prediction
Predictions = predict(cross_validated_model.Trained{1}, filled_data(test(cv),1:end-1));

%----------------Analyzing the Result
%Confusion Matrix: / diagonal will give the false predictions, \ will
    %be the rigth predictions.
Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions); 