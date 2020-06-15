clear;
% Import the dataset
dataread1 = readtable ('C:\Users\Asus\Desktop\necessary\Data_1.csv');
dataread2 = readtable ('C:\Users\Asus\Desktop\necessary\Data_2.csv');
dataread3 = readtable ('C:\Users\Asus\Desktop\necessary\Data_3.csv');
dataread4 = readtable ('C:\Users\Asus\Desktop\necessary\Data_4.csv');
% Data preprocessing

% Handling missing values
    % Method 1: Remove column or row with missing values
        % Remove entries which has missing valued
        removed1 = rmmissing(dataread1);
        % Remove columns which has at least one missing value
        removed2 = rmmissing(dataread1,2);
        % Remove rows which has at lesat two missing values
        removed3 = rmmissing(dataread2, 'MinNumMissing', 2);
    
    % Method 2: Use mean value and replace with missing value
        mean_age = mean(dataread1.Age, 'omitnan');
        filled_age = fillmissing(dataread1.Age, 'constant', mean_age);
        filleddata = dataread1 ;
        filleddata.Age = filled_age;
    
    % Method 3: Dealing with non-numerical data
        dataread3.Opinion = categorical(dataread3.Opinion); %Change to categorical type to use function
        most_frequent_opinion = mode(dataread3.Opinion);
        filled_opinion = fillmissing(dataread3.Opinion, 'constant',cellstr(most_frequent_opinion));
        dataread3.Opinion = filled_opinion;
       
% Feature Scalling
    % Method 1: Standardization
    stand_age = (dataread4.Age - mean(dataread4.Age)) / std(dataread4.Age);
    % dataread4.Age = stand_age;
    
    % Method 2: Normalization
    normalize_age = (dataread4.Age - min(dataread4.Age)) / (max(dataread4.Age) - min(dataread4.Age));
    %dataread4 = normalize_age;
    