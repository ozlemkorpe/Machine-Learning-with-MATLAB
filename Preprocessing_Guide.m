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
    
% Handling Outliers (Aykýrý Deðerler)
    dataread5 = readtable ('C:\Users\Asus\Desktop\necessary\Data_5.csv');
    
    % Method 1: Deleting Rows
    outliers = isoutlier(dataread5.Age); % 10X1 array consist of 1s and 0s. 1 is where outlier is
    % Calculates the median and decides 3 times greater or lesser of the median as outlier
    
    outlierdeleted_dataread5 = dataread5(~outliers, :); % convert 1s to 0s, 0s to 1. 0 is outlier now. And remove.
    
    % Detect and fill the value of outlier with mean or median.
    % Age = filloutlier(dataread5.Age,'center') %Fill the outliers with center value
    
    Age = filloutliers(dataread5.Age,'clip');
    %If the value is lower then the lower threshold matlab will fill the
    %value with the lower threshold, else if it is greater than the upper
    %threshold matlab fill it with the upper threshold
    
   
    %------ Dealing with Categorical Data without order relation
    datacategorical = categorical_data_to_dummy_variables(dataread5, dataread5.Location);
    datacategorical.Location = [];
    
    % Categorical data must not have space character.
    
    %----- Dealing with Categorical Data with order relation
    
  dataread6 = readtable ('C:\Users\Asus\Desktop\necessary\Data_6.csv');
  new_variable = encoding_categorical_data(dataread6.YearlyIncome,{'Low','Average','High','Very High'}, [1 2 3 5]) ;
  encodeddata6 = dataread6;
  encodeddata6.YearlyIncome = new_variable ;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAVE PROCESSED DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
writetable(encodeddata6, 'C:\Users\Asus\Desktop\saved_data.csv');

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

    %Function to handle categorical data which have an order relation
function new_variable = encoding_categorical_data(variable,values_set,numbers) 
[rows,col] = size(variable);
new_variable = zeros(rows,1); 
for i=1:length(values_set)
    indices = ismember(variable,values_set{i});
    new_variable(indices) = numbers(i);
end 
end 