function projectfinal

%Import data from Titanic
[n,t,r] = xlsread('titanic3.xls');
sexmap = containers.Map({'female', 'male'},[1,2]);
columnstokeep = [1,2,4,5,6,7,9];
splitpercentage = 0.5;
headers = r(1,:);headers(2) = [];
data = r(2:end,:);
rows = size(data,1);

%Discretize Sex column
for j = 1:rows
    data{j,4} = sexmap(data{j,4});
end

%Fill missing Fare
for i = 1:rows
    if(isnan(data{i,9}))
        data{i,9} = 33.2955;
    end
end

%Build data matrix with only 'columnstokeep' columns
tdata = data(:,columnstokeep);
tdata = tdata(randperm(rows),:);
tdata = cell2mat(tdata);

%divide into training and test parts 50%/50%
setsize = int32(rows*splitpercentage)
training_features = tdata(1:setsize,:);
training_labels = int32(training_features(:,2));
training_features(:,2) = [];
test_features = tdata((setsize+1):rows,:);
test_labels = int32(test_features(:,2));
test_features(:,2) = [];

%Build a model based on some classifier
model = fitcsvm(training_features,training_labels);

%Predict classes
ypred = predict(model,test_features);

accu = calcaccuracy(test_labels,ypred)
end

function accu = calcaccuracy(correct, predict)
total = length(correct);
cpred = 0;
for i = 1:length(correct)
    if(correct(i) == round(predict(i)))
        cpred = cpred + 1;
    end
end
accu = cpred/total;
end