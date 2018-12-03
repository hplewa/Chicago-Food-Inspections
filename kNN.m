% setup data
% DBA name, Result (0 Pass, 1 Fail), Latitude, Longitude, Location (Lat, Lon)
D = readtable('Food_Inspections_Grouped.xls');

% Lati
X_train = table2array(D(:, 3:4));
y_train = table2array(D(:, 2));
size_train = size(y_train,1);

% setup meshgrid
[x1, x2] = meshgrid(41.649:0.001:42.021, -87.914:0.001:-87.525);
grid_size = size(x1);
X12 = [x1(:) x2(:)];

% compute 1NN decision 
n_X12 = size(X12, 1);
decision = zeros(n_X12, 1);

% apply knn on noised dataset
k = 3;
decision = knn_predict(X12, X_train, y_train, k);

% plot decisions in the grid
decisionmap = reshape(decision, grid_size);
imagesc(41.649:0.001:42.021, -87.914:0.001:-87.525, decisionmap);
set(gca,'ydir','normal');

% colormap for the classes
% class 1 = light red, 0 = light green
cmap = [0.8 1 0.8; 1 0.8 0.8;];
colormap(cmap);

% scatter plot data
hold on;
scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 10, 'r');
scatter(X_train(y_train == 0, 1), X_train(y_train == 0, 2), 10, 'g');
hold off;

% training error (LOOCV)
error = loocv(X_train, y_train);
fprintf('the training error is %.2f\n', error);

% Leave-one-out cross validation
function error = loocv(X_train, y_train)
    n = size(X_train, 1);
    errors = zeros(n, 1);
    k = 1;
    for i = 1:n
        ypred = knn_predict(X_train(i, :), X_train(1:end ~= i, :), y_train(1:end ~= i), k);
        errors(i) = (ypred ~= y_train(i));
    end
    error = sum(errors)/n; 
end

% KNN function
function decision = knn_predict(X_test, X_train, y_train, k)
    n_X_test = size(X_test, 1);
    decision = zeros(n_X_test, 1);
    for i=1:n_X_test
        point = X_test(i, :);

        % compute euclidan distance from the point to all training data
        dist = pdist2(X_train, point);

        % sort the distance, get the index
        [~, idx_sorted] = sort(dist);

        % find the most frequent class among the k nearest neighbour
        pred = mode( y_train(idx_sorted(1:k)) );

        decision(i) = pred;
    end
end