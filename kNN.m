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


% apply knn
k_vals = [1:2:101];
error_rates = zeros(length(k_vals), 1);

for i = 1: length(k_vals)
    k = k_vals(i);
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
    error = loocv(X_train, y_train, k);
    fprintf('the training error is %.4f\n', error);
    error_rates(i) = error; 
end % End for

k_vals = [1:2:101];
error_rates = [0.365634156550858;0.334344914190038;0.310799497697782;0.302846379238175;0.302951025533696;0.296044370029301;0.294370029300963;0.290916701548765;0.288091251569694;0.289974884889075;0.285475094181666;0.284323984930933;0.279300962745919;0.280452071996651;0.280766010883215;0.278254499790707;0.278463792381750;0.280138133110088;0.281498534951863;0.280033486814567;0.280766010883215;0.278045207199665;0.277312683131017;0.274801172038510;0.274277940560904;0.273859355378820;0.273545416492256;0.275115110925073;0.275010464629552;0.274382586856425;0.274591879447468;0.273859355378820;0.274068647969862;0.274696525742989;0.274068647969862;0.273231477605693;0.273126831310172;0.272603599832566;0.272603599832566;0.273022185014651;0.272917538719129;0.272708246128087;0.273650062787777;0.272603599832566;0.273022185014651;0.273440770196735;0.272708246128087;0.272812892423608;0.272708246128087;0.273126831310172;0.273231477605693];
majority = 2607/9557*ones(length(k_vals), 1);
%%% Plot example
hold on;
plot( k_vals, error_rates, 'r');
plot(k_vals, majority, 'b');
% line plot example random data
ylabel('error rate');
xlabel('k');
title('K-NN on Chicago Food Inspections');


% Leave-one-out cross validation
function error = loocv(X_train, y_train, k)
    n = size(X_train, 1);
    errors = zeros(n, 1);
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
        pred = mode(y_train(idx_sorted(1:k)));

        decision(i) = pred;
    end
end
