% setup data
D_tr = readtable('Food_Inspections.xls');

% Group by restaurant - if any failed, label is failed
% id, lat, lon, failed