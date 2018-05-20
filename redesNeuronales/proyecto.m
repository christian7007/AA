X_train = dlmread('../datos/xtrain.csv');
y_train = dlmread('../datos/ytrain.csv');

X_val = dlmread('../datos/xval.csv');
y_val = dlmread('../datos/yval.csv');

X_test = dlmread('../datos/xtest.csv');
y_test = dlmread('../datos/ytest.csv');

y_train = y_train.+1;
y_val = y_val.+1;
y_test = y_test.+1;

num_entradas = columns(X_train);
num_ocultas = 5;
num_etiquetas = length(unique(y_train));

lambda = 1;

Theta1 = pesosAleatorios(num_entradas, num_ocultas);
Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);

params_rn_ini = [Theta1(:); Theta2(:)];
options = optimset('MaxIter', 250);

values = [0.01, 0.1, 0.2, 0.3, 1, 3, 0];
lambda = 0;

maxAccu = 0;
bestlambda = 0;
auxCost = 0;

%for i = 1:columns(values)
%	lambda = values(i);
%	cost = @(t) costeRN(t, num_entradas, num_ocultas, num_etiquetas, X_train, y_train, lambda);
%    [params_rn, J] = fmincg(cost, params_rn_ini, options);
%	Theta11 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
%    Theta21 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));
%    accu = accuracyNN(Theta11, Theta21, X_val, y_val);
%	if accu > maxAccu
%		maxAccu = accu;
%		bestlambda = lambda;
%		cost = auxCost;
%	end
%end

%m = rows(X_train)/100;
m = 100;
for i = 1 : m
	%row = 1:min(i*100,rows(X_train));
	row = 1:i;
    cost = @(t) costeRN(t, num_entradas, num_ocultas, num_etiquetas, X_train(row, :), y_train(row, :), lambda);
    [params_rn, J] = fmincg(cost, params_rn_ini, options);
    % error
    J_train(i) = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X_train(row, :), y_train(row, :), lambda);
    Jval(i) = costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X_val, y_val, lambda);
endfor

plot(1:m, J_train, 1:m, Jval);
legend("Train", "Val");
axis([0 12 0 150]);
pause;

cost = @(t) costeRN(t, num_entradas, num_ocultas, num_etiquetas, X_train, y_train, bestlambda);
[params_rn, J] = fmincg(cost, params_rn_ini, options);
Theta11 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta21 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));
accu = accuracyNN(Theta11, Theta21, X_test, y_test);
fprintf("Accuracy %f with lambda = %f\n", accu, bestlambda);