data = dlmread('../datoslimpios.csv');

X = data(:, 2:columns(data) - 1);
y = data(:, columns(data));
y = y.+1;

num_entradas = columns(X);
num_ocultas = 5;
num_etiquetas = length(unique(y));

lambda = 1;

Theta1 = pesosAleatorios(num_entradas, num_ocultas);
Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);

params_rn_ini = [Theta1(:); Theta2(:)];
options = optimset('MaxIter', 500);

[J grad] = costeRN(params_rn_ini, num_entradas, num_ocultas, num_etiquetas, X, y, lambda);
fprintf("Coste: %f\n", J);

checkNNGradients(lambda);

cost = @(t) costeRN(t, num_entradas, num_ocultas, num_etiquetas, X, y, lambda);
[params_rn, J] = fmincg(cost, params_rn_ini, options);
Theta11 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta21 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));
accu = accuracyNN(Theta11, Theta21, X, y);
fprintf("Accuracy with lambda = %f: %f\n", lambda, accu);
