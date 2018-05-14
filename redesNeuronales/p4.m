load('ex4data1.mat');
load('ex4weights.mat')

% costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)
num_entradas = 400;
num_ocultas = 25;
num_etiquetas = length(unique(y))
lambda = 1;
params_rn_ini = [Theta1(:); Theta2(:)];

[J grad] = costeRN(params_rn_ini, num_entradas, num_ocultas, num_etiquetas, X, y, lambda);
fprintf("Coste: %f\n", J);

checkNNGradients(lambda);

Theta1 = pesosAleatorios(num_entradas, num_ocultas);
Theta2 = pesosAleatorios(num_ocultas, num_etiquetas);
params_rn_ini = [Theta1(:); Theta2(:)];
% lambda = 1
options = optimset('MaxIter', 250);
cost = @(t) costeRN(t, num_entradas, num_ocultas, num_etiquetas, X, y, lambda);
[params_rn, J] = fmincg(cost, params_rn_ini, options);
Theta11 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
Theta21 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));
accu = accuracyNN(Theta11, Theta21, X, y)
fprintf("Accuracy with lambda = 1: %f\n", accu);
