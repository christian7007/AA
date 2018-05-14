data = dlmread('../datoslimpios.csv');

X = data(:, 2:columns(data) - 1);
y = data(:, columns(data));

## REGRESION LOGISTICA
negativos = find(y == 0);
positivos = find(y == 1);

X = [ones(length(y), 1), X];
theta_inicial = zeros(columns(X), 1);

opciones = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, cost] = fminunc(@(t)(costeLog(t, X, y)), theta_inicial, opciones);
fprintf("Coste regresion logistica: %f\n", cost);

accu = accuracyLog(theta, X, y);
fprintf("Accuracy regresion logistica: %f\n", accu);
fprintf("\n");

## REGRESION LOGISTICA REGULARIZADA
lambda = 0.00001;
[theta, cost] = fminunc(@(t)(costeLogReg(t, X, y, lambda)), theta_inicial, opciones);
fprintf("Coste regresion logistica reg: %f\n", cost);

accu = accuracyLog(theta, X, y);
fprintf("Accuracy regresion logistica reg: %f\n", accu);

