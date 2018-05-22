X_train = dlmread('../DatosAgrupados/xtrain.csv');
y_train = dlmread('../DatosAgrupados/ytrain.csv');

X_val = dlmread('../DatosAgrupados/xval.csv');
y_val = dlmread('../DatosAgrupados/yval.csv');

X_test = dlmread('../DatosAgrupados/xtest.csv');
y_test = dlmread('../DatosAgrupados/ytest.csv');

## REGRESION LOGISTICA
X_train = [ones(length(y_train), 1), X_train];
X_val = [ones(length(y_val), 1), X_val];
X_test = [ones(length(y_test), 1), X_test];
theta_inicial = zeros(columns(X_train), 1);

opciones = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, cost] = fminunc(@(t)(costeLog(t, X_train, y_train)), theta_inicial, opciones);
fprintf("Coste regresion logistica sin regularizar: %f\n", cost);

## CURVAS DE APRENDIZAJE
m = rows(X_train)/200;
%m = 100;
for i = 1 : m
	row = 1:min(i*200,rows(X_train));
	%row = 1:i;
	cost = @(t) costeLogReg(t, X_train(row, :), y_train(row, :), 0);
    theta = fminunc(cost, theta_inicial, opciones);
    % error
    J(i) = costeLogReg(theta, X_train(row, :), y_train(row, :), 0);
    Jval(i) = costeLogReg(theta, X_test, y_test, 0);
endfor

plot(1:m, J, 1:m, Jval);
legend("Train", "Val");
axis([0 12 0 150]);
pause;

accu = accuracyLog(theta, X_test, y_test);
fprintf("Accuracy regresion logistica sin regularizar: %f\n", accu);
fprintf("\n");

%## REGRESION LOGISTICA REGULARIZADA MEJOR LAMBDA
values = [0.01, 0.1, 0.2, 0.3, 1, 3];
lambda = 0;

maxAccu = 0;
bestlambda = 0;
auxTheta = theta_inicial;
auxCost = 0;

for i = 1:columns(values)

	lambda = values(i);
	[auxTheta, auxCost] = fminunc(@(t)(costeLogReg(t, X_train, y_train, lambda)), theta_inicial, opciones);
	accu = accuracyLog(auxTheta, X_val, y_val)
	Jreg(i) = costeLogReg(auxTheta, X_train(2:end, :), y_train(2:end, :), lambda);
	Jvalreg(i) = costeLogReg(auxTheta, X_test, y_test, lambda);

	if accu > maxAccu
		maxAccu = accu;
		bestlambda = lambda;
		theta = auxTheta;
		cost = auxCost;
	end
end

func = @(t)(costeLogReg(t, X_train, y_train, bestlambda));
[theta, cost] = fminunc(func, theta_inicial, opciones);

accu = accuracyLog(theta, X_test, y_test);
fprintf("Mejor lambda: %f\n", bestlambda);
fprintf("Accuracy regresion logistica reg: %f\n", accu);