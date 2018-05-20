X_train = dlmread('../datos/xtrain.csv');
y_train = dlmread('../datos/ytrain.csv');

X_val = dlmread('../datos/xval.csv');
y_val = dlmread('../datos/yval.csv');

X_test = dlmread('../datos/xtest.csv');
y_test = dlmread('../datos/ytest.csv');

## REGRESION LOGISTICA
X_train = [ones(length(y_train), 1), X_train];
X_val = [ones(length(y_val), 1), X_val];
X_test = [ones(length(y_test), 1), X_test];
theta_inicial = zeros(columns(X_train), 1);

opciones = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, cost] = fminunc(@(t)(costeLog(t, X_train, y_train)), theta_inicial, opciones);
fprintf("Coste regresion logistica: %f\n", cost);

m = rows(X_train)/100;
for i = 1 : m
	row = 1:min(i*100,rows(X_train));
	cost = @(t) costeLogReg(t, X_train(row, :), y_train(row, :), 0);
    theta = fminunc(cost, theta_inicial, opciones);
    % error
    J(i) = costeLogReg(theta, X_train(row, :), y_train(row, :), 0);
    Jval(i) = costeLogReg(theta, X_val, y_val, 0);
endfor

plot(1:m, J, 1:m, Jval);
axis([0 12 0 150]);
pause;

accu = accuracyLog(theta, X_test, y_test);
fprintf("Accuracy regresion logistica: %f\n", accu);
fprintf("\n");

## REGRESION LOGISTICA REGULARIZADA
values = [0.01, 0.1, 0.2, 0.3, 1, 3];
lambda = 0;

maxAccu = 0;
bestlambda = 0;
auxTheta = theta_inicial;
auxCost = 0;

for i = 1:columns(values)
	lambda = values(i);
	[auxTheta, auxCost] = fminunc(@(t)(costeLogReg(t, X_train, y_train, lambda)), theta_inicial, opciones);
	accu = accuracyLog(auxTheta, X_val, y_val);
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

