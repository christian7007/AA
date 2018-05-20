X_train = dlmread('../datos/xtrain.csv');
y_train = dlmread('../datos/ytrain.csv');

X_val = dlmread('../datos/xval.csv');
y_val = dlmread('../datos/yval.csv');

X_test = dlmread('../datos/xtest.csv');
y_test = dlmread('../datos/ytest.csv');

%% Eleccion de los parametros C y sigma
values = [0.01, 0.1, 1];
maxAccu = 0;
auxC = 0;
auxSigma = 0;
for i = 1:columns(values)
	sigma = values(i);
	for j = 1:columns(values)
		C = values(j);
		model = svmTrain(X_train, y_train, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, X_val);
		accu = 1 - sum(abs(pred - y_val)) / length(y_val);
		if accu > maxAccu
			maxAccu = accu;
			auxC = C;
			auxSigma = sigma;
		end
	end
	fprintf("Test: C %f sigma %f accu %f\n", C, sigma, accu);
end

fprintf("Los mejores son maxAccu -> %f, C -> %f, sigma -> %f\n", maxAccu, auxC, auxSigma);
model = svmTrain(X_train, y_train, auxC, @(x1, x2) gaussianKernel(x1, x2, auxSigma));
pred = svmPredict(model, X_val);
accu = 1 - sum(abs(pred - y_val)) / length(y_val);
fprintf("Accuracy %f with C = %f and sigma = %f\n", accu, auxC, auxSigma);
