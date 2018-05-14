load('ex6data1.mat');
% plot original data
X
pause;
plotData(X, y);
pause;

% plot with SVM con C = 1
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
pause;

% plot with SVM con C = 100
C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
pause;

load('ex6data2.mat');
plotData(X, y);
pause;

% Kernel gausiano
C = 1;
sigma = 0.1;
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
pause;
% Eleccion de los parametros C y sigma
load('ex6data3.mat');
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
maxAccu = 0;
auxC = 0;
auxSigma = 0;
for i = 1:columns(values)
	sigma = values(i)
	for j = 1:columns(values)
		C = values(j)
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
		accu = 1 - sum(abs(pred - yval)) / length(yval);
		if accu > maxAccu
			maxAccu = accu;
			auxC = C;
			auxSigma = sigma;
		end
	end
end
fprintf("Los mejores son maxAccu -> %f, C -> %f, sigma -> %f\n", maxAccu, auxC, auxSigma);
model = svmTrain(X, y, auxC, @(x1, x2) gaussianKernel(x1, x2, auxSigma));
model = svmTrain(X, y, 1, @(x1, x2) gaussianKernel(x1, x2, 0.1));
visualizeBoundary(X, y, model);
pause;
