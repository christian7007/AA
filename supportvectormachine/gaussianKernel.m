function sim = gaussianKernel(x1, x2, sigma)

	sim = exp(-(abs(x1-x2) ^ 2) / ((sigma ^ 2) * 2));

end
