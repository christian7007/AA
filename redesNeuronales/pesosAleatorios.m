function W = pesosAleatorios(L_in, L_out)
    
    epsilon_ini = sqrt(6) / sqrt(L_in + L_out);
    
    W = [2*epsilon_ini * rand(L_out, 1 + L_in) - epsilon_ini];
end