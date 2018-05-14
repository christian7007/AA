function [X, y] = readEmails(vocabulario, folder, spam, length, nWords)

    X = zeros(length, nWords);
    y = zeros (length, 1);
    
    for i=1: length
      file_contents = readFile(sprintf("%s/%04d.txt", "easy_ham", 24));
      email = processEmail(file_contents);
      
      while ~isempty(email)
          [str, email] = strtok(email, [' ']);
          if(isfield(vocabulario, str) > 0)
              codEmail(i, vocabulario.(str)) = 1;
          endif 
      endwhile
      
      y(i) = spam;
      
    endfor

end