function [X_train, X_test] = tfidf(X_train, X_test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Calculates the idf on the train set and performs tf-idf normalization of
% both matrices. Also does L2 normalization.
%
% tf-idf = tf * log(|D| / n_occurences)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idf = log(size(X_train,1) ./ (sum(X_train>0) + eps));
IDF = spdiags(idf', 0, size(idf,2), size(idf,2));
X_train = X_train * IDF;
X_train = L2_norm_row(X_train);

X_test = X_test * IDF;
X_test = L2_norm_row(X_test);


function Xnorm = L2_norm_row(X)
  Xnorm = spdiags(1 ./ (sqrt(sum(X.*X,2)) + eps), 0, size(X,1), size(X,1)) * X;
end

end
