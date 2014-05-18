function [result] = NDCG(P,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P is the ranking provided
% Y is the binary matrix annotating the labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n, m] = size(P);
num_rel = sum(Y,2);
[void, idx] = sort(P, 2, 'descend');
result = 0;
denominator = [1 1./log2(2:m)];

for i=1:n,
	DCG = sum(Y(i,idx(i,:)) .* denominator);
	IDCG = sum([1 1./log2(2:num_rel(i))]);
	result = result + ((DCG / IDCG) / n);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
