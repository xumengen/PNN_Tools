function lec9ENSEMBLE(x, y, h)

fprintf('-----lec9 ENSEMBLE METHOD -----\n');
epoch = 3;
[xrow, xnum] = size(x);  % xrow = attribute, xcol = 共有多少个点
hnum = size(h,2);  % 共有多少个hx
syms hx1 hx2 hx3 hx4 hx5 hx6 hx7 hx8
hx = [hx1, hx2, hx3, hx4, hx5, hx6, hx7, hx8];
classifier = 0;

w = 1./xnum .* ones(1, xnum);  % 每个点的权重，特殊需求可改此处

for epo = 1 : epoch
    fprintf('---- epoch %d ----\n', epo);
    train_error = ( (h<0) * w.').';  % 把每行求和，变为8列，每列为错误率。变列为行向量。
    [sort_error, sort_index] = sort(train_error);
    index = sort_index(1);  % 本次选择的hx
    if sort_error(1) > 0.5
        fprintf('---- NO WAY!!!!! ------\n'); disp(train_error);
        break;
    end
    e1 = train_error(index);  % 一波西龙
    aerfa = 1/2*log((1-e1)/e1);  % 阿尔法
    aerfa_y_h = aerfa .* h(index,:);  %  .* y，已经发现误分类了，所以不用乘y
    w_e_ayh = w .* exp(-1 .* aerfa_y_h);  % 权重更新，尚未归一化
    z = sum(w_e_ayh);
    w_new = w_e_ayh ./ z;  % 新权重，已归一化
    w = w_new;
    classifier = classifier + roundn(aerfa, -4) .* hx(index);
    
    % 输出结果
    fprintf('train_error为：\n'); disp(train_error);
    fprintf('本次选择的hx为：hx %d\n', index);
    fprintf('一波西龙ε = %f, 阿尔法α = %f\n', e1, aerfa); 
    fprintf('未归一化权重：\n'); disp(w_e_ayh);
    fprintf('归一化权重：\n'); disp(w_new);
    fprintf('classifier分类器为：\n'); disp(vpa(classifier));  % ensemble classifier is sgn(...)
    
    
end



%{
 输出结果
fprintf('lmbda共有：\n');disp(lmb);
fprintf('w方程为：\n');disp(w);
fprintf('y(wx+w0)-1方程为：\n');disp(y_wxw0_1.');
fprintf('Σlmbda*y方程为：\n');disp(lmb_y);
fprintf('解方程变量结果为：\n');disp(solution);
fprintf('hyperplane超平面为：\n');disp(hyperplane_after==0);
%}

end
