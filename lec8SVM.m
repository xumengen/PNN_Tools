function lec8SVM(x, y)

fprintf('-----lec8 SVM -----\n');

syms lmb1 lmb2 lmb3 lmb4 lmb5 lmb6 w0
syms x1 x2 x3 x4
lmb = [lmb1, lmb2, lmb3, lmb4, lmb5, lmb6];
lmb = lmb(1: size(y,2));  % 删除不需要的lambda
sym_x = [x1, x2, x3, x4];
sym_x = sym_x(1: size(x, 1)).';  % 确认x的属性共有几个

w = x * (lmb .* y).';  %2x1矩阵，对w求导公式
% w2 = lmb1*y(1)*x(:,1) + lmb2*y(2)*x(:,2) + lmb3*y(3)*x(:,3) + lmb4*y(4)*x(:,4) 
% hyperplane = w' * x + w0;  % 1x4矩阵
y_wxw0_1 = y .* (w.' * x + w0) - 1;  % 1x4矩阵，4个方程
lmb_y = sum(lmb .* y);  % 1个方程

% 解方程
% [lmb1, lmb2, lmb3, lmb4, w0] = solve([y_wxw0_1==0, lmb_y]==0)
solution = solve([y_wxw0_1==0, lmb_y]==0);

% 将结构体内部syms改为double
fileds = fieldnames(solution);
allsyms = cell(1, length(fileds));
allvalues = zeros(1, length(fileds));
for i=1:length(fileds)
    k = fileds(i);
    key = k{1};  % 变量名称
    solution.(key) = double( solution.(key) );  %变量值改为double型
    allsyms{i} = key;
    allvalues(i) = solution.(key);
end
hyperplane = w.' * sym_x + w0;  % 1x4矩阵
hyperplane_after = subs(hyperplane, allsyms, allvalues);


% 输出结果
fprintf('lmbda共有：\n');disp(lmb);
fprintf('w方程为：\n');disp(w);
fprintf('y(wx+w0)-1方程为：\n');disp(y_wxw0_1.');
fprintf('Σlmbda*y方程为：\n');disp(lmb_y);
fprintf('解方程变量结果为：\n');disp(solution);
fprintf('hyperplane超平面为：\n');disp(vpa(hyperplane_after==0));

end