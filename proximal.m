function [x,k,opt] = proximal(A,b,gamma)
% %利用proximal算法，求解压缩感知问题
% %   输入的测量矩阵为A，测量值为b,gamma控制信号的稀疏程度
% %   A：m*n b:n*1  gamma:1*1
% %   求解的问题为：
% %   min(0.5*(norm(A*x-b)^2)+gamma*norm(x))
% %       f(x)=0.5*(norm(A*x-b)^2)
% %       g(x)=gamma*norm(x)
% %   迭代过程：
% %      z^k=x^k-gamma^k*gradient_g(x^k);
% %      x^(k+1)=prox_gamma(gamma^k*h)(z^k)
% %   软阈值函数：
% %      prox_f(x)=sign(x)*max{|x|-gamma，0}
% %
% error=1e-20;  %设置达到最优值是允许的误差
% MAX_ITER =40000;    %设置程序最大的迭代次数
% [~,n]=size(A);  %返回矩阵A的行数和列数
% x = zeros(n,1);  %x的初始点怎样选取？
% AtA=A'*A;
% Atb=A'*b;
% f = @(u) 0.5*norm(A*u-b)^2;%%为了确定线搜索步长
% lambda =1;  % s设置的固定步长为1
% beta = 0.25;  %线搜索步长0.25 
% for k = 1:MAX_ITER
%         grad_x = AtA*x - Atb;  %f(x)的梯度
%         % gamma表示软阈值函数的阈值（非负值）
%         % z是要更新的递推迭代式的xk 
%         z = soft_threshold(x - grad_x,gamma);%%迭代更新x， % 这里计算得到的z很大。。
%         %线性搜索步长（注释掉即为固定步长）
% %         if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*lambda))*(norm(z - x))^2
% %             break;
% %         end
% %         lambda = beta*lambda;   %减小步长
%     x = z;
%     h.prox_optval(k) = log(objective(A, b, gamma, x));  %第k步迭代得到的目标函数值
%     if (k>1) && (abs(h.prox_optval(k) - h.prox_optval(k-1)) < error)
%         break;   %如果目标函数的最新两次迭代得到的函数差值小于误差，则停止迭代。???停止准则貌似有问题。
%     end
% end
% opt=h.prox_optval(end);
% disp(k);
% figure('name','目标函数值的变化');
% plot(log(h.prox_optval))   %目标函数值随着迭代步数的变化
% title("目标函数值随着迭代步数的变化");
% xlabel("迭代步数k");
% ylabel("目标函数值(log)");
% h.x_prox = x;
% h.p_prox = h.prox_optval(end);
% end
% 
% function p = objective(A, b, gamma, x)
% %目标函数
%   p = 0.5*(norm(A*x - b))^2 + gamma*norm(x,1);
% end
% function X=soft_threshold(b ,gamma) 
% % 软阈值函数
%   X=sign(b).*max(abs(b) - gamma,0);
% end
MAX_ITER = 4000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
tic

% cvx_begin quiet
%     cvx_precision low
%     variable x(n)
%     minimize(0.5*sum_square(A*x - b) + gamma*norm(x,1))
% cvx_end
% 
% h.x_cvx = x;
% h.p_cvx = cvx_optval;
% h.cvx_toc = toc;

f = @(u) 0.5*sum_square(A*u-b);
lambda = 1;
beta = 0.5;

tic;
[~,n]=size(A);  %返回矩阵A的行数和
x = zeros(n,1);
xprev = x;
AtA=A'*A;
Atb=A'*b;
for k = 1:MAX_ITER
    while 1
        grad_x = AtA*x - Atb;
        z = prox_l1(x - lambda*grad_x,gamma);
        if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*lambda))*sum_square(z - x)
            break;
        end
        lambda = beta*lambda;
    end
    xprev = x;
    x = z;

    h.prox_optval(k) = objective(A, b, gamma, x, x);
    if k > 1 && abs(h.prox_optval(k) - h.prox_optval(k-1)) < ABSTOL
        break;
    end
end

h.x_prox = x;
h.p_prox = h.prox_optval(end);
h.prox_grad_toc = toc;

function p = objective(A, b, gamma, x, z)
    p = 0.5*sum_square(A*x - b) + gamma*norm(z,1);
end
opt=h.prox_optval(end);

end
