function [x,k,opt] = proximal(A,b,gamma)
% %����proximal�㷨�����ѹ����֪����
% %   ����Ĳ�������ΪA������ֵΪb,gamma�����źŵ�ϡ��̶�
% %   A��m*n b:n*1  gamma:1*1
% %   ��������Ϊ��
% %   min(0.5*(norm(A*x-b)^2)+gamma*norm(x))
% %       f(x)=0.5*(norm(A*x-b)^2)
% %       g(x)=gamma*norm(x)
% %   �������̣�
% %      z^k=x^k-gamma^k*gradient_g(x^k);
% %      x^(k+1)=prox_gamma(gamma^k*h)(z^k)
% %   ����ֵ������
% %      prox_f(x)=sign(x)*max{|x|-gamma��0}
% %
% error=1e-20;  %���ôﵽ����ֵ����������
% MAX_ITER =40000;    %���ó������ĵ�������
% [~,n]=size(A);  %���ؾ���A������������
% x = zeros(n,1);  %x�ĳ�ʼ������ѡȡ��
% AtA=A'*A;
% Atb=A'*b;
% f = @(u) 0.5*norm(A*u-b)^2;%%Ϊ��ȷ������������
% lambda =1;  % s���õĹ̶�����Ϊ1
% beta = 0.25;  %����������0.25 
% for k = 1:MAX_ITER
%         grad_x = AtA*x - Atb;  %f(x)���ݶ�
%         % gamma��ʾ����ֵ��������ֵ���Ǹ�ֵ��
%         % z��Ҫ���µĵ��Ƶ���ʽ��xk 
%         z = soft_threshold(x - grad_x,gamma);%%��������x�� % �������õ���z�ܴ󡣡�
%         %��������������ע�͵���Ϊ�̶�������
% %         if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*lambda))*(norm(z - x))^2
% %             break;
% %         end
% %         lambda = beta*lambda;   %��С����
%     x = z;
%     h.prox_optval(k) = log(objective(A, b, gamma, x));  %��k�������õ���Ŀ�꺯��ֵ
%     if (k>1) && (abs(h.prox_optval(k) - h.prox_optval(k-1)) < error)
%         break;   %���Ŀ�꺯�����������ε����õ��ĺ�����ֵС������ֹͣ������???ֹͣ׼��ò�������⡣
%     end
% end
% opt=h.prox_optval(end);
% disp(k);
% figure('name','Ŀ�꺯��ֵ�ı仯');
% plot(log(h.prox_optval))   %Ŀ�꺯��ֵ���ŵ��������ı仯
% title("Ŀ�꺯��ֵ���ŵ��������ı仯");
% xlabel("��������k");
% ylabel("Ŀ�꺯��ֵ(log)");
% h.x_prox = x;
% h.p_prox = h.prox_optval(end);
% end
% 
% function p = objective(A, b, gamma, x)
% %Ŀ�꺯��
%   p = 0.5*(norm(A*x - b))^2 + gamma*norm(x,1);
% end
% function X=soft_threshold(b ,gamma) 
% % ����ֵ����
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
[~,n]=size(A);  %���ؾ���A��������
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
