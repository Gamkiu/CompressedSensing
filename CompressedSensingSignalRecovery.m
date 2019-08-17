clear all; close all;
%% generate the original signal
%-Using trigonometric functions to generate discrete signals in the frequency domain or DCT domain-
n = 512;
t = [0: n-1];
f = cos(2*pi/256*t) + sin(2*pi/128*t);   % generate sparse signal
%-------------------------------信号降采样率-----------------------
n = length(f);
a = 0.2;            %    取原信号的 a%
m = double(int32(a*n));
%--------------------------------------画原始信号图--------------------------------------

ft = dct(f);
disp('ft = dct(f)')
disp(['信号稀疏度：',num2str(length(find((abs(ft))>0.1)))])
figure('name', '原始信号图');
subplot(2, 1, 1);
plot(f);
title("原始信号时域表示")
xlabel('Time (s)'); 
ylabel('f(t)');
subplot(2, 1, 2);
plot(ft);
title("原始信号频域表示")
xlabel('Frequency (Hz)'); 
ylabel('DCT(f(t))');

%% Generating a perceptual matrix and a sparse representation matrix
%--------------------------利用感知矩阵生成测量值---------------------

Phi = PartHadamardMtx(m,n);       % 感知矩阵（测量矩阵）    部分哈达玛矩阵
f2 = (Phi * f')';                 % 通过感知矩阵获得测量值
Psi = dct(eye(n,n));            %离散余弦变换正交基 代码亦可写为Psi = dctmtx(n);
disp('Psi = dct(eye(n,n));')

A = Phi * Psi;                    % 恢复矩阵 A = Phi * Psi
%%             reconstruct signal
%---------------------使用CVX工具求解L1范数最小值-----------------
cvx_begin;
    variable x(n) complex;
    minimize(norm(x,1));
    subject to
      A*x == f2' ;
cvx_end;
figure('name','使用CVX工具恢复得到的信号图');
subplot(2,1,2)
plot(real(x));
title('Using L1 Norm（Frequency Domain）with cvx');
ylabel('DCT(f(t))'); xlabel('Frequency (Hz)');
sig1 = dct(real(x));
disp('sig1 = dct(real(x))')
subplot(2,1,1);
plot(f);
hold on;plot(sig1);hold off
title('Using L1 Norm (Time Domain) with cvx');
ylabel('f(t)'); xlabel('Time (s)');
legend('Original','Recovery')
optimal1=0.5*(norm(A*x - f2'))^2 + 0.2*norm(x,1);  %通过cvx工具求得的目标函数最优值

% %-----------------------------使用proximal算法对L1范数问题优化求解-----------------
[x,k,opt]=proximal(A,f2',0.2); %k为迭代次数，x为恢复的信号，opt是目标函数最优值
disp('Proximal算法进行迭代的次数为：')
disp(k);
disp('目标函数最优值：')
disp(opt);
figure('name','使用proximal算法得到的恢复信号');
subplot(2,1,2);
plot(real(x));
disp('plot(real(x))')
title('Using L1 Norm（Frequency Domain）with proximal algorithm');
subplot(2,1,1);
sig2 = dct(real(x));
disp('sig2 = dct(real(x))')
subplot(2,1,1);
plot(f);hold on;
plot(sig2);
hold off
legend('Original','Recovery')
title('Using L1 Norm (Time Domain) with proximal algorithm');
calc_error=sig2-f';
figure('name','误差曲线');
plot(calc_error);
