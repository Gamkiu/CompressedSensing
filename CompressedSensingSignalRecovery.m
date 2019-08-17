clear all; close all;
%% generate the original signal
%-Using trigonometric functions to generate discrete signals in the frequency domain or DCT domain-
n = 512;
t = [0: n-1];
f = cos(2*pi/256*t) + sin(2*pi/128*t);   % generate sparse signal
%-------------------------------�źŽ�������-----------------------
n = length(f);
a = 0.2;            %    ȡԭ�źŵ� a%
m = double(int32(a*n));
%--------------------------------------��ԭʼ�ź�ͼ--------------------------------------

ft = dct(f);
disp('ft = dct(f)')
disp(['�ź�ϡ��ȣ�',num2str(length(find((abs(ft))>0.1)))])
figure('name', 'ԭʼ�ź�ͼ');
subplot(2, 1, 1);
plot(f);
title("ԭʼ�ź�ʱ���ʾ")
xlabel('Time (s)'); 
ylabel('f(t)');
subplot(2, 1, 2);
plot(ft);
title("ԭʼ�ź�Ƶ���ʾ")
xlabel('Frequency (Hz)'); 
ylabel('DCT(f(t))');

%% Generating a perceptual matrix and a sparse representation matrix
%--------------------------���ø�֪�������ɲ���ֵ---------------------

Phi = PartHadamardMtx(m,n);       % ��֪���󣨲�������    ���ֹ��������
f2 = (Phi * f')';                 % ͨ����֪�����ò���ֵ
Psi = dct(eye(n,n));            %��ɢ���ұ任������ �������дΪPsi = dctmtx(n);
disp('Psi = dct(eye(n,n));')

A = Phi * Psi;                    % �ָ����� A = Phi * Psi
%%             reconstruct signal
%---------------------ʹ��CVX�������L1������Сֵ-----------------
cvx_begin;
    variable x(n) complex;
    minimize(norm(x,1));
    subject to
      A*x == f2' ;
cvx_end;
figure('name','ʹ��CVX���߻ָ��õ����ź�ͼ');
subplot(2,1,2)
plot(real(x));
title('Using L1 Norm��Frequency Domain��with cvx');
ylabel('DCT(f(t))'); xlabel('Frequency (Hz)');
sig1 = dct(real(x));
disp('sig1 = dct(real(x))')
subplot(2,1,1);
plot(f);
hold on;plot(sig1);hold off
title('Using L1 Norm (Time Domain) with cvx');
ylabel('f(t)'); xlabel('Time (s)');
legend('Original','Recovery')
optimal1=0.5*(norm(A*x - f2'))^2 + 0.2*norm(x,1);  %ͨ��cvx������õ�Ŀ�꺯������ֵ

% %-----------------------------ʹ��proximal�㷨��L1���������Ż����-----------------
[x,k,opt]=proximal(A,f2',0.2); %kΪ����������xΪ�ָ����źţ�opt��Ŀ�꺯������ֵ
disp('Proximal�㷨���е����Ĵ���Ϊ��')
disp(k);
disp('Ŀ�꺯������ֵ��')
disp(opt);
figure('name','ʹ��proximal�㷨�õ��Ļָ��ź�');
subplot(2,1,2);
plot(real(x));
disp('plot(real(x))')
title('Using L1 Norm��Frequency Domain��with proximal algorithm');
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
figure('name','�������');
plot(calc_error);
