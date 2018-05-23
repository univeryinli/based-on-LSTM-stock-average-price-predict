clc
clear all

% N=7; % number of days to calculate average (odd)
% n=(N+1)/2;
% length=200; % number of days
% data=load('path'); % load data
% % MA middle line
% MA=zeros(length-N,1);
% for i=1:length-N+1
%     MA(i)=sum(data(i:i+N-1))/N;
% end
% % up_line & down_line
% UL=zeros(length-N,1);
% DL=zeros(length-N,1);
% for i=1:length-N+1
%     aver=sum((data(i:i+N-1)-MA(i))^2)/N;
%     MD=sqrt(aver);
%     UL(i)=MA(i)+2*MD;
%     DL(i)=MA(i)-2*MD;
% end
% 
% %select sample
% theta=0.5;  %threshold value
% num=1;
% sample_length=7;
% patch=zeros(0,sample_length,1);
% label=zeros(0,1);
% for i=1:length-N+1
%     if UL(i)-DL(i)<=theta&&i>=sample_length
%         patch(num,:,1)=data(i-sample_length:i);
%         num=num+1;
%     end
% end
% save 'path' patch
M = csvread('D:\data\econo\50ETF__data\50ETF__data\1_XSHG.601318.h5.csv',1,1);
length=371040;
N=50;
patch=zeros(18522,128,2);
value=zeros(18522,1);
k=1;
for i=1:20:length-600
    A=(log(M(i+1:i+128,4)./M(i:i+127,4))).^2;
    patch(k,:,1)=smooth(A,10)*10e6;
    %patch(k,:,1)=patch(k,:,1)/max(patch(k,:,1));
    %plot(patch(k,:,1));
    patch(k,:,2)=smooth(A,20)*10e6;
    %plot(patch(k,:,2))
    %patch(k,:,6)=M(i:i+49,6)/10e3;
    log_rate=zeros(11,1);
    n=1;
    for m=i+129:i+138
        open=M(m-1,4);
        close=M(m,4);
        log_rate(n)=(log(close/open))^2;
        n=n+1;
    end
    vt=sum(log_rate)/11;  
    value(k,1)=vt;
    disp(k)
    k=k+1;
end

%value(41371:41374)=2.48992*10e-6;
value=value*10e6;
for i=1:18522
    if value(i)>=200
        value(i)=200;
    end
end
value=smooth(value,100);

plot(value)
save C:\Users\Charlie\Desktop\input.mat patch
save C:\Users\Charlie\Desktop\output.mat value






