%% *****************************************************************************************************************************
% paper: Learning and Controlling Multi-scale Dynamics in Spiking Neural Networks using Recursive Least Square Modifications
% author: Liyuan Han et. al.
% Uploading Time: 2023.09.07
% code availability: https://github.com/LiyuanHan/multiscale-SNN
% file: IZ_task1_sine_popularity.m, This code is for point-to-point control task.
% remark 1: This code is adapted from https://www.nature.com/articles/s41467-017-01827-3.
% remark 2: If you cite this paper, please also cite the paper https://www.nature.com/articles/s41467-017-01827-3.
% *******************************************************************************************************************************
%% Forward process with Izhikevich Network
clear all
close all
clc
%%
T = 5000; %Total time in ms
dt = 0.04; %Integration time step in ms
nt = round(T/dt); %Time steps
N = 1000; %Number of neurons

%% Izhikevich Parameters
C = 250;  %capacitance
vr = -60;   %resting membrane
b = -2;  %resonance parameter
ff = 2.5;  %k parameter for Izhikefvich, gain on v
vpeak = 30;  % peak voltage
vreset = -65; % reset voltage
vt = vr+40-(b/ff); %threshold  %threshold
u_ada = zeros(N,1);  %initialize adaptation
a = 0.01; %adaptation reciprocal time constant
d = 200; %adaptation jump current
tr = 2;  %synaptic rise time
td = 20; %decay time
pp = 0.08; %sparsity
G = 7.5*1000; 
%% parameters in control
rng(2)
n = 1;
m = n;
n_cho = 5;
direc = sort(randperm(N,n_cho))

% input matrix
B = zeros(N,n_cho);
bb = direc;
% bb = b_temp(1:n);
for q = 1:length(bb)
    B(bb(q),q) = 1;
end

% initial weight matrix A0*G
A0 = G*randn(N,N).* (rand(N,N)<pp)/(pp*sqrt(N));
for i = 1:1:N 
    QS = find(abs(A0(i,:))>0);
    A0(i,QS) = A0(i,QS) - sum(A0(i,QS))/length(QS);
end

T_temp = T/1000;

% initial state
x0 = 0*rand(N,1);
% control target
x_target = zeros(N,1);
x_target(direc) = rand(n,1)*50;
x_f=x_target;
[save_x, u_ctr] = x0_xT_DP_True(A0/G,B,T/1000,x0,x_f,direc);

save_x_avg = sum(save_x(direc,:),1)/length(direc)
% cftool
u_ctr_avg = sum(u_ctr,1)/length(direc)



%%
sav_x_sd = std(save_x(direc,:),1);
u_ctr_sd = std(u_ctr,1);

sav_x_var = var(save_x(direc,:),1);
u_ctr_var = var(u_ctr,1);
x_target(direc)

figure(101)
errorbar(0:T/1000,save_x_avg,sav_x_sd,'-ok','linewidth',1.5)
grid on
xlabel('Time (s)')
ylabel('State')

figure(102)
errorbar(0:T/1000-1,u_ctr_avg,u_ctr_sd,'-ok','linewidth',1.5)
grid on
xlabel('Time (s)')
ylabel('Control')

% var
% figure(103)
% errorbar(0:T/1000,save_x_avg,sav_x_var,'-ok','linewidth',1.5)
% grid on
% xlabel('Time (s)')
% ylabel('State')
% 
% figure(104)
% errorbar(0:T/1000-1,u_ctr_avg,u_ctr_var,'-ok','linewidth',1.5)
% grid on
% xlabel('Time (s)')
% ylabel('Control')
%% Target signal  
period_l = 5;
% len_mar = 0.001;
delta = [];
zx_temp = [];
for i = 1:T_temp
    delta(:,i) = (save_x_avg(:,i+1)-save_x_avg(:,i))/(1000/dt);
    for j = 1:size(save_x_avg,1)
        zx_temp(j,:) = sin(2*period_l*pi*1/(abs(save_x_avg(:,i+1)-save_x_avg(:,i)))*(save_x_avg(j,i)+delta(j,i):delta(j,i):save_x_avg(j,i+1)));
%         zx_temp(j,:) = sin(2*period_l*pi*(save_x_avg(j,i)+delta(j,i):delta(j,i):save_x_avg(j,i+1)));
%         zx_temp(j,:) = sin(2*period_l*pi*1/len_mar*(save_x(j,i)+delta(j,i):delta(j,i):save_x(j,i+1)));
    end
    figure(12)
    plot(dt*((i-1)*1000/dt+1:1:i*1000/dt)/1000,zx_temp,'LineWidth',2)
    xlim([0,dt*nt/1000]);ylim([-1.5,1.8]);
    hold on
    zx(:,(i-1)*1000/dt+1:i*1000/dt) = zx_temp;
end

%% rectangular wave
% zx = zeros(1,125000);
% for i = 1:2:9
%     zx(:,(i-1)*12500+1:12500*i) = 1;
%     zx(:,12500*i+1:12500*(i+1)) = -1;
% end

%% Storage variables for synapse integration
IPSC = zeros(N,1); %post synaptic current
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

%-----Initialization---------------------------------------------
% rng(2)
v = vr+(vpeak-vr)*rand(N,1); %initial distribution
v_ = v; %These are just used for Euler integration, previous time step storage
%% 
k = size(zx,1); %used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.
BPhi = zeros(N,1); %initial decoder--save most important k-dimentional vectors of weight A.  Best to keep it at 0.

tspike = zeros(5*nt,2);  %If you want to store spike times,
ns = 0; %count toal number of spikes
BIAS = 1000; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons

%%
% Pinv = eye(N)*2; %initial correlation matrix, coefficient is the regularization constant as well
step = 1/dt; %optimize with RLS only every 1ms
imin = round(1000/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round((T-500)/dt); %end simulation at this time step

current = zeros(nt,1);  %store the approximant
RECB = zeros(nt,5); %store the decoders
REC = zeros(nt,2*5); %Store voltage and adaptation variables for plotting

%%
err = zeros(N,1);
P = eye(N)*2; %initial correlation matrix, coefficient is the regularization constant as well
%% Simulation
direc = direc(1);
x_appro = zeros(1,1);
A0_temp = A0(:,direc);
A = A0/(G);
sA = [];
s_v_max = zeros(2,nt);
s_fir_rate = zeros(N,nt);
save_err = [];
for j = 1:1:nt  
    j
    I = IPSC + A0_temp*x_appro +BIAS;
    v = v + dt*(( ff.*(v-vr).*(v-vt) - u_ada + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
    u_ada = u_ada + dt*(a*(b*(v_-vr)-u_ada)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    
    %%
    index = find(v>=vpeak);
%     len_index(j) = length(index);
    if length(index)>0
        JD = sum(G*A(:,index),2); %compute the increase in current due to spiking
        %          tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];  %uncomment this
        tspike(ns+1:ns+length(index),:) = [index,0*index+dt*j];  %uncomment this
        %if you want to store spike times.  Takes longer.
        ns = ns + length(index);
    end
    
    %synapse for double exponential
    IPSC = IPSC*exp(-dt/td) + h*dt;
    h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td);  %Integrate the current
    
    r = r*exp(-dt/td) + hr*dt;
    hr = hr*exp(-dt/tr) + (v>=vpeak)/(tr*td);
    
    s_fir_rate(:,j) = r;

    
    %% Update process
    x_appro = BPhi'*r;%dimention:kX1
    Tem_err = x_appro - zx(:,j);
    %% RLS steps
    if mod(j,step) == 1
        if j>imin
            save_err(1,round(j*dt)+1) = BPhi'*r - zx(:,j);
%             if j<icrit
                    
                A(:,direc) = A(:,direc) - P*(r)*Tem_err';
                P = P - ( P*r*(P*r)' )/( 1+r'*P*r );
                BPhi = A(:,direc);
                
%             end
            save_err(2,round(j*dt)+1) = BPhi'*r - zx(:,j);
        end
    end
    
    
    
    
    %% Store, and plot.
    u_ada = u_ada + d*(v>=vpeak);  
    v = v+(vreset-v).*(v>=vpeak); 
    v_ = v; 
    REC(j,:) = [v(459:463)',u_ada(459:463)'];
    current(j,:) = mean(x_appro'); % save the state value from 0:T-1
    RECB(j,:)= BPhi(459:463);
    
end

%%
disp('Plot')
%% Plotting neurons before and after learning
figure(30)
for j = 459:1:463
    plot((1:1:nt)*dt/(1000),REC(1:1:nt,j-458)/(vpeak-vreset)+j), hold on
end
hold on 
figure(30)
plot([1,1],[458,464],'Color',[0.5 0.5 0.5],'LineWidth',2)
hold on
plot([4.5,4.5],[458,464],'Color',[0.5 0.5 0.5],'LineWidth',2)
hold on
% plot([4,4],[0,6],'Color',[0.5 0.5 0.5],'LineWidth',2)
xlim([0,T/1000])
xlabel('Time (s)');  ylabel('Index')

% figure(31)
% for j = 1:1:5
%     plot((1:1:nt)*dt/(1000/Factor),REC(1:1:nt,j)/(vpeak-vreset)+j), hold on
% end
% xlim([0,imin*dt/(1000)]) 
% xlabel('Time (s)'); ylabel('Neuron Index')
% title('Pre-Learning')


figure(12)
hold on
for i = 1:T_temp
    hold on
    plot(dt*((i-1)*round(1000/dt)+1:1:i*round(1000/dt))/1000,current((i-1)*round(1000/dt)+1:1:i*round(1000/dt),:),'b-.','LineWidth',2)
end
hold on
plot([1,1],[-1.5,1.5],'Color',[0.5 0.5 0.5],'LineWidth',2)
hold on
plot([4.5,4.5],[-1.5,1.5],'Color',[0.5 0.5 0.5],'LineWidth',2)
ylim([-1.5,1.5])
xlabel('Time (s)');ylabel('$\hat{X}_t$','Interpreter','LaTex');
legend('0-1 s','1-2 s','2-3 s','3-4 s','4-5 s','Appro.')

figure(2)
plot(save_err(1,:))
hold on
plot(save_err(2,:))
xlabel('Time (ms)')
ylabel('Pre-and post error with RLS')
legend('$\tilde{e}_{-}$','$\tilde{e}_{+}$','Interpreter','LaTex')
grid on
save('task-sine_popularity.mat','s_fir_rate','zx','N','T','dt','pp','G','A0','direc','period_l','REC','RECB','tspike','s_spike','x0','x_target','x_f')


s_fir_rate_2 = [];
flag =1;
for i = 1:nt
    if mod(i,25)==0
        s_fir_rate_2(:,flag) = sum(s_fir_rate(:,(flag-1)*25+1:flag*25),2);
        flag = flag+1;
    end
end

figure(135)
x = 1:5000;
y = 455:550;
% x = 3500:4500;
% y=459:463;
% y = 1:N;
[X,Y]=meshgrid(x,y);
surf(X,Y,s_fir_rate_2(y,x))
map = addcolorplus(309);
colormap(map)
colorbar
shading flat
view(0,90)
axis tight
grid off
xlim([0,x(end)])
ylim([y(1),y(end)])
% ylim([455,550])
xlabel('Time (ms)')
ylabel('Index')

hold on % 3500:4500
axes('position',[0.1 0.7 0.09*5 0.016*5]);
x2 = 3500:4500;
y2 = 459:463;
[X,Y]=meshgrid(x2,y2);
surf(X,Y,s_fir_rate_2(y2,x2))
map = addcolorplus(309);
colormap(map)
% colorbar
shading flat
view(0,90)
axis tight
grid off
xlim([x2(1),x2(end)])
ylim([y2(1),y2(end)])

hold on % 500:1500;
axes('position',[0.1 0.3 0.09*5 0.016*5]);
x3 = 500:1500;
y3 = 459:463;
[X,Y]=meshgrid(x3,y3);
surf(X,Y,s_fir_rate_2(y3,x3))
map = addcolorplus(309);
colormap(map)
% colorbar
shading flat
view(0,90)
axis tight
grid off
xlim([x3(1),x3(end)])
ylim([y3(1),y3(end)])

