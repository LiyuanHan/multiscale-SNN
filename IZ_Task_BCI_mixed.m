%% *****************************************************************************************************************************
% paper: Learning and Controlling Multi-scale Dynamics in Spiking Neural Networks using Recursive Least Square Modifications
% author: Liyuan Han et. al.
% Uploading Time: 2023.09.07
% code availability: https://github.com/LiyuanHan/multiscale-SNN
% file: IZ_Task_BCI_mixed.m, This code is for mixed neurons in our SNN model.
% remark 1: This code is adapted from https://www.nature.com/articles/s41467-017-01827-3.
% remark 2: If you cite this paper, please also cite the paper https://www.nature.com/articles/s41467-017-01827-3.
% *******************************************************************************************************************************

%% Forward process with Izhikevich Network
clear all
close all
clc
% data = load('target.mat')
% zx = data.zx;
data = load('data_for_BCI_RLS.mat') % final saved result : task-lorenz_1e-4_T50.mat
data_t = linspace(0,48,length(data.position));
time_win = 1;
color_set = {[0,148,255]/255, [169,169,169]/255,...
             [0,141,0]/255, [169,169,169]/255,...
             [255,146,0]/255, [169,169,169]/255,...
             [207,185,158]/255, [169,169,169]/255,...
             [229,8,106]/255, [169,169,169]/255,...
             [80,29,138]/255, [169,169,169]/255,...
             [170,52,116]/255, [169,169,169]/255,...
             [238,140,125]/255, [169,169,169]/255};


%% 
T = 480; %Total time in ms
dt = 0.025*time_win; %Integration time step in ms
nt = round(T/dt); %Time steps

data.spike = data.spike(17:80,:);
plus_num = 32;
N = size(data.spike,1) + plus_num; %Number of neurons

n = N;
zx = zeros(n,length(data.position));
for i = 1:n
    if i<=n/2
        zx(i,:) = data.position(1,:);
%         zx(i,:) = data_postion(1,:);
    else
        zx(i,:) = data.position(2,:);
%         zx(i,:) = data_postion(2,:);
    end
end
% data.spike = data.spike([25:47,61:85],:);
%% Izhikevich Parameters
C = 100;  %capacitance
vr = -60;   %resting membrane
b = -2;  %resonance parameter
ff = 2.5;  %k parameter for Izhikefvich, gain on v
vpeak = 30;  % peak voltage
vreset = -65; % reset voltage
vt = vr+40-(b/ff); %threshold  %threshold
u_ada = zeros(N,1);  %initialize adaptation
a = 0.01; %adaptation reciprocal time constant
d = 200; %adaptation jump current
tr = 0.05;  %synaptic rise time
td = 10; %decay time
pp = 0.3; %sparsity
G = 5*10^2; 

epos = 3;
s_cur = zeros(nt,n);
s_spi = zeros(N,T*epos);
s_spi_each = zeros(N,nt);
s_fir = zeros(N,nt);

for epo = 1:epos
%Storage variables for synapse integration
IPSC = zeros(N,1); %post synaptic current
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

%-----Initialization---------------------------------------------
rng(epo)
% v = vr+(vpeak-vr)*rand(N,1); %initial distribution
num = randperm(N,1);
v = vr+(vreset-vr).*([data.spike(:,1); rand(plus_num,1)]);
v_ = v; %These are just used for Euler integration, previous time step storage
% 
%% parameters in control
% initial weight matrix A0
A0 = G*randn(N,N).* (rand(N,N)<pp)/(pp*sqrt(N));
for i = 1:1:N 
    QS = find(abs(A0(i,:))>0);
    A0(i,QS) = A0(i,QS) - sum(A0(i,QS))/length(QS);
end

direc = randperm(N,n);
A0_temp = A0(:,direc);

%%

k = size(zx,1); %used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.
BPhi = zeros(N,k); %initial decoder--save most important k-dimentional vectors of weight A.  Best to keep it at 0.

tspike = zeros(5*nt,2);  %If you want to store spike times,
ns = 0; %count toal number of spikes
BIAS = 10; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons
step = 1; %optimize with RLS only every 1ms
imin = round(16/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round(464/dt); %end simulation at this time step

current = zeros(nt,k);  %store the approximant
REC = zeros(nt,50*2); %Store voltage and adaptation variables for plotting


%%
err = zeros(N,1);
P = eye(N)*2; 
%% Simulation
xx = zeros(N,1);
x_appro = zeros(n,1);
flag = 1;
A = A0/G;
sA = [];
save_spike = zeros(N,1);
s_spike = zeros(n,nt);
s_spi_each_temp = zeros(N,nt);
s_fir_temp = zeros(N,nt);
for j = 1:1:nt  
    j
    I = IPSC + A0_temp*x_appro +BIAS;
    v = v + dt*(( ff.*(v-vr).*(v-vt) - u_ada + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
    u_ada = u_ada + dt*(a*(b*(v_-vr)-u_ada)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    
    %%
    index1 = find(data.spike(:,j)>0); % using BCI dataset
    index2 = find(v(N-plus_num+1:N) >= vpeak)+N-plus_num;
    index = [index1;index2];
    s_spike(index1,j) = data.spike(index1,j);
    s_spike(index2,j) = 1;
    if length(index)>0
        JD = sum(G*A(:,index).* [data.spike(index1,j);ones(length(index2),1)]',2);
    end
    
    %synapse for double exponential
    IPSC = IPSC*exp(-dt/td) + h*dt;
    h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td);  %Integrate the current
    
    r = r*exp(-dt/td) + hr*dt;
    hr = hr*exp(-dt/tr) + ([data.spike(:,j); (v(N-plus_num+1:N) >= vpeak)])/(tr*td);
    
    s_fir_temp(:,j) = r;
   
    
    %% Update process

    x_appro = BPhi'*r;%dimention:kX1
    Tem_err = x_appro - zx(:,j);
    %% RLS steps
    if mod(j,step) == 0
%         if j>imin
%             if j<icrit
                A(:,direc) = A(:,direc) - P*(r)*Tem_err';
                P = P - ( P*r*(P*r)' )/( 1+r'*P*r );
                BPhi = A(:,direc);
%             end
%         end
    end
    
    
    
    
    %% Store, and plot.
    u_ada = u_ada + d*([data.spike(:,j);(v(N-plus_num+1:N) >= vpeak)]);  %implements set u to u+d if v>vpeak, component by component.
    v = v+(vreset-v).*([data.spike(:,j);(v(N-plus_num+1:N) >= vpeak)]); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    v_ = v;  % sets v(t-1) = v for the next itteration of loop
    REC(j,:) = [v(1:50)', u_ada(1:50)'];
    current(j,:) = x_appro'; % save state value from 0:T-1


end

s_cur = (epo-1)/epo * s_cur + 1/epo * current;
s_fir = (epo-1)/epo * s_fir + 1/epo * s_fir_temp;
end
%%
disp('Plot')

%% Plotting neurons before and after learning
figure(30)
for j = 35:5:75
    id_temp = find(s_spike(direc(j),:)>0);
    id = zeros(1,3*length(id_temp));
    id(1:3:end) = id_temp; id(2:3:end) = id_temp; id(3:3:end) = NaN;
    
    if direc(j) > size(data.spike,1)
        s_spike(direc(j),id_temp) = s_spike(direc(j),id_temp) + 1;
    end
    y_val = zeros(1,3*length(id_temp));
    y_val(1:3:end) = j; y_val(2:3:end) = s_spike(direc(j),id_temp)+j; y_val(3:3:end) = NaN;
    hold on 
    plot([id,1:nt]*dt,[y_val,ones(1,nt)*j])
end
ylim([35,80])
xlabel('Time (s)');  ylabel('Neuron Index')
title('Mixed Model')

%%  Figures Approximant
fig_n = 2

current = s_cur;
c1 = mean(current(1:nt,1:n/2), 2);
c2 = mean(current(1:nt,n/2+1:n), 2);
zx1 = mean(zx(1:n/2, :));
zx2 = mean(zx(n/2+1:n, :));

%% Target & Approximant

% X - direction
figure(21)
subplot(2,1,1)
plot((1:1:nt)*dt,zx1(1:nt),'-','Color',[0.6627 0.6627 0.6627],'linewidth',1),hold on
plot((1:1:nt)*dt,c1(1:nt),'b-','linewidth',1),hold off
grid on
hold on
% % plot((nt:1:nt)*dt,current(nt:1:nt,1),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('Position-X')
ylim([-200,200])
legend('Target','Appro.','Orientation', 'horizontal')

% Y - direction
figure(21)
subplot(2,1,2)
plot((1:1:nt)*dt,zx2(1:nt),'-','Color',[0.6627 0.6627 0.6627],'linewidth',1),hold on
plot((1:1:nt)*dt,c2(1:nt),'b-','linewidth',1),hold off
grid on
hold on
% % plot((nt/2:1:nt)*dt,current(nt/2:1:nt,2),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('Position-Y')
ylim([-200,200])
legend('Target','Appro.','Orientation', 'horizontal')

R2_x = 1 - sum((zx1 - c1').^2)/sum((c1'-mean(zx1)).^2);
R2_y = 1 - sum((zx2 - c2').^2)/sum((c2'-mean(zx2)).^2);
R2_x
R2_y

%% [0 4, 2 4, 5 4, 7 4, 8 4, 6 4, 3 4, 1 4]
ref_spi_reange = zeros(N,nt);
IZ_fir_reange = zeros(N,nt);
 
% aaa = [];
id_flag = 0;
for k = 1:1:16
    for i = k:16:nt/40
        id_flag = id_flag + 1;
        IZ_fir_reange(:,(id_flag-1)*40+1:id_flag*40) = s_fir(:,(i-1)*40+1:i*40);
    end
end

%% max_abs_vals = max(abs(IZ_fir_reange), [], 2); % 每一行的最大绝对值
max_abs_vals = max(max(abs(IZ_fir_reange)));
IZ_fir_reange_normalized = IZ_fir_reange ./ max_abs_vals;
IZ_fir_reange_normalized2 = zeros(N,nt+16);

for i = 1:nt/1200
    IZ_fir_reange_normalized2(:,(i-1)*1200+i:i*1200+i-1) = IZ_fir_reange_normalized(:,(i-1)*1200+1:i*1200);
    IZ_fir_reange_normalized2(:,i*1200+i) = NaN; % NaN is just for line chart discontinuity in Matlab.
end

%% **** IZ_fir_normalized_hot_N.eps ***
figure(11)
pcolor((1:nt)*dt,1:N,IZ_fir_reange_normalized);
% map = addcolorplus(292); colormap(map);
shading interp
xlabel('Time (s)');ylabel('Neuron Index');
title('Heatmap of normalized firing rate of Mixed Neurons from 1 to 96')
%% **** IZ_fir_normalized_line_N.eps ****
figure(16)
flag = 1;
for i=35:5:75
    hold on
    plot((1:nt+16)*dt,IZ_fir_reange_normalized2(i,:)+flag,'Linewidth',2)
    flag = flag + 1;
end
xlabel('Time (s)'); ylabel('Neuron Index');
% ylim([61,72])

%% **** IZ_fir_normalized_point_N.eps ****
% % max_abs_vals = max(abs(s_fir), [], 2);
max_abs_vals = max(max(abs(s_fir)));
s_fir_normalized = s_fir ./ max_abs_vals;
figure(17)
flag = 1;
for i=35:5:75
    hold on
    plot((1:nt)*dt,s_fir_normalized(i,:)+flag,'Linewidth',2)
    flag = flag + 1;
end
xlabel('Time (s)');ylabel('Neuron Index');
% ylim([61,72])


%% **** fir_normalized_value_N_1-96.eps ****
figure(1)
% data:IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(1:30,1:40:1200);
% heatmap
imshow(data_temp, 'InitialMagnification', 'fit');
map = addcolorplus(292); colormap(map);
colorbar;
[num_rows, num_cols] = size(data_temp);
id_flag = 0;
% write text
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(data_temp(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xlabel('Time'); ylabel('Neuron Index 1 \sim 30');
title('Normalized firing rate of Mixed Nrueons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

figure(2)
% data:IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(1:30,1:40:1200);
% heatmap
imshow(data_temp, 'InitialMagnification', 'fit');
map = addcolorplus(292); colormap(map);
colorbar;
[num_rows, num_cols] = size(data_temp);
id_flag = 0;
% write text
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(data_temp(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xlabel('Time'); ylabel('Neuron Index 31 \sim 60');
title('Normalized firing rate of Mixed Nrueons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

figure(3)
% data:IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(1:30,1:40:1200);
% heatmap
imshow(data_temp, 'InitialMagnification', 'fit');
map = addcolorplus(292); colormap(map);
colorbar;
[num_rows, num_cols] = size(data_temp);
id_flag = 0;
% write text
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(data_temp(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xlabel('Time'); ylabel('Neuron Index 61 \sim 96');
title('Normalized firing rate of Mixed Nrueons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

