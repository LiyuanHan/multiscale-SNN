%% *****************************************************************************************************************************
% paper: Learning and Controlling Multi-scale Dynamics in Spiking Neural Networks using Recursive Least Square Modifications
% author: Liyuan Han et. al.
% Uploading Time: 2023.09.07
% code availability: https://github.com/LiyuanHan/multiscale-SNN
% file: IZ_Task_BCI.m, This code is for Izhikevich neurons in our SNN model.
% remark 1: This code is adapted from https://www.nature.com/articles/s41467-017-01827-3.
% remark 2: If you cite this paper, please also cite the paper https://www.nature.com/articles/s41467-017-01827-3.
% *******************************************************************************************************************************

%% Forward process with Izhikevich Network
clear all
close all
clc
data = load('data_for_BCI_RLS.mat') 
data_t = linspace(0,48,length(data.position));
color_set = {[0,148,255]/255, [169,169,169]/255,...
             [0,141,0]/255, [169,169,169]/255,...
             [255,146,0]/255, [169,169,169]/255,...
             [207,185,158]/255, [169,169,169]/255,...
             [229,8,106]/255, [169,169,169]/255,...
             [80,29,138]/255, [169,169,169]/255,...
             [170,52,116]/255, [169,169,169]/255,...
             [238,140,125]/255, [169,169,169]/255};
% color_set = {'#0094ff', '#A9A9A9',...
%              '#008d00', '#A9A9A9',...
%              '#ff9200', '#A9A9A9',...
%              '#cfb99e', '#A9A9A9',...
%              '#e5086a', '#A9A9A9',...
%              '#501d8a', '#A9A9A9',...
%              '#aa3474', '#A9A9A9',...
%              '#ee8c7d', '#A9A9A9'};
% **** BCI_data ****
for i = 1:length(data_t)/40
    if mod(i,16) == 0
        colour = '#A9A9A9';
    else
        colour = color_set{mod(i,16)};
    end
    figure(1)
%     hold on 
%     plot3(data_t((i-1)*40+1:i*40),data.position(1,(i-1)*40+1:i*40),data.position(2,(i-1)*40+1:i*40),'Color',[colour],'marker','.','linewidth',2)  
    plot(data.position(1,(i-1)*40+1:i*40),data.position(2,(i-1)*40+1:i*40),'Color',[colour],'marker','.','linewidth',2)  
    
    hold on 
end
xlim([-155, 150]);ylim([-150,150])
xlabel('Position-X');ylabel('Position-Y')
%% Normalized position
pos_mean = mean(data.position,2);       
pos_std = std(data.position,0,2);
data_postion = (data.position - pos_mean) ./ pos_std;
% % method -2
% pos_max = max([data.position]')';
% pos_min = min([data.position]')';
% data_postion = -1 + 2 * (data.position - pos_min) ./ (pos_max - pos_min);
%%

T = 480; %Total time in ms
dt = 0.025; %Integration time step in ms
nt = round(T/dt); %Time steps
N = 96; %Number of neurons
% 
n = N;
zx = zeros(n,length(data.position));
for i = 1:n
    if i<=n/2
%         zx(i,:) = data.position(1,:);
        zx(i,:) = data_postion(1,:);
    else
%         zx(i,:) = data.position(2,:);
        zx(i,:) = data_postion(2,:);
    end
end

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
td = 10; %decay time
pp = 0.3; %sparsity
if N > 96
    G = 3*10^4;
else
    G = 7.175*10^4;
end
% G = 8*10^4; %Gain on the static matrix with 1/sqrt(N) scaling weights.  Note that the units of this have to be in pA.
% #neuron = 96, G = 8*10^4;
% #neuron > 96, G = 3*10^4;
% 

epos = 3;
s_cur = zeros(nt,n);
s_spi = zeros(n,nt);
s_spi_each = zeros(N,nt);
s_fir = zeros(N,nt);
s_rec = zeros(nt,n);
for epo = 1:epos
%Storage variables for synapse integration
IPSC = zeros(N,1); %post synaptic current
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

%-----Initialization---------------------------------------------
rng(epo+3)
v = vr+(vpeak-vr)*rand(N,1); %initial distribution
v_ = v; %These are just used for Euler integration, previous time step storage
% 
%% parameters in control
% initial weight matrix A0
A0 = G*randn(N,N).* (rand(N,N)<pp)/(pp*sqrt(N));
for i = 1:1:N 
    QS = find(abs(A0(i,:))>0);
    A0(i,QS) = A0(i,QS) - sum(A0(i,QS))/length(QS);
end

% Gramian matrix W[0,T]
direc = randperm(N,n);
A0_temp = A0(:,direc);
% [A0_temp,order] = max_eig_n(A0,n);
% [V,D]=eig(A0);
% lam_max = abs(D(order(1),order(1)));
% A0 = 1/lam_max * A0;
% A0n = A0_temp'*A0*A0_temp;


%%
% A0 = OMEGA;
k = size(zx,1); %used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.
BPhi = zeros(N,k); %initial decoder--save most important k-dimentional vectors of weight A.  Best to keep it at 0.
% BPhi = A0_temp;
ns = 0; %count toal number of spikes
BIAS = 10; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons
% E = (2*rand(N,k)-1)*Q;  %Weight matrix is OMEGA0 + E*BPhi';
%%
% Pinv = eye(N)*2; %initial correlation matrix, coefficient is the regularization constant as well
step = 1; %optimize with RLS only every 1ms
imin = round(16/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round(4640/dt); %end simulation at this time step
% current = zeros(nt,k);  %store the approximant
%  RECB = zeros(nt,5); %store the decoders
%  REC = zeros(nt,10); %Store voltage and adaptation variables for plotting

current = zeros(nt,k);  %store the approximant
RECB = zeros(nt,50*n); %store the decoders
REC = zeros(nt,n); %Store voltage and adaptation variables for plotting
i=1;

%%
% save_P = repmat(zeros(N,N),1,T);


err = zeros(N,1);
P = eye(N)*2; %initial correlation matrix, coefficient is the regularization constant as well
%% Simulation

ilast = i;
%[BPhi, order] = max_eig_n(A0,k);
% BPhi = An;
xx = zeros(N,1);
x_appro = zeros(n,1);
flag = 1;
A = A0/G;
sA = [];
save_spike = zeros(N,1);
s_spike = zeros(n,nt);% calculate #spikes fired by which one neuron in each 1s

s_spi_each_temp = zeros(N,nt);
s_fir_temp = zeros(N,nt);
for j = ilast:1:nt  
%     if mod(j,10000) == 0
        j
%     end
%     xx(order(1:n)) = x_appro;
%     I = IPSC + Q*A0_temp*x_appro +BIAS;
    I = IPSC + A0_temp*x_appro +BIAS;
%     I = IPSC + A*r*x_appro +BIAS;
    v = v + dt*(( ff.*(v-vr).*(v-vt) - u_ada + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
    u_ada = u_ada + dt*(a*(b*(v_-vr)-u_ada)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    
    %%
    index = find(v>=vpeak);
    s_spike(index,j) = 1;
    if length(index)>0
        JD = sum(G*A(:,index),2); %compute the increase in current due to spiking
    end
    
    %synapse for double exponential
    IPSC = IPSC*exp(-dt/td) + h*dt;
    h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td);  %Integrate the current
    
    r = r*exp(-dt/td) + hr*dt;
    hr = hr*exp(-dt/tr) + (v>=vpeak)/(tr*td);
    
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
    u_ada = u_ada + d*(v>=vpeak);  %implements set u to u+d if v>vpeak, component by component.
    v = v+(vreset-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    v_ = v;  % sets v(t-1) = v for the next itteration of loop
    REC(j,:) = v';
    current(j,:) = x_appro'; % 存储的是0:T-1时刻的状态值

end
s_cur = (epo-1)/epo * s_cur + 1/epo * current;
s_spi = (epo-1)/epo * s_spi + 1/epo * s_spike;
s_rec = (epo-1)/epo * s_rec + 1/epo * REC;
s_spi_each = (epo-1)/epo * s_spi_each + 1/epo * s_spi_each_temp;
s_fir = (epo-1)/epo * s_fir + 1/epo * s_fir_temp;
end
%%
disp('Plot')

%% Plotting neurons before and after learning
% REC = s_rec;
time_win = 1;
s_spi_temp = zeros(N,nt/time_win);
for i = 1:nt/time_win
    s_spi_temp(:,i) = sum(s_spike(:,(i-1)*time_win+1:i*time_win),2);
end


figure(30)
for j = 35:5:75
    id_temp = find(s_spike(j,:)>0);
    id = zeros(1,3*length(id_temp));
    id(1:3:end) = id_temp; id(2:3:end) = id_temp; id(3:3:end) = NaN;
    y_val = zeros(1,3*length(id_temp));
    y_val(1:3:end) = j; y_val(2:3:end) = s_spike(j,id_temp)+j+1; y_val(3:3:end) = NaN;
    hold on 
    plot([id,1:nt]*dt,[y_val,ones(1,nt)*j])
end
ylim([35,80])
xlabel('Time (s)');  ylabel('Neuron Index')
title('Izhikevich Model')

%% Approximant 
fig_n = 2
% for i = 1:n
%     figure(10)
%     subplot(n,1,i)
%     plot((1:1:nt)*dt,zx(i,:),'k.',(1:1:nt)*dt,current(:,i),'b-.','LineWidth',2)
% end
current = s_cur;
c1 = mean(current(1:nt,1:n/2), 2);
c2 = mean(current(1:nt,n/2+1:n), 2);
zx1 = mean(zx(1:n/2, :));
zx2 = mean(zx(n/2+1:n, :));

% % % method -1
c1 = c1 * pos_std(1) + pos_mean(1);
c2 = c2 * pos_std(2) + pos_mean(2);
zx1 = zx1 * pos_std(1) + pos_mean(1);
zx2 = zx2 * pos_std(2) + pos_mean(2);
% % method -2
% c1 = (c1 + 1)/2 *(pos_max(1)-pos_min(1)) + pos_min(1);
% c2 = (c2 + 1)/2 *(pos_max(2)-pos_min(2)) + pos_min(2);
% zx1 = (zx1 + 1)/2 *(pos_max(1)-pos_min(1)) + pos_min(1);
% zx2 = (zx2 + 1)/2 *(pos_max(2)-pos_min(2)) + pos_min(2);

%% Target & Approximant
%[0.5020,0.5412, 0.5294]
% X - direction
figure(21)
subplot(2,1,1)
plot((1:1:nt)*dt,zx1,'-','Color',[0.6627 0.6627 0.6627],'LineWidth',1),hold on
plot((1:1:nt)*dt,c1,'b-','LineWidth',1),hold off
grid on
hold on
% % plot((nt:1:nt)*dt,current(nt:1:nt,1),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('Position-X')
ylim([-200,200])
legend('Target','Appro.','Orientation', 'horizontal')

% Y - direction
figure(21)
subplot(2,1,2)
plot((1:1:nt)*dt,zx2,'-','Color',[0.6627 0.6627 0.6627],'LineWidth',1),hold on
plot((1:1:nt)*dt,c2,'b-','LineWidth',1),hold off
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

%% Ground truth
figure(119)
plot(zx1,zx2,'k-','LineWidth',1);
xlabel('x'); ylabel('y');
grid on;
% Approximation
figure(119)
hold on
plot(c1,c2,'b-','LineWidth',1);
xlabel('x'); ylabel('y');
grid on;
legend('Target','Appro.')


% Prediction

%% [0 4, 2 4, 5 4, 7 4, 8 4, 6 4, 3 4, 1 4]
ref_spi_reange = zeros(96,length(data.spike));
IZ_fir_reange = zeros(N,length(data.spike));
IZ_spi_reange = zeros(N,length(data.spike));
 
% aaa = [];
id_flag = 0;
for k = 1:1:16
    for i = k:16:length(data.spike)/40
        id_flag = id_flag + 1;
        ref_spi_reange(:,(id_flag-1)*40+1:id_flag*40) = data.spike(:,(i-1)*40+1:i*40);
        IZ_fir_reange(:,(id_flag-1)*40+1:id_flag*40) = s_fir(:,(i-1)*40+1:i*40);
        IZ_spi_reange(:,(id_flag-1)*40+1:id_flag*40) = s_spi_each(:,(i-1)*40+1:i*40);
%         aaa = [aaa, (i-1)*40+1:i*40];
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
xlabel('Time (s)');  ylabel('Neuron Index')
title('Heatmap of normalized firing rate of Izhikevich Neurons from 1 to 96')
%% **** IZ_fir_normalized_line_N.eps ****
figure(16)
flag = 1;
for i=35:5:75
    hold on
    plot((1:nt+16)*dt,IZ_fir_reange_normalized2(i,:)+flag,'Linewidth',2)
    flag = flag + 1;
end
xlabel('Time (s)');  ylabel('Neuron Index')
% ylim([50,95]) % 后期调整纵轴刻度值

%% **** IZ_fir_normalized_point_N.eps ****
% max_abs_vals = max(abs(s_fir), [], 2);
max_abs_vals = max(max(abs(s_fir)));
s_fir_normalized = s_fir ./ max_abs_vals;
figure(17)
flag = 1;
for i=35:5:75
    hold on
    plot((1:nt)*dt,s_fir_normalized(i,:)+flag,'Linewidth',2)
    flag = flag + 1;
end
xlabel('Time (s)');  ylabel('Neuron Index')
% ylim([50,95])


%% **** fir_normalized_value_N_1-96.eps ****
figure(2)
% data: IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(1:30,1:40:1200);
% heatmap
imshow(data_temp, 'InitialMagnification', 'fit');
map = addcolorplus(292); colormap(map);
colorbar;
[num_rows, num_cols] = size(data_temp);
id_flag = 0;
% write value
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(data_temp(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xlabel('Time'); ylabel('Neuron Index 31 \sim 60');
title('Normalized firing rate of Izhikevich Neurons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

figure(3)
% data: IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(31:60,1:40:1200);
% heatmap
imshow(data_temp, 'InitialMagnification', 'fit');
map = addcolorplus(292); colormap(map);
colorbar;
[num_rows, num_cols] = size(data_temp);
id_flag = 0;
% write value
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(data_temp(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xlabel('Time'); ylabel('Neuron Index 31 \sim 60');
title('Normalized firing rate of Izhikevich Neurons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

figure(4)
% data: IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(61:96,1:40:1200);
% heatmap
imshow(data_temp, 'InitialMagnification', 'fit');
map = addcolorplus(292); colormap(map);
colorbar;
[num_rows, num_cols] = size(data_temp);
id_flag = 0;
% write value
for i = 1:num_rows
    for j = 1:num_cols
        text(j, i, num2str(data_temp(i, j), '%.2f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w');
    end
end
xlabel('Time'); ylabel('Neuron Index 61 \sim 96');
title('Normalized firing rate of Izhikevich Neurons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;


