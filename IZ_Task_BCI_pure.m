%% *****************************************************************************************************************************
% paper: Learning and Controlling Multi-scale Dynamics in Spiking Neural Networks using Recursive Least Square Modifications
% author: Liyuan Han et. al.
% Uploading Time: 2023.09.07
% code availability: 
% file: IZ_Task_BCI_pure.m, This code is for biological neurons in our SNN model.
% remark 1: This code is adapted from https://www.nature.com/articles/s41467-017-01827-3.
% remark 2: If you cite this paper, please also cite the paper https://www.nature.com/articles/s41467-017-01827-3.
% *******************************************************************************************************************************

%% Forward process with Izhikevich Network
clear all
close all
clc
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
N = 96; %Number of neurons
% 
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
tr = 0.06;  %synaptic rise time
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
v = vr+(vreset-vr).*(data.spike(1:N,1));
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

%%
% A0 = OMEGA;
k = size(zx,1); %used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.
BPhi = zeros(N,k); %initial decoder--save most important k-dimentional vectors of weight A.  Best to keep it at 0.
% BPhi = A0_temp;
BIAS = 10; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons
%%
step = 1; %optimize with RLS only every 1ms
imin = round(16/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round(464/dt); %end simulation at this time step

current = zeros(nt,k);  %store the approximant
RECB = zeros(nt,50*n); %store the decoders
REC = zeros(nt,50*2); %Store voltage and adaptation variables for plotting
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
    index = find(data.spike(1:N,j)>0); % using BCI dataset
    if length(index)>0
        JD = sum(G*A(:,index).* data.spike(index,j)',2);  
        s_spi_each_temp(:,index) =  1;
    end
    
    %synapse for double exponential
    IPSC = IPSC*exp(-dt/td) + h*dt;
    h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td);  %Integrate the current
    
    r = r*exp(-dt/td) + hr*dt;
    hr = hr*exp(-dt/tr) + (data.spike(1:N,j))/(tr*td);
    
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
    u_ada = u_ada + d*(data.spike(1:N,j));  
    v = v+(vreset-v).*(data.spike(1:N,j)); 
    v_ = v;  
    REC(j,:) = [v(1:50)', u_ada(1:50)'];
    current(j,:) = x_appro'; 
end
s_cur = (epo-1)/epo * s_cur + 1/epo * current;
s_spi_each = (epo-1)/epo * s_spi_each + 1/epo * s_spi_each_temp;
s_fir = (epo-1)/epo * s_fir + 1/epo * s_fir_temp;
end
%%
disp('Plot')
%% BCI_spike.eps
figure(29)
for j = 35:5:75
    
    id_temp = find(data.spike(j,:)>0);
    id = zeros(1,3*length(id_temp));
    id(1:3:end) = id_temp; id(2:3:end) = id_temp; id(3:3:end) = NaN;
    y_val = zeros(1,3*length(id_temp));
    y_val(1:3:end) = j; y_val(2:3:end) = data.spike(j,id_temp)+j; y_val(3:3:end) = NaN;
    hold on 
    plot([id,1:nt]*dt,[y_val,ones(1,nt)*j])
end
xlabel('Time (s)');  ylabel('Neuron Index')
ylim([35,80])
title('Biological Model')

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
plot((1:1:nt)*dt,zx1(1:nt),'-','Color',[0.6627 0.6627 0.6627],'LineWidth',1),hold on
plot((1:1:nt)*dt,c1(1:nt),'b-','LineWidth',1),hold off
grid on
hold on
% % plot((nt:1:nt)*dt,current(nt:1:nt,1),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('Position-X')
ylim([-200,200])
legend('Target','Appro.','Orientation', 'horizontal')

% Y - direction
figure(21)
subplot(2,1,2)
plot((1:1:nt)*dt,zx2(1:nt),'-','Color',[0.6627 0.6627 0.6627],'LineWidth',1),hold on
plot((1:1:nt)*dt,c2(1:nt),'b-','LineWidth',1),hold off
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
IZ_spi_reange = zeros(N,nt);
 
% aaa = [];
id_flag = 0;
for k = 1:1:16
    for i = k:16:nt/40
        id_flag = id_flag + 1;
        ref_spi_reange(:,(id_flag-1)*40+1:id_flag*40) = data.spike(1:N,(i-1)*40+1:i*40);
        IZ_fir_reange(:,(id_flag-1)*40+1:id_flag*40) = s_fir(:,(i-1)*40+1:i*40);
        IZ_spi_reange(:,(id_flag-1)*40+1:id_flag*40) = s_spi_each(:,(i-1)*40+1:i*40);
%         aaa = [aaa, (i-1)*40+1:i*40];
    end
end


%% max_abs_vals = max(abs(IZ_fir_reange), [], 2); % 每一行的最大绝对值
max_abs_vals = max(max(abs(IZ_fir_reange)));
% IZ_fir_reange_normalized = IZ_fir_reange ;
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
title('Heatmap of normalized firing rate of Biological Neurons from 1 to 96')
%% **** IZ_fir_normalized_line_N.eps ****
figure(16)
flag = 1;
for i=35:5:75
    hold on
    plot((1:nt+16)*dt,IZ_fir_reange_normalized2(i,:)+flag,'Linewidth',2)
    flag = flag + 1;
end
xlabel('Time (s)'); ylabel('Neuron Index');
% ylim([10,90])

%% **** IZ_fir_normalized_point_N.eps ****
% max_abs_vals = max(abs(s_fir), [], 2);
max_abs_vals = max(max(abs(s_fir)));
s_fir_normalized = s_fir ./ max_abs_vals;
figure(17)
flag = 1;
for i=35:5:75%61:1:71
    hold on
    plot((1:nt)*dt,s_fir_normalized(i,:)+flag,'Linewidth',2)
    flag = flag + 1;
end
xlabel('Time (s)'); ylabel('Neuron Index');
% ylim([50,95])
% ylim([61,72])


%% **** fir_normalized_value_N_1-96.eps ****
figure(1)
% data: IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(1:30,30:40:1200);
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
xlabel('Time'); ylabel('Neuron Index 1 \sim 30');
title('Normalized firing rate of Biological Neurons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

figure(2)
% data: IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(31:60,30:40:1200);
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
title('Normalized firing rate of Biological Neurons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;

figure(3)
%data: IZ_fir_reange_normalized
data_temp = IZ_fir_reange_normalized(61:96,30:40:1200);
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
title('Normalized firing rate of Biological Neurons');
xticks(1:num_cols); yticks(1:num_rows);
xtickangle(45);
set(gcf, 'Position', [100, 100, 600, 600]);
grid on;







