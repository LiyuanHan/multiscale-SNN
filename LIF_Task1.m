%% *****************************************************************************************************************************
% paper: Learning and Controlling Multi-scale Dynamics in Spiking Neural Networks using Recursive Least Square Modifications
% author: Liyuan Han et. al.
% Uploading Time: 2023.09.07
% code availability: https://github.com/LiyuanHan/multiscale-SNN
% file: LIF_Task.m, This code is for point-to-point task using LIF neurons in our SNN model.
% remark 1: This code is adapted from https://www.nature.com/articles/s41467-017-01827-3.
% remark 2: If you cite this paper, please also cite the paper https://www.nature.com/articles/s41467-017-01827-3.
% *******************************************************************************************************************************


%% Forward process with LIF Network
clear all
close all
clc
%%
data = load('lor_data_4e-5-v2.mat');
zx = data.zx;

T = 5; %Total time in ms
dt = 0.00004; %Integration time step in ms
nt = round(T/dt); %Time steps
N = 1000; %Number of neurons
% 
%% LIF Parameters
tref = 0.002; %Refractory time constant in seconds 
tm = 0.01; %Membrane time constant 
vreset = -65; %Voltage reset 
vpeak = -40; %Voltage peak. 
tr = 0.002;%synaptic rise time
td = 0.02;%decay time

pp = 0.08; %sparsity
G = 0.025; %Gain on the static matrix with 1/sqrt(N) scaling weights.  Note that the units of this have to be in pA.

%% parameters in control
% rng(2)
direc2 = [26   419   435   436   549];
direc = randperm(N,100);
A0 = G*(randn(N,N)).*(rand(N,N)<pp)/(sqrt(N)*pp);

for i = 1:1:N 
    QS = find(abs(A0(i,:))>0);
    A0(i,QS) = A0(i,QS) - sum(A0(i,QS))/length(QS);
end

T_temp = T;

%% Storage variables for synapse integration
IPSC = zeros(N,1); %post synaptic current
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

%-----Initialization---------------------------------------------
% rng(2)
v = vreset + rand(N,1)*(30-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v; 
k = size(zx,1); 
BPhi = zeros(N,1); 


BIAS = vpeak; 
%%
step = 50; 
imin = round(1/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round((T-0.5)/dt); %end simulation at this time step

current = zeros(nt,1);  %store the approximant
RECB = zeros(nt,5); %store the decoders
REC = zeros(nt,5); %Store voltage and adaptation variables for plotting


%%
err = zeros(N,1);
alpha = dt * 0.1;
P = eye(N)*alpha; %initial correlation matrix, coefficient is the regularization constant as well
%% Simulation
x_appro = zeros(1,1);
A0_temp = sum(A0(:,direc),2)/length(direc); %A0(:,direc)/G;
A = A0/G;
A_direc = sum(A0(:,direc),2)/length(direc);

save_spike(:,1) = zeros(N,1);
s_spike = zeros(N,T);
s_fir_rate = zeros(N,nt);
save_err = [];

tlast = zeros(N,1); %This vector is used to set  the refractory times 
for j = 1:1:nt  
    j
    I = IPSC + A0_temp*x_appro +BIAS;
    dv = (dt*j>tlast + tref).*(-v+I)/tm; %Voltage equation with refractory period 
    v = v + dt*(dv);    
    %%
    index = find(v>=vpeak);
    if length(index)>0
        JD = sum(A0(:,index),2); %compute the increase in current due to spiking
    end
    tlast = tlast + (dt*j -tlast).*(v>=vpeak);  %Used to set the refractory period of LIF neurons 

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
            if j<icrit          
                A_direc = A_direc - P*(r)*Tem_err';
                P = P - ( P*r*(P*r)' )/( 1+r'*P*r );
                BPhi = A_direc;
                
            end
        end
    end
    
    
    
    
    %% Store, and plot.
    v = v+(30-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    REC(j,:) = v(459:463)';
    v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
    current(j,:) = mean(x_appro'); % save state value from 0:T-1
    RECB(j,:)= BPhi(1:5);
   
end

%%
disp('Plot')
%% Plotting neurons before and after learning
figure(30)
for j = 459:1:463
    plot((1:1:nt)*dt,REC(1:1:nt,j-458)/(30-vreset)+j), hold on
end
hold on 
figure(30)
plot([1,1],[458,464],'Color',[0.5 0.5 0.5],'LineWidth',2)
hold on
plot([4.5,4.5],[458,464],'Color',[0.5 0.5 0.5],'LineWidth',2)
xlim([0,T])
xlabel('Time (s)');  ylabel('Index')


figure(12)
hold on
for i = 1:T_temp
    hold on
    plot(dt*((i-1)*round(1/dt)+1:1:i*round(1/dt)),zx(:,(i-1)*round(1/dt)+1:1:i*round(1/dt)),'LineWidth',2)
    hold on
end
hold on
plot(dt*(1:1:nt),current(1:1:nt,:),'b-.','LineWidth',2)
hold on 
plot([1,1],[-1.5,1.5],'Color',[0.5 0.5 0.5],'LineWidth',2)
hold on
plot([4.5,4.5],[-1.5,1.5],'Color',[0.5 0.5 0.5],'LineWidth',2)
ylim([-1.5,1.5])
xlabel('Time (s)');ylabel('$\hat{X}_t$','Interpreter','LaTex');
legend('0-1 s','1-2 s','2-3 s','3-4 s','4-5 s','Appro.')



