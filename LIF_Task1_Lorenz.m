%% Forward process with Izhikevich Network
clear all
close all
clc

% data = load('task-lorenz_1e-4_T50.mat') % final saved result : task-lorenz_1e-4_T50.mat
% zx = data.zx;

data = load('lor_data_4e-5-v2.mat')
zx = data.zx;

Factor = 1;
T = 5/Factor; %Total time in ms
dt = 0.00001; %Integration time step in ms
nt = round(T/dt); %Time steps
N =3000; %Number of neurons
% 
%% LIF Parameters
tref = 0.002; %Refractory time constant in seconds 
tm = 0.01; %Membrane time constant 
vreset = -65; %Voltage reset 
vpeak = -30; %Voltage peak. % yuan lai shi -40
tr = 0.002;%synaptic rise time
td = 0.02;%decay time
pp = 0.2;
G = 0.025;


epos = 5;
s_cur = zeros(nt,3*epos);
s_spi = zeros(N,T*epos);
for epo = 1:epos
%Storage variables for synapse integration
IPSC = zeros(N,1); %post synaptic current
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

%-----Initialization---------------------------------------------
% rng(2)
v = vreset + rand(N,1)*(30-vreset); %initial distribution
v_ = v; %These are just used for Euler integration, previous time step storage
% 
%% parameters in control
n = 3;

% initial weight matrix A0
A0 = G*randn(N,N).* (rand(N,N)<pp)/(pp*sqrt(N));
for i = 1:1:N 
    QS = find(abs(A0(i,:))>0);
    A0(i,QS) = A0(i,QS) - sum(A0(i,QS))/length(QS);
end

% Gramian matrix W[0,T]
popular = 1000;
direc = randperm(N,popular*3);
A0_temp(:,1) = sum(A0(:,direc(1:popular)),2)/popular;
A0_temp(:,2) = sum(A0(:,direc(popular*1+1:2*popular)),2)/popular;
A0_temp(:,3) = sum(A0(:,direc(popular*2+1:3*popular)),2)/popular;

A_direc = A0_temp;



%%
% A0 = OMEGA;
k = size(zx,1); %used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.
BPhi = zeros(N,k); %initial decoder--save most important k-dimentional vectors of weight A.  Best to keep it at 0.
% BPhi = A0_temp;

tspike = zeros(5*nt,2);  %If you want to store spike times,
ns = 0; %count toal number of spikes
BIAS = vpeak; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons


step = 25; %optimize with RLS only every 1ms
imin = round(1/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round(2000/dt); %end simulation at this time step


current = zeros(nt,k);  %store the approximant
RECB = zeros(nt,50*3); %store the decoders
REC = zeros(nt,50); %Store voltage and adaptation variables for plotting


err = zeros(N,1);
alpha = 0.1*dt;
P = eye(N)*alpha; %initial correlation matrix, coefficient is the regularization constant as well
%% Simulation


xx = zeros(N,1);
x_appro = zeros(n,1);
flag = 1;
A = A0/G;
sA = [];
save_spike = zeros(N,1);
s_spike = zeros(N,T);% calculate #spikes fired by which one neuron in each 1s
tlast = zeros(N,1);
for j = 1:1:nt  
    j

    I = IPSC + A0_temp*x_appro +BIAS;
    dv = (dt*j>tlast + tref).*(-v+I)/tm; %Voltage equation with refractory period 
    v = v + dt*(dv);
    %%
    index = find(v>=vpeak);

    if length(index)>0
        JD = sum(G*A(:,index),2); %compute the increase in current due to spiking
    end
    
    tlast = tlast + (dt*j -tlast).*(v>=vpeak);  %Used to set the refractory period of LIF neurons 

    
    %synapse for double exponential
    IPSC = IPSC*exp(-dt/td) + h*dt;
    h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td);  %Integrate the current
    
    r = r*exp(-dt/td) + hr*dt;
    hr = hr*exp(-dt/tr) + (v>=vpeak)/(tr*td);

    
    % % Update process
    x_appro = BPhi'*r;%dimention:kX1
    Tem_err = x_appro - zx(:,j);
    
    % % RLS steps
    if mod(j,step) == 1
        if j>imin
%             if j<icrit    
                A_direc = A_direc - P*(r)*Tem_err';
                P = P - ( P*r*(P*r)' )/( 1+r'*P*r );
                BPhi = A_direc;
%             end
         end
    end
    
    % % Store, and plot.
     v = v+(30-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    REC(j,:) = v(1:50)';
    v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
    current(j,:) = x_appro'; 
    RECB(j,:) = [BPhi(1:50,1)', BPhi(1:50,2)', BPhi(1:50,3)'];

end
s_cur(:,(epo-1)*3+1:epo*3) = current;
s_spi(:,(epo-1)*T+1:epo*T) = s_spike;
end
%%
disp('Plot')

%% Plotting neurons before and after learning
% figure(300)
% for j = 1:1:5
%     plot((1:1:nt)*dt,REC(1:1:nt,j)/(30-vreset)+j), hold on
% end
% hold on 
% figure(300)
% % plot([1,1],[0,6],'Color',[0.5 0.5 0.5],'LineWidth',2)
% % hold on
% % plot([4,4],[0,6],'Color',[0.5 0.5 0.5],'LineWidth',2)
% xlim([0,T])
% xlabel('Time (ms)');  ylabel('Neuron Index')
% title('Pre-and Post Learning')


%% Approximant 
figure(118)
c1=current(1:nt,1);
c2=current(1:nt,2);
c3=current(1:nt,3);
plot3(c1,c2,c3,'Color',[0.00 0.45 0.74]);
xlabel('x'); ylabel('y'); zlabel('z');
hold on
rho =28; rho2=60; beta = 8/3;
plot3(current(1,1),current(1,2),current(1,3),'*k','LineWidth',2)
hold on
plot3(sqrt(beta*(rho-1)),sqrt(beta*(rho-1)),rho-1,'rp')
hold on
plot3(-sqrt(beta*(rho-1)),-sqrt(beta*(rho-1)),rho-1,'rp')
grid on
legend('0~50 s','Initial Point-1','Attractor-1','Attractor-2')
xlim([-20,21])
ylim([-23,40])
zlim([0,60])

% figure(119)
% c4=data.current(1:nt,1);
% c5=data.current(1:nt,2);
% c6=data.current(1:nt,3);
% plot3(c4,c5,c6,'Color',[0.00 0.45 0.74]);
% xlabel('x'); ylabel('y'); zlabel('z');
% hold on
% rho =28; rho2=60; beta = 8/3;
% plot3(current(1,1),current(1,2),current(1,3),'*k','LineWidth',2)
% plot3(sqrt(beta*(rho-1)),sqrt(beta*(rho-1)),rho-1,'rp')
% plot3(-sqrt(beta*(rho-1)),-sqrt(beta*(rho-1)),rho-1,'rp')
% grid on
% legend('0~50 s','Initial Point-1','Attractor-1','Attractor-2')
% xlim([-20,21])
% ylim([-23,40])
% zlim([0,60])
