%% Forward process with Izhikevich Network
clear all
close all
clc

data = load('task-lorenz_1e-4_T50.mat') % final saved result : task-lorenz_1e-4_T50.mat
zx = data.zx;

% data = load('lor_data_4e-5-v2.mat')
% zx = data.zx;

T = 50;
dt = 0.0001;
nt = round(T/dt); %Time steps
N =3000; %Number of neurons
% 
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
tr = 0.5;  %synaptic rise time
td = 5; %decay time
pp = 0.2; %sparsity
G = 10*10^3; 

epos = 5;
s_cur = zeros(nt,3);
s_spi = zeros(N,T*epos);
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
v = vr+(vpeak-vr)*rand(N,1); % initial distribution
v_ = v; %These are just used for Euler integration, previous time step storage
% 
%% parameters in control
n = 3;
m = n;


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

tspike = zeros(5*nt,2);  %If you want to store spike times,
ns = 0; %count toal number of spikes
BIAS = 10; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons

%%

step = 25; 
imin = round(1/dt); %time before starting RLS, gets the network to chaotic attractor
icrit = round(2000/dt); %end simulation at this time step

current = zeros(nt,k);  %store the approximant
RECB = zeros(nt,50*3); %store the decoders
REC = zeros(nt,50*2); %Store voltage and adaptation variables for plotting
i=1;

%%
% save_P = repmat(zeros(N,N),1,T);


err = zeros(N,1);
P = eye(N)*2; %initial correlation matrix, coefficient is the regularization constant as well
%% Simulation
ilast = i;

xx = zeros(N,1);
x_appro = zeros(n,1);
flag = 1;
A = A0/G;
sA = [];
save_spike = zeros(N,1);
s_spike = zeros(N,T);% calculate #spikes fired by which one neuron in each 1s
s_fir_temp = zeros(N,nt);
for j = ilast:1:nt  

        j

    I = IPSC + A0_temp*x_appro +BIAS;
    v = v + dt*(( ff.*(v-vr).*(v-vt) - u_ada + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
    u_ada = u_ada + dt*(a*(b*(v_-vr)-u_ada)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    
    %%
    index = find(v>=vpeak);
    if length(index)>0
        JD = sum(G*A(:,index),2); %compute the increase in current due to spiking
      
        tspike(ns+1:ns+length(index),:) = [index,0*index+dt*j];  %uncomment this
      
        ns = ns + length(index);
        save_spike(index) = save_spike(index)+ 1;
        if mod(j,1/dt)==0
            s_spike(:,j*dt) = save_spike;
            save_spike = zeros(N,1);
        end
    end
    
    %synapse for double exponential
    IPSC = IPSC*exp(-dt/td) + h*dt;
    h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td);  %Integrate the current
    
    r = r*exp(-dt/td) + hr*dt;
    hr = hr*exp(-dt/tr) + (v>=vpeak)/(tr*td);
    
    s_fir_temp(:,j) = r;
    % % Update process
    x_appro = BPhi'*r;%dimention:kX1
    Tem_err = x_appro - zx(:,j);
    % % RLS steps
    if mod(j,step) == 1
%         if j>imin
%             if j<icrit
                save_err(1:3,round(j*dt)+1) = BPhi' * r -zx(:,j);
                
                A(:,direc) = A(:,direc) - P*(r)*Tem_err';
                P = P - ( P*r*(P*r)' )/( 1+r'*P*r );
                BPhi = A(:,direc);
                 
                save_err(4:6,round(j*dt)+1) = BPhi' * r -zx(:,j);
%             end
%         end
    end
    
    % % Store, and plot.
    u_ada = u_ada + d*(v>=vpeak);  %implements set u to u+d if v>vpeak, component by component.
    v = v+(vreset-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    v_ = v;  % sets v(t-1) = v for the next itteration of loop

    REC(j,:) = [v(1:50)', u_ada(1:50)'];
    current(j,:) = x_appro'; % 存储的是0:T-1时刻的状态值

    RECB(j,:) = [BPhi(1:50,1)', BPhi(1:50,2)', BPhi(1:50,3)'];


end
% s_cur(:,(epo-1)*3+1:epo*3) = current;
s_spi(:,(epo-1)*T+1:epo*T) = s_spike;
s_cur = (epo-1)/epo * s_cur + 1/epo * current;
s_fir = (epo-1)/epo * s_fir + 1/epo * s_fir_temp;
end
%%
disp('Plot')

%% Plotting neurons before and after learning
figure(30)
sect_neu = [1,4,5,12,22,2,49];
id_flag = 0;
for j = 1:length(sect_neu)
    id_flag = id_flag + 1;
    plot((1:1:nt)*dt,REC(1:1:nt,sect_neu(j))/(vpeak-vreset)+id_flag), hold on
end
hold on 
figure(30)
% plot([1,1],[0,6],'Color',[0.5 0.5 0.5],'LineWidth',2)
% hold on
% plot([4,4],[0,6],'Color',[0.5 0.5 0.5],'LineWidth',2)
xlim([0,T]);ylim([0,length(sect_neu)+1])
xlabel('Time (s)');  ylabel('Neuron Index')
% title('Pre-and Post Learning')



figure(12)
subplot(3,1,1)
plot(save_err(1,:))
hold on
plot(save_err(4,:))
xlabel('Time (s)');ylabel('X');grid on;xlim([0,T]);
legend('$\tilde{e}_{-}$','$\tilde{e}_{+}$','Interpreter','LaTex','Orientation','horizontal');
subplot(3,1,2)
plot(save_err(2,:))
hold on
plot(save_err(5,:))
xlabel('Time (s)');ylabel('Y');grid on;xlim([0,T])
legend('$\tilde{e}_{-}$','$\tilde{e}_{+}$','Interpreter','LaTex','Orientation','horizontal');
subplot(3,1,3)
plot(save_err(3,:))
hold on
plot(save_err(6,:))
xlabel('Time (s)');ylabel('Z');grid on;xlim([0,T])
legend('$\tilde{e}_{-}$','$\tilde{e}_{+}$','Interpreter','LaTex','Orientation','horizontal');


s_fir_2 = [];
flag = 1;
for i = 1:nt
    if mod(i,25)==0
        s_fir_2(:,flag) = sum(s_fir(:,(flag-1)*25+1:flag*25),2);
        flag = flag + 1;
    end
end
figure(11)
x = (1:25:nt)*dt;
y = 1:40;
[X,Y]=meshgrid(x,y);
surf(X,Y,s_fir_2(y,:))
map = addcolorplus(309);
colormap(map)
colorbar
shading interp
view(0,90)
xlim([0,T]);ylim([1,40])
xlabel('Time (s)');ylabel('Neuron Index')
%save('Task1-0522.mat',''zx')

%% Approximant 
% c1=current(1:nt,1);
% c2=current(1:nt,2);
% c3=current(1:nt,3);

% c1 = sum(s_cur(:,1:5:end),2)/5;
% c2 = sum(s_cur(:,2:5:end),2)/5;
% c3 = sum(s_cur(:,3:5:end),2)/5;
c1 = s_cur(1:nt,1);
c2 = s_cur(1:nt,2);
c3 = s_cur(1:nt,3);
%作图



% % c1_2=current(nt/2:end,1);
% % c2_2=current(nt/2:end,2);
% % c3_2=current(nt/2:end,3);
% % %作图
% % hold on
% % figure(18)
% % subplot(1,2,2)
% % plot3(c1_2,c2_2,c3_2,'Color',[0.47 0.67 0.19]);
% % xlabel('x'); ylabel('y'); zlabel('z');
% % hold on
% % rho =28; rho2=60; beta = 8/3;
% % plot3(current(nt/2,1),current(nt/2,2),current(nt/2,3),'*m','LineWidth',2)
% % plot3(sqrt(beta*(rho2-1)),sqrt(beta*(rho2-1)),rho2-1,'gd','LineWidth',2)
% % plot3(-sqrt(beta*(rho2-1)),-sqrt(beta*(rho2-1)),rho2-1,'gd','LineWidth',2)
% % grid on;
% % hold on
% % legend('0~50 ms','Initial Point-1','Attractor-1','Attractor-2','50~100 ms','Initial Point-2','Attractor-3','Attractor-4')


%% figure(21) Target & Approximant
% X - direction
figure(21)
subplot(3,1,1)
plot((1:1:nt)*dt,zx(1,1:1:nt),'k.'),hold on
plot((1:1:nt)*dt,current(1:1:nt,1),'b-.','LineWidth',2),hold off
grid on
hold on
% % plot((nt:1:nt)*dt,current(nt:1:nt,1),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('$\hat{X}_t$','Interpreter','LaTex')
legend('Target-X','Appro.-X','Orientation','horizontal')

% Y - direction
figure(21)
subplot(3,1,2)
plot((1:1:nt)*dt,zx(2,1:1:nt),'k.'),hold on
plot((1:1:nt)*dt,current(1:1:nt,2),'b-.','LineWidth',2),hold off
grid on
hold on
% % plot((nt/2:1:nt)*dt,current(nt/2:1:nt,2),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('$\hat{Y}_t$','Interpreter','LaTex')
legend('Target-Y','Appro.-Y','Orientation','horizontal')

% Z - direction
figure(21)
subplot(3,1,3)
plot((1:1:nt)*dt,zx(3,1:1:nt),'k.'),hold on
plot((1:1:nt)*dt,current(1:1:nt,3),'b-.','LineWidth',2),hold off
grid on
hold on
% % plot((nt/2:1:nt)*dt,current(nt/2:1:nt,3),'-.','Color',[0.47 0.67 0.19],'LineWidth',2)
xlabel('Time (s)'); ylabel('$\hat{Z}_t$','Interpreter','LaTex')
legend('Target-Z','Appro.-Z','Orientation','horizontal')


%% figure(119) Ground truth
figure(119)
% subplot(1,2,1)
rho =28; beta = 8/3;
plot3(zx(1,1:nt),zx(2,1:nt),zx(3,1:nt));
xlabel('x'); ylabel('y'); zlabel('z');
hold on
plot3(zx(1,1),zx(2,1),zx(3,1),'*k','LineWidth',2)
plot3(sqrt(beta*(rho-1)),sqrt(beta*(rho-1)),rho-1,'rp')
plot3(-sqrt(beta*(rho-1)),-sqrt(beta*(rho-1)),rho-1,'rp')
grid on;
legend('0~50 s','Initial Point-1','Attractor-1','Attractor-2')

%% figure(120) Approximation
figure(120)
% subplot(1,2,1)
rho =28; beta = 8/3;
plot3(current(1:nt,1),current(1:nt,2),current(1:nt,3));
xlabel('x'); ylabel('y'); zlabel('z');
hold on
plot3(current(1,1),current(1,2),current(1,3),'*k','LineWidth',2)
plot3(sqrt(beta*(rho-1)),sqrt(beta*(rho-1)),rho-1,'rp')
plot3(-sqrt(beta*(rho-1)),-sqrt(beta*(rho-1)),rho-1,'rp')
grid on;
legend('0~50 s','Initial Point-1','Attractor-1','Attractor-2')

