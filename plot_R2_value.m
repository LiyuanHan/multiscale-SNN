%% *****************************************************************************************************************************
% paper: Learning and Controlling Multi-scale Dynamics in Spiking Neural Networks using Recursive Least Square Modifications
% author: Liyuan Han et. al.
% Uploading Time: 2023.09.07
% code availability: https://github.com/LiyuanHan/multiscale-SNN
% file: plot_R2_value.m, This code is plotting for R2 scores computed by three scenarios, i.e., Izhi. Bio. Mixded.
% remark 1: This code is adapted from https://www.nature.com/articles/s41467-017-01827-3.
% remark 2: If you cite this paper, please also cite the paper https://www.nature.com/articles/s41467-017-01827-3.
% *******************************************************************************************************************************
%%
close all
clear all
clc
%%
figure(1)
hold on
R2_x_y = [0.4228 0.6313 0.6506 NaN 0.4229 0.6867 0.7444]; % Izhi. Bio. Mixded; 'NaN' is a blank in figure.
% id = [1 2 3 4 5 6 7];
colors = {'#e55709','#1c8041','#501d8a','#501d8a','#e55709','#1c8041','#501d8a'};
for i = 1:length(R2_x_y)
    b = bar(i,R2_x_y(i));  %0.75是柱形图的宽，可以更改
    set(b(1),'facecolor',colors{i})
end
box on
Xlabel = {'Position-X','Position-Y'};
set(gca,'XTick',[2 6]);
set(gca,'XTickLabel',Xlabel);%设置柱状图每个柱子的横坐标
set(gca,'YLim',[0 1]);%设置纵坐标的数值间隔
ylabel('R^2  Value') 
grid on
legend('Izhi.','Bio.','Mixed')
% legend('Izhi.','Bio.','Mixed','Orientation', 'horizontal')


%%
figure(2)
hold on
% Izhi-R2_x,R2_y; Mixed-R2_x,R2_y
R2_x_y_neu = [0.4228 0.5771 0.5963 0.6274 0.6658 0.6870 0.7193;
              0.4229 0.5711 0.5900 0.6238 0.6483 0.6881 0.7117;
              0.6506 0.8153 0.8614 0.8855 0.9050 0.9290 0.9440;
              0.7444 0.8554 0.8929 0.9114 0.9231 0.9422 0.9551];
num_neu = [96 192 256 304 368 496 656];        
colors = {[229 87 9]/255, [229 87 9]/255, [80 29 138]/255, [80 29 138]/255};
names = {'Izhi.','Izhi.','Mixed','Mixed'};
subplot(2,1,1)
hold on
plot(num_neu,R2_x_y_neu(1,:),'Color',colors{1})
plot(num_neu,R2_x_y_neu(3,:),'Color',colors{3})
scatter(num_neu,R2_x_y_neu(1,:),'filled','MarkerFaceColor',colors{1})
scatter(num_neu,R2_x_y_neu(3,:),'filled','MarkerFaceColor',colors{3})
xlim([96 660]);ylim([0 1]);
xlabel('Number of Neurons')
ylabel('R^2  Value of Position-X')
grid on
legend('Izhi.','Mixed','Izhi.','Mixed','Location', 'best')
subplot(2,1,2)
hold on
plot(num_neu,R2_x_y_neu(2,:),'Color',colors{2})
plot(num_neu,R2_x_y_neu(4,:),'Color',colors{4})
scatter(num_neu,R2_x_y_neu(2,:),'filled','MarkerFaceColor',colors{2})
scatter(num_neu,R2_x_y_neu(4,:),'filled','MarkerFaceColor',colors{4})
xlim([96 660]);ylim([0 1]);
xlabel('Number of Neurons')
ylabel('R^2  Value of Position-Y')
grid on
legend('Izhi.','Mixed','Izhi.','Mixed','Location', 'best')
% for i = 1:size(R2_x_y_neu,1)
%     hold on
%     plot(num_neu,R2_x_y_neu(i,:),'Color',colors{i},'DisplayName',names{i})
%     hold on
%     scatter(num_neu,R2_x_y_neu(i,:),'filled','MarkerFaceColor',colors{i})
% end
% xlim([96 660]);ylim([0 1]);
% xlabel('Number of Neurons')
% ylabel('R^2  Value')
% grid on
% legend('Location', 'best')
