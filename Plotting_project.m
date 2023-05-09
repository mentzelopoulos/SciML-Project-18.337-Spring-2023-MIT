close all;
clear;clc;

y_sol = textread('.\Results_Full_training\y_sol.txt');
ypp_sol = textread('.\Results_Full_training\ypp_sol.txt');
zplot = textread('.\Results_Full_training\z_plot.txt');
xplot = textread('.\Results_Full_training\x_plot.txt');
zeval = textread('.\Results_Full_training\zeval.txt');

y_sol80 = textread('.\Results_Partial_training\y_sol.txt');
ypp_sol80 = textread('.\Results_Partial_training\ypp_sol.txt');
zplot80 = textread('.\Results_Partial_training\z_plot.txt');
xplot80 = textread('.\Results_Partial_training\x_plot.txt');
zeval80 = textread('.\Results_Partial_training\zeval.txt');


D = 0.0363;     %Riser diameter in [m]
R = D/2;        %Riser RADIUS (not diam.) in [m] 
L = 152.524;    %Riser length [m]

given_strain_rms = rms(xplot);
predicted_strain_rms = rms(ypp_sol*R);

given_strain_rms80 = rms(xplot80);
predicted_strain_rms80 = rms(ypp_sol80*R);

figure(1);
subplot(2,1,2);hold on;
plot(zplot/L, given_strain_rms, 'ok','linewidth',1.3,'markersize',4);
plot(zeval/L, predicted_strain_rms, 'linewidth',1.4);
plot(zeval/L, predicted_strain_rms80,'-.', 'linewidth',2);
xlabel('L* = z/L','Interpreter','latex');
ylabel('\boldmath{ $\varepsilon$} (-)','Interpreter','latex');
legend('Measurements (RMS)', 'Predictions (RMS)', 'Predictions (RMS) - 80% Training');
set(gca, 'fontname','times new roman','fontsize',11)
box on; grid minor;legend boxoff;

subplot(2,1,1);hold on;
plot(zeval/L, rms(y_sol)/D, 'linewidth', 1.4);
plot(zeval/L, rms(y_sol80)/D,'-.', 'linewidth', 2);
xlabel('L* = z/L','Interpreter','latex');
ylabel('$\sqrt{\bar{y^2}}$','Interpreter','latex');
set(gca, 'fontname','times new roman','fontsize',11)
legend('Predictions (RMS)', 'Predictions (RMS) - 80% Training');legend boxoff;
box on; grid minor;

figure(2);hold on;
ts = sort(randsample([1:1050],4));
for i= 1:4
    subplot(4,1,i);hold on;
    t = ts(i);
    plot(zplot/L, xplot(t,:), 'ok','linewidth',1.3,'markersize',4);
    plot(zeval/L, ypp_sol(t,:)*R,'linewidth',1.4);
    title(['t = ',num2str(round(50+t/50.48,2)), ' s']);
    if i == 4
        xlabel('L* = z/L');
    end
    ylabel('\boldmath{ $\varepsilon$} (-)','Interpreter','latex');
    legend('Measurements', 'Prediction')
    set(gca, 'fontname','times new roman','fontsize',11)
    box on; grid minor;legend boxoff;
end

