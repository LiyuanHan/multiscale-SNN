% clear all
% close all
% clc 
% function [Tr_save_x,Tr_save_u]=x0_xT_DP_True(A,B,T,x0,x_target)%(A,B,T,x0,x_target,direc)
function [Tr_save_x,Tr_save_u]=x0_xT_DP_True(A,B,T,x0,x_target,direc)
n = length(x0);
m = size(B,2);

% Q = eye(n);
Q = zeros(n);
R = eye(m);

Tr_save_x = zeros(n,T+1);
Tr_save_x(:,1) = x0;
Tr_save_u = zeros(m,T);

Ku = zeros(m,m*T);
Kx = zeros(m,n*T);
KT = zeros(m,n*T);
K_uj_xj = zeros(n,m*T);
Vxj_j = zeros(n,n*T);
Vxj_T = zeros(n,n*T);

%%
 for j = T-1
     Ku(:,m*j+1:m*(j+1)) = B'*B;
     Kx(:,n*j+1:n*(j+1)) = B'*A;
     KT(:,n*j+1:n*(j+1)) = B'*eye(n);
     K_uj_xj(:,m*j+1:m*(j+1)) = -Kx(:,n*j+1:n*(j+1))'*( Ku(:,m*j+1:m*(j+1))^(-1) )';
     Vxj_j(:,n*j+1:n*(j+1)) = Q + K_uj_xj(:,m*j+1:m*(j+1)) * R * K_uj_xj(:,m*j+1:m*(j+1))';
     Vxj_T(:,n*j+1:n*(j+1)) = K_uj_xj(:,m*j+1:m*(j+1)) * R * Ku(:,m*j+1:m*(j+1))^(-1) * KT(:,n*j+1:n*(j+1));
 end
 
 for j = T-2:-1:0
     Ku(:,m*j+1:m*(j+1)) = R + B' * Vxj_j(:,n*(j+1)+1:n*(j+2)) * B;
     Kx(:,n*j+1:n*(j+1)) = B' * Vxj_j(:,n*(j+1)+1:n*(j+2)) * A;
     KT(:,n*j+1:n*(j+1)) = -B' * Vxj_T(:,n*(j+1)+1:n*(j+2));
     K_uj_xj(:,m*j+1:m*(j+1)) = -Kx(:,n*j+1:n*(j+1))'*( Ku(:,m*j+1:m*(j+1))^(-1) )';
     Vxj_j(:,n*j+1:n*(j+1)) = Q + K_uj_xj(:,m*j+1:m*(j+1)) * R * K_uj_xj(:,m*j+1:m*(j+1))' + (A'+K_uj_xj(:,m*j+1:m*(j+1))*B') * Vxj_j(:,n*(j+1)+1:n*(j+2)) * A+ (A'+K_uj_xj(:,m*j+1:m*(j+1))*B') * Vxj_j(:,n*(j+1)+1:n*(j+2)) * B * K_uj_xj(:,m*j+1:m*(j+1))';
     Vxj_T(:,n*j+1:n*(j+1)) = K_uj_xj(:,m*j+1:m*(j+1)) * R * Ku(:,m*j+1:m*(j+1))^(-1) * KT(:,n*j+1:n*(j+1)) + (A'+K_uj_xj(:,m*j+1:m*(j+1))*B')*Vxj_T(:,n*(j+1)+1:n*(j+2))+ (A'+K_uj_xj(:,m*j+1:m*(j+1))*B') * Vxj_j(:,n*(j+1)+1:n*(j+2)) * B * Ku(:,m*j+1:m*(j+1))^(-1) * KT(:,n*j+1:n*(j+1));
 end
 

 set_N = 1:n;
 for j = 0:1:T-1
     Tr_save_u(:,j+1) = K_uj_xj(:,m*j+1:m*(j+1))'* Tr_save_x(:,j+1) + Ku(:,m*j+1:m*(j+1))^(-1)* KT(:,n*j+1:n*(j+1)) * x_target;
     Tr_save_x(:,j+2) = A * Tr_save_x(:,j+1) + B * Tr_save_u(:,j+1);
      Tr_save_x(setdiff(set_N,direc),j+2) = 0;
 end

% W_k = [];
% for k = 0:T-1
%     W_k = W_k + A^(T-k-1)*B*B'*(A^(T-k-1))';
% end
% rank(W_k)
% for j = 0:T-1
%     uk(:,j+1) = B'*(A^(T-j-1)')*W_k^(-1)*x_target;
% end
% W_k
% uk

end
%%
