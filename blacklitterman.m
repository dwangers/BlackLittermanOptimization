%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% David Wang, Engineering Science 1T8+PEY
% MIE377 - Financial Optimization Models
% blacklitterman.m 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%0. PORTFOLIO BUILD, PHASE 0 - INPUT ASSETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load('new.mat');
capweights=zeros(1,8);
for i=1:8
    capweights(1,i)=marketcap(1,i)/sum(marketcap);
end;

%find returns for 8 different asset class
rets=zeros(59,8);
for i=1:8
    rets(:,i)=prices(2:end,i)./prices(1:end-1,i)-1;
end

rf=0.0045; %risk free rate 
mu=mean(rets);

%compute covariance matrix sigma
sigma=cov(rets-rf);

%find risk aversion lambda 
expected_ret=mu*capweights';
lambda=(expected_ret-rf)/(capweights*sigma*capweights');

%find CAPM  portofolio return (return should be same as pi)
capm_ret=(expected_ret-rf)*(sigma*capweights')/(capweights*sigma*capweights')
%calculate implied equilibrium vector

pi=lambda*sigma*capweights';

%recommended portofolio weight based on pi should be same as market capweights
w_pi=inv(lambda*sigma)*pi;


%% part2 incorporateing views
%1.US Small Value will have an absolute return of 0.3% (25% of confidence);
%2.international bond will outperform us bound by 0.1%; (tilt away from
%international) (50% of confidence)
%3.international Dev Equity and int'l Emerg Equity will outperform  US Large Growth and US small growth by 0.15%.(65% of confidence) 

%view 3, find weighted averge implied return for two set.
%for us large growth and samll growth (nominally 'underperforming' asset)
totalcap1=marketcap(1,3)+marketcap(1,5);
weightedpi_1=(marketcap(1,3)/totalcap1)*pi(3,1)+(marketcap(1,5)/totalcap1)*pi(5,1);

%for int'l dev equity and int'l emerging equity (nominally 'outperforming' asset)
totalcap2=marketcap(1,7)+marketcap(1,8);
weightedpi_2=(marketcap(1,7)/totalcap2)*pi(7,1)+(marketcap(1,8)/totalcap2)*pi(8,1);

weighted_difference=weightedpi_2-weightedpi_1;
%build view vector Q and vector P that matches view to assests

Q=[0.003;0.001;0.0015];
p=[0,0,0,0,0,1,0,0;-1,1,0,0,0,0,0,0;0,0,-0.9961,0,-0.0039,0,0.2872,0.7128];

%pick scaling constant tau to be 0.025;

tau=0.025;

%build variance of view - omeaga matrix
omega=zeros(3,3);
w1=p(1,:)*sigma*p(1,:)'*tau;
w2=p(2,:)*sigma*p(2,:)'*tau;
w3=p(3,:)*sigma*p(3,:)'*tau;
omega(1,1)=w1;
omega(2,2)=w2;
omega(3,3)=w3;

%calculate new(posterior) combined return vector from black-litter model
first=inv(inv(tau*sigma)+p'*inv(omega)*p);
second=[inv(tau*sigma)*pi+p'*inv(omega)*Q];
newreturn=first*second;

newweight=inv(lambda*sigma)*newreturn;%new recommend weight of portofolio

%% Part3 The New Method- An intuitive Approach 
%step 1
ret_100=pi+tau*sigma*p'*inv(p*tau*sigma*p')*(Q-p*pi);

%step 2
w100=inv(lambda*sigma)*ret_100;

%step 3
D_100=w100-capweights';

%step 4
c_k=[0.5;0.5;0.65;0;0.65;0.25;0.65;0.65];

tilt=zeros(8,1);
for i=1:8
    tilt(i,1)=D_100(i,1)*c_k(i,1);
end
%step5
target_weight=capweights'+tilt;

%step6
omega_k=zeros(3,3);
for i=1:3
    wk = @(x)inv(lambda*sigma)*inv(inv(tau*sigma)+p(i,:)'*inv(x)*p(1,:))*(inv(tau*sigma)*pi+p(i,:)'*inv(x)*Q(i,1));
    func = @(x)(target_weight-wk(x))'*(target_weight-wk(x));
    omega_k(i,i) = fminsearch(func,0.01);

end

final_ret = inv(inv(tau*sigma)+p'*inv(omega_k)*p)*(inv(tau*sigma)*pi+p'*inv(omega_k)*Q);
final_weight = inv(lambda*sigma)*final_ret;








