clear;
%testdataset: MNIST
%%%optimization problem
%we take two layer network as an example, this program can also be used to solve multilayers neurons network
%  min   loss1 = 1/2/n*dist2(V2,y_train)+beta/n*sum(sum(V1-max(U1,0)))+beta/n*sum(sum(V2-max(U2,0)))...
%W,V,U,b        +1/2/rho/n*(sum(sum(xi1+rho*U1-rho*(repmat(b1,1,N)+ W1* x_train))).^2)-sum(sum(xi1.^2)))...
%               +1/2/rho/n*(sum(sum(xi2+rho*U2-rho*(repmat(b2,1,N)+ W2* V1)).^2)-sum(sum(xi2.^2)))...
%               +rv1/n/2*sum(sum(V1.^2))+rv1/n/2*sum(sum(V2.^2))+rv/n*sum(sum(V1))+rv/n*sum(sum(V2))...
%               +rw/2/n*sum(sum(W1.^2))+rw/2/n*sum(sum(W2.^2))+......; 
%   s.t.  V >= sigma(U)
%sigma: ReLU, or leReLU or identity function

%--------------------------------------------
%parameters to be modified
%dimension of the hidden layer, such as [400;200;100], can be empty.
%if we set d_hid=[400;50], there are three layers.
N=1000;
d_hid=[200];
layer=size(d_hid,1)+1;
arelu = 0.1; %ReLU
%model parameters. we suggest setting rv1=rv=rw=0 when there are two
%layers. we suggest setting rv1=rv=0, rw=10 when there are at least three layers.
beta=1; rv1=0;rv=0;
%ini parameters
gamma=1;  %Wb subproblem, proximal parameter
tau=1;          %lagragian parameter
%ini norm parameter and norm
% 1:Fro+Fro, 2: Fro+no, 3:l21+no,
% correspond to different `update_Wb'
normcase=4; 
switch normcase
    case 1 %updateWb_total
        rw1=10; rw2=10; rw3=10; rw=rw1; alpha=0.1; 
    case 2 %updateWb_total2
        rw1=10; rw2=10; rw3=10; alpha=0.1;
    case 3 %bblinesearch
        rw1=10; rw2=2; rw3=1; alpha=0.1;gamma=0;
    case 4 %traditional bb step size
        rw1=1; rw2=1; rw3=1; alpha=0;gamma=0;
end
%initial value of rho
rho=1;
%max iteration of the bcd subproblem
bcdtotal=10;
%max iteration 
niter=500;
%initial value of \epsilon_k
epk = 1e-1;  
%activation functions (1-L-1 layers and L layer)
%1:ReLU(max(z,0))+ReLU, 2:ReLU+iden, 3:leReLU(max(z,0.1z))+leReLU, 4:leReLU+iden
acticationcase=1;
%SGS step or bcd step or parallel step, 0(parallel) or 1(bcd) or >1(sGS)
sgscase=0;proWcase=0;
%guess the value of gamma, 0 or >0
gammaguess=1;
%when UWerror(k)>UWerror(k-feasiindex:k-1) (loss2), we increase the value of rho
%we suggest setting feasiindex=2*layer
feasiindex=2*layer;
tortrainerr=1e-8;
%whether we compute the test error
testdatashow=1;

%-------------------------------------------------

%-------------------------------------------------
%dimension of the hidden layer, such as [100;200;300], cannot be empty.
if layer==2
    epk = 1e-1; 
    if normcase==1 || normcase==2
        rw1=0; rw2=0;
    end
end
bcdinicase=2;bcdmintotal=3;
%-------------------------------------------------

% MNIST dataset
%-------------------------------------------------
 load('/Users/test/Desktop/report_DL/code/BCD/mnist/train_x.mat');
 load('/Users/test/Desktop/report_DL/code/BCD/mnist/train_y.mat');
 load('/Users/test/Desktop/report_DL/code/BCD/mnist/test_x.mat');
 load('/Users/test/Desktop/report_DL/code/BCD/mnist/test_y.mat');
 [x_train,y_train,x_test,y_test,train_y,test_y] = mnist_datanum(N/10, train_x,test_x,train_y,test_y);

%dimension
[d0,n]=size(x_train); N=n;
[~,n2]=size(x_test); N2=n2; N_test = N2;
%true output, test set

%-------------------------------------------------------------------
%initialization
d_net=[d0;d_hid;10];


%loss1: AL function value; loss2: UW FeasVI; loss4: UV FeasVI; 
%loss3: trainerr; acc_train: accuracy for the training set; acc_test: accuracy for the test set; 
%time: CPU time
%losskj: AL function value in solving the subproblem
loss1=zeros(niter,1);loss2=zeros(niter,1);loss3=zeros(niter,1);loss4=zeros(niter,1);
acc_train=zeros(niter,1);acc_test=zeros(niter,1);
time=zeros(niter,1);
losskj=zeros(bcdtotal+1,1);
%bound for the subproblem

%--------------------------------------------
%initialization
[W,b,U,V] = iniDNN(d_net,x_train);W111=W;b111=b;U111=U;V111=V;% when we use a random initial point
% index=0;
% W=W111;b=b111;U=U111;V=V111;% when we use a given initial point

%initialize the feasibility error and U, W, V, b, lagrangian multiplier.
for i=1:layer
    name=['W', num2str(i), '=reshape(W(1:d_net(i)*d_net(i+1)),d_net(i+1),d_net(i));'];
    eval(name);
    W(1:d_net(i)*d_net(i+1))=[];
    name=['b', num2str(i), '=reshape(b(1:d_net(i+1)),d_net(i+1),1);'];
    eval(name);
    b(1:d_net(i+1))=[];
    name=['V', num2str(i), '=reshape(V(1:d_net(i+1)*n),d_net(i+1),n);'];
    eval(name);
    V(1:d_net(i+1)*n)=[];
    name=['U', num2str(i), '=reshape(U(1:d_net(i+1)*n),d_net(i+1),n);'];
    eval(name);
    U(1:d_net(i+1)*n)=[];
    name=['xi', num2str(i), '=zeros(size(U',num2str(i),'));'];
    eval(name);
    name=['U', num2str(i), 'ini=U',num2str(i),';'];
    eval(name);
    name=['a', num2str(i), '_train=V',num2str(i),';'];
    eval(name);
    name=['error', num2str(i), '=U', num2str(i),'-U', num2str(i),';'];
    eval(name);
end
V0=x_train;
%--------------------------------------------

%--------------------------------------------
%main iteration
fprintf('Train on %d samples, validate on %d samples\n', N, N_test);
for k = 1:niter
    
    bcdmaxiter=max(bcdmintotal,min(bcdtotal,ceil(k/20)));
    index=0;
    
    tic;
    %--------------------------------------------
    %no guess for \gamma
    if gammaguess==0
         for i=2:layer
            if bcdinicase<3 
                name=['eigenW', num2str(i), '=svds(W',num2str(i),',1);'];
                eval(name);
                name=['gamma', num2str(i), '=(max(1,eigenW',num2str(i),')).^2*rho;'];
                eval(name);
            end
         end
    else
    %guess for \gamma
        for i=2:layer
            if bcdinicase==3
                name=['gamma', num2str(i), '=(max(1,eigenW',num2str(i),')).^2*rho+1;'];
                eval(name);
            else
                name=['eigenW', num2str(i), '=svds(W',num2str(i),',1);'];
                eval(name);
                name=['gamma', num2str(i), '=max(1,eigenW',num2str(i),')*max(min(sqrt(rho),rho/10),1);'];
                eval(name);
            end
        end
    end
    %--------------------------------------------
    
    %--------------------------------------------
    %determine the ini for subproblem 
    %solve the subproblem, we omit the case 1
    switch bcdinicase
        case 1 % we use (W^{k-1},...) as the initial point of the bcd subproblem
            for i=1:layer
                name=['V', num2str(i), 'k=V',num2str(i),';'];
                eval(name);
                name=['W', num2str(i), 'k=W',num2str(i),';'];
                eval(name);
                name=['U', num2str(i), 'k=U',num2str(i),';'];
                eval(name);
                name=['b', num2str(i), 'k=b',num2str(i),';'];
                eval(name);
            end
            losskj(1)=loss1(k-1);
        case 2 % we use ({\bar W^{k}},...) as the initial point of the bcd subproblem
            for i=1:layer
                name=['U', num2str(i), '=U',num2str(i),'ini;'];
                eval(name);
                name=['V',num2str(i),'=a', num2str(i), '_train;'];
                eval(name);
                name=['V', num2str(i), 'k=V',num2str(i),';'];
                eval(name);
                name=['W', num2str(i), 'k=W',num2str(i),';'];
                eval(name);
                name=['U', num2str(i), 'k=U',num2str(i),';'];
                eval(name);
                name=['b', num2str(i), 'k=b',num2str(i),';'];
                eval(name);
            end
            losskj=0;
            if k>1
                losskj(1)=loss3(k-1);
                for i=1:layer
                    name=['losskj(1)=losskj(1)','+rv1/n/2*sum(sum(V',num2str(i),'.^2))',...
                        '+rw',num2str(i),'/n/2*sum(sum(W',num2str(i),'.^2))','+rv/n*sum(sum(V',num2str(i),'));'];
                    eval(name);
                end
            end
        case 3  %we do not change the value of V1k, U1k
            for i=1:layer
                name=['V', num2str(i), '=V',num2str(i),'k;'];
                eval(name);
                name=['W', num2str(i), '=W',num2str(i),'k;'];
                eval(name);
                name=['U', num2str(i), '=U',num2str(i),'k;'];
                eval(name);
                name=['b', num2str(i), '=b',num2str(i),'k;'];
                eval(name);
            end
            losskj(2:end)=0;
    end
    %--------------------------------------------
    
    %--------------------------------------------
    %solving the subproblem
    for kj=1:bcdmaxiter
        
        %--------------------------------------------
        % UV update    
        if  index==0
            for i=1:layer
                name=['V', num2str(i), 'kj=V',num2str(i),';'];
                eval(name);
                name=['U', num2str(i), 'kj=U',num2str(i),';'];
                eval(name);
                if proWcase
                     name=['W', num2str(i), 'kj=W',num2str(i),';'];
                     eval(name);
                     name=['b', num2str(i), 'kj=b',num2str(i),';'];
                     eval(name);
                end
            end
        end
        
        %--------------------------------------------
        % WB update, we take 2 layers as an example as follows 
        %[W2, b2] = updateWb_total(U2+xi2/rho,V1,W2,b2,alpha,rho,rw);
        %[W1, b1] = updateWb_total(U1+xi1/rho,x_train,W1,b1,alpha,rho,rw);
        if index==0
            %Rwkj=0;
            for i=1:layer
                switch normcase 
                    case 1
                        name=['[W', num2str(i),',b',num2str(i),']=updateWb_total(U', num2str(i),'+xi', num2str(i),'/rho,V', num2str(i-1),...
                            ',W', num2str(i),',b', num2str(i),',alpha,rho,rw',num2str(i),');'];
                    case 2
                        name=['[W', num2str(i),',b',num2str(i),']=updateWb_total2(U', num2str(i),'+xi', num2str(i),'/rho,V', num2str(i-1),...
                            ',W', num2str(i),',b', num2str(i),',alpha,rho,rw',num2str(i),');'];
                    case 3
                        name=['[W', num2str(i),',b',num2str(i),']=bblinesearch(W', num2str(i),',V', num2str(i-1),',U', num2str(i),',b',...
                            num2str(i),',xi', num2str(i),',rho,alpha,rw', num2str(i),');'];
                    case 4
                        name=['[W', num2str(i),',b',num2str(i),']=bblinesearch4(W', num2str(i),',V', num2str(i-1),',U', num2str(i),',b',...
                            num2str(i),',xi', num2str(i),',rho,alpha,rw', num2str(i),');'];
                end
                eval(name);
                %name=['Rwkj=Rwkj+rw/n/2*sum(sum(W',num2str(i),'.^2));'];
                %eval(name);
            end
        end
        %--------------------------------------------

        if kj==1 && bcdinicase==2
            [V1,U1] = updateUV_dnn_lerelu((rv1+gamma2)/beta,(rho + 0)/beta,(rv+gamma2*V1+W2'*xi2)/beta,(0*U1+rho*(repmat(b1,1,N)+W1*x_train)-xi1)/beta);
        else 
            [V1,U1] = updateUV_dnn_lerelu((rv1+gamma2)/beta,(rho + 0)/beta,(rv+gamma2*V1+W2'*xi2+rho*W2'*(U2-(W2*V1+repmat(b2,1,N))))/beta,(0*U1+rho*(repmat(b1,1,N)+W1*x_train)-xi1)/beta);
        end
        if layer>2
            for i=layer-1:-1:2
                %two kinds of updating method. 1.2-blcok BCD 2. L+1-block BCD
%                 [Vi,Ui] = updateUV_dnn((rv1+gammai+1)/beta,(gammai+1 + 0)/beta,(rv+gammai+1*Vi+Wi+1'*xi(i+1))/beta,(gamma2*Ui-xii)/beta);
                if sgscase==0
                     name=['[V', num2str(i), ',U',num2str(i),']=updateUV_dnn_lerelu((gamma',num2str(i+1),'+rv1)/beta,gamma',num2str(i),'/beta,',...
                          '(rv+gamma',num2str(i+1),'*V',num2str(i),'+W',num2str(i+1),'''*xi',num2str(i+1)...
                          ,'+rho*W',num2str(i+1),'''*(U',num2str(i+1),'kj-(W',num2str(i+1),'*V',num2str(i),'+repmat(b',num2str(i+1),',1,N))))/beta,',...
                          '(gamma',num2str(i),'*U',num2str(i),'-xi',num2str(i),...
                          '-rho*(U',num2str(i),'-(W',num2str(i),'*V',num2str(i-1),'kj+repmat(b',num2str(i),',1,N))))/beta);'];
                else
                %[Vi,Ui] = updateUV_dnn((rv1+gammai+1)/beta,(rho + 0)/beta,(rv+gammai+1*Vi+Wi+1'*xii+1
                %          +rho*Wi+1'*(Ui+1-(Wi+1*Vi+repmat(bi+1,1,N))))/beta,(0*Ui+rho*Uiini-xii)/beta);
                     name=['[V', num2str(i), ',U',num2str(i),']=updateUV_dnn_lerelu((gamma',num2str(i+1),'+rv1)/beta,rho/beta,',...
                           '(rv+gamma',num2str(i+1),'*V',num2str(i),'+W',num2str(i+1),'''*xi',num2str(i+1)...
                            ,'+rho*W',num2str(i+1),'''*(U',num2str(i+1),'-(W',num2str(i+1),'*V',num2str(i),'+repmat(b',num2str(i+1),',1,N))))/beta,',...
                            '(-xi',num2str(i),'+rho*(W',num2str(i),'*V',num2str(i-1),'+repmat(b',num2str(i),',1,N)))/beta);'];
                end
                eval(name);
            end   
        end
        % 1. exact solution 
        %[V2,U2] = updateUV_dnn((1+0)/beta,(rho + 0)/beta,(rv+y_train+0*V2)/beta,(0*U2+rho*(repmat(b2,1, N)+W2*V1)-xi2)/beta);
        %2. inexact solution
        %[V2,U2] = updateUV_dnn((1+gamma)/beta,(rho + gamma)/beta,(rv+y_train+gamma*V2)/beta,(gamma*U2+rho*repmat(b2,1,N)+rho*W2*V1-xi2)/beta);  
        name=['[V', num2str(layer), ',U',num2str(layer),']=updateUV_dnn_lerelu((1+rv1)/beta,rho/beta,',...oi
            '(rv+y_train)/beta,(-xi',num2str(layer),'+rho*(repmat(b',num2str(layer),',1, N)+W',num2str(layer),'*V',num2str(layer-1),'))/beta);'];
        eval(name);
        if sgscase>1
            if layer>2
                for i=layer-1:-1:2
                    %two kinds of updating method. 1.2-blcok BCD 2. L+1-block BCD
                    %[Vi,Ui] = updateUV_dnn((rv1+gammai+1)/beta,(rho + 0)/beta,(rv+gammai+1*Vi+Wi+1'*xii+1
                    %           +rho*Wi+1'*(Ui+1-(Wi+1*Vi+repmat(bi+1,1,N))))/beta,(0*Ui+rho*Uiini-xii)/beta);
                    name=['[V', num2str(i), ',U',num2str(i),']=updateUV_dnn_lerelu((gamma',num2str(i+1),'+rv1)/beta,rho/beta,',...
                           '(rv+gamma',num2str(i+1),'*V',num2str(i),'+W',num2str(i+1),'''*xi',num2str(i+1)...
                            ,'+rho*W',num2str(i+1),'''*(U',num2str(i+1),'-(W',num2str(i+1),'*V',num2str(i),'+repmat(b',num2str(i+1),',1,N))))/beta,',...
                            '(-xi',num2str(i),'+rho*(W',num2str(i),'*V',num2str(i-1),'+repmat(b',num2str(i),',1,N)))/beta);'];
                    eval(name);
                end   
            end
            [V1,U1] = updateUV_dnn_lerelu((rv1+gamma2)/beta,(rho + 0)/beta,(rv+gamma2*V1+W2'*xi2+rho*W2'*(U2-(W2*V1+repmat(b2,1,N))))/beta,(0*U1+rho*(repmat(b1,1, N)+W1*x_train)-xi1)/beta);
        end
        %--------------------------------------------
                 %--------------------------------------------
        % WB update, we take 2 layers as an example as follows 
        %[W2, b2] = updateWb_total(U2+xi2/rho,V1,W2,b2,alpha,rho,rw);
        %[W1, b1] = updateWb_total(U1+xi1/rho,x_train,W1,b1,alpha,rho,rw);
        if proWcase && normcase==3
            for i=1:layer
                name=['[W', num2str(i),',b',num2str(i),']=bblinesearch(W', num2str(i),',V', num2str(i-1),',U', num2str(i),',b',...
                            num2str(i),',xi', num2str(i),',rho,alpha,rw',num2str(i),');'];
                eval(name);
                %name=['Rwkj=Rwkj+rw/n/2*sum(sum(W',num2str(i),'.^2));'];
                %eval(name);
            end
        end
        if proWcase && normcase==4
            for i=1:layer
                name=['[W', num2str(i),',b',num2str(i),']=bblinesearch4(W', num2str(i),',V', num2str(i-1),',U', num2str(i),',b',...
                            num2str(i),',xi', num2str(i),',rho,alpha,rw',num2str(i),');'];
                eval(name);
                %name=['Rwkj=Rwkj+rw/n/2*sum(sum(W',num2str(i),'.^2));'];
                %eval(name);
            end
        end
        %--------------------------------------------
        %-------------------------------------------- 
        %compute the fval
        U1ini=repmat(b1,1, N)+W1*x_train;
        error1=U1-U1ini;
        name=['losskj(kj+1) = 1/2/n*dist2(V', num2str(layer), ',y_train)+1/2/rho/n*(sum(sum((xi1+rho*error1).^2))-sum(sum(xi1.^2)));'];
        eval(name);
        for i=1:layer
            name=['losskj(kj+1) = losskj(kj+1) +beta/n*sum(sum(V',num2str(i),'-max(U',num2str(i),',0)))+rv1/n/2*sum(sum(V',num2str(i),'.^2))',...
                    '+rv/n*sum(sum(V',num2str(i),'));'];
            eval(name);
            switch normcase
                case 1
                    name=['losskj(kj+1)=losskj(kj+1)+rw',num2str(i),'/n/2*sum(sum(W',num2str(i),'.^2))+rw/n/2*sum(b',num2str(i),'.^2);'];
                case 2
                    name=['losskj(kj+1)=losskj(kj+1)+rw',num2str(i),'/n/2*sum(sum(W',num2str(i),'.^2));'];
                case 3
                    name=['losskj(kj+1)=losskj(kj+1)+rw',num2str(i),'/n/2*columnnormp(W',num2str(i),');'];
                case 4
                    name=['losskj(kj+1)=losskj(kj+1)+rw',num2str(i),'/n/2*columnnormp(W',num2str(i),');'];
            end
            eval(name);
            if i>1
                name=['error', num2str(i), '=U', num2str(i), '-repmat(b', num2str(i), ',1,N)-W', num2str(i), '*V', num2str(i-1), ';'];
                eval(name);
                name=['losskj(kj+1) = losskj(kj+1)+1/2/rho/n*(sum(sum((xi',num2str(i),'+rho*error',num2str(i),...
                    ').^2))-sum(sum(xi',num2str(i),'.^2)));'];
                eval(name);
            end
        end
        bcdvio = losskj(kj+1)-losskj(kj-index);
        if k>1 && bcdvio>min(epk,1e-3) && kj<bcdmaxiter
            for i=1:layer
                name=['V', num2str(i), '=V',num2str(i),'kj;'];
                eval(name);
                %name=['W', num2str(i), '=W',num2str(i),'kj;'];
                %eval(name);
                name=['U', num2str(i), '=U',num2str(i),'kj;'];
                eval(name);
                %name=['b', num2str(i), '=b',num2str(i),'kj;'];
                %eval(name);
            end
            for i=2:layer
                %name=['gamma', num2str(i), '=(max(1,eigenW',num2str(i),').^2)*max(rho,1);'];
                name=['gamma', num2str(i), '=gamma', num2str(i),'*eigenW',num2str(i),';'];
                eval(name); 
            end
            index=index+1;
            continue;
        end
        if bcdvio>-epk 
            break;
        end  
        index=0;
    end
     time(k) = toc;
    %-------------------------------------------- 
    
    %-------------------------------------------- 
    %test the accuracy for the training data 
    %the accuracy for the test data is be also tested
    if  (losskj(1)>=losskj(kj+1)-epk) || layer<3 || k<=2 || bcdinicase==3
        for i=1:layer
            name=['a',num2str(i),'_train = max(U',num2str(i),'ini,0);'];
            eval(name);
            if i<layer 
                name=['U',num2str(i+1),'ini=repmat(b',num2str(i+1),',1, N)+W',num2str(i+1),'*a',num2str(i),'_train;'];
                eval(name);
            end
        end
        name=['loss3(k) = 1/n/2 *dist2(a',num2str(layer),'_train,y_train);']; 
        eval(name);
        name=['[~,pred] = max(a',num2str(layer),'_train);'];
        eval(name);
        correct_train = pred == train_y';
        acc_train(k) = mean(correct_train);
        a1_test = max(repmat(b1,1, N2)+W1*x_test,0);
        for i=2:layer
            name=['a',num2str(i),'_test = max(repmat(b',num2str(i),',1, N2)+W',num2str(i),'*a',num2str(i-1),'_test,0);'];
            eval(name);
        end
        name=['[~,pred] = max(a',num2str(layer),'_test);'];
        eval(name);
        correct_test = pred == test_y';
        acc_test(k) = mean(correct_test);
        for i=1:layer
            name=['loss4(k) =  loss4(k) +dist2(V',num2str(i),',max(U',num2str(i),',0))/n;'];
            eval(name);
        end
        if k>1 && loss3(k)>1.02*loss3(k-1) && 1.01*acc_train(k)<acc_train(k-1) 
%             if bcdinicase==3
%                 bcdinicase=2;
%                 rho=min(1000, rho*1.1);
%             else 
                bcdinicase=3;
                rho=min(1000, rho*1.1);
%             end
                epk=epk/1.2;
        else 
            %choosen the case 1
%             name=['rho-gamma',num2str(layer),'>0;'];
%             if eval(name) && 1.1*loss1(k)<loss3(k) && bcdinicase~=1
%                 bcdinicase=1;
%             else
            bcdinicase=2;
%             end
        end
    else
        bcdinicase=3;
        %bcdmintotal=min(bcdmintotal+1,10);
        rho=min(1000, rho*1.2);
        epk=epk/1.2;
%         epk=epk/1.2;
    end
    
    tic;
    
    if bcdinicase<3 || k==1
        for i=1:layer
            name=['xi', num2str(i), '=xi', num2str(i), '+tau*rho*error', num2str(i), ';'];
            eval(name);
            name=['loss2(k) =loss2(k) +dist2(0,error', num2str(i),')/n;'];
            eval(name);
        end
        loss1(k)=losskj(kj+1)+tau* rho * loss2(k);
        if k>feasiindex  && (loss2(k)>=max(loss2(k-feasiindex:k-1)) || var(loss3(k-4:k-1))/loss3(k-1)<tortrainerr)%max([loss2(k-3),loss2(k-2),loss2(k-1)])
            rho=min(1000, rho*1.2);
            %xi1=xi1-xi1;xi2=xi2-xi2;
            epk=epk/1.5;
            if epk<=1e-6 
                break;
            end
%             if bcdinicase==1
%                 rhovar=rho1-rho;
%                 for i=1:layer
%                     name=['losskj(kj+1) = losskj(kj+1)+1/2/rhovar/n*(sum(sum(xi',num2str(i),'+rhovar*error',num2str(i),...
%                             ').^2)-sum(sum(xi',num2str(i),'.^2)));'];
%                     eval(name);
%                 end
%                 index=0;
%             end
%             rho=rho1;
        end
        if  k>1 && loss4(k)>10*loss2(k)
            beta=beta*1.2;
        end
    else 
        loss1(k)=loss1(k-1);
        loss2(k)=loss2(k-1);
        acc_train(k)=acc_train(k-1);
        loss3(k)=loss3(k-1);
        loss4(k)=loss4(k-1);
        acc_test(k)=acc_test(k-1);
    end
    
    time(k) = time(k) + toc;
    
    %print results
    fprintf('Epoch: %d \\ %d \n - time: %.5f - lfv_loss: %.8f - feaUW_loss: %.16f- feaVU_loss: %.16f -  trainerr: %.5f - acc: %5f\n - acc_test: %5f - rho: %f - error: %.5f\n',...
            k , niter, time(k),loss1(k), loss2(k),loss4(k),loss3(k), acc_train(k),acc_test(k),rho,epk);

    if rho>999
        break;
    end
        

end





function [WW,bb] = updateWb_total(U, V, W, b, alpha, rho,rwb)
%rw+rb, F norm
    if nargin<7
        rwb=0;
    end
    [d,N] = size(V);
    I = eye(d+1);
    
    Wb=[W,b];VEN=[V',ones(N,1)];
    WWden = (alpha+rwb)*I+rho*(VEN'*VEN);
    WWmol = alpha*Wb+rho*U*VEN;
    Wstar = WWmol/WWden;
    WW = Wstar(:,1:(end-1));
    bb = Wstar(:,end);
end



function [WW,bb] = updateWb_total2(U, V, W, b, alpha, rho,rwb)
%rwb>0
%rw, F norm
    [d,N] = size(V);
    I = eye(d+1);
    I2  = [eye(d),zeros(d,1) ;  zeros(1,d+1)];
    Wb=[W,b];VEN=[V',ones(N,1)];
    WWden = alpha*I+rwb*I2+rho*(VEN'*VEN);
    WWmol = alpha*Wb+rho*U*VEN;
    Wstar = WWmol/WWden;
    WW = Wstar(:,1:(end-1));
    bb = Wstar(:,end);
end

function [V,U,f] = updateUV_dnn(a,b,c,d,index)
    %min  a/2(x-c/a)^2+b/2(y-d/b)^2+x-max(y,0)
    %s.t. x>=y, x>=0
    %V represents the solution for x, V represents the solution for y
    %for ReLU
    if nargin<5
        index=0;
    end
    
    U=zeros(size(c));
    V=zeros(size(c));
    
    c=c/a;d=d/b;e=a+b;
    
    dis1 = c-1/a;
    dis2 = d+1/b;
    dis3 = c*a+d*b;
    dis4 = c-1/a/2;
    dis5 = d+1/b/2;
    dis6 = a*b*(c-d).*(c-d)/e/2;

    
    index1 = d>=0;  
    index2 = ~index1;
    index3 = dis1 >= dis2; 
    index4 = ~index3;
    index5 = dis3 > 0;
    index6 = dis6 > dis4;
    index7 = ~index6;
    index8 = dis1 >= 0;
    index9 = ~index8;
    index10 = dis5 >= 0;
    index11 = ~index10;
    
    indicies1 =  index3 & index10;
    V(indicies1)=dis1(indicies1);
    U(indicies1)=dis2(indicies1);
    
    indicies2 = (index1 & index4 & index5) | (index2 & index4 & index8 & index7) | (index2 & index4 & index9 & index5 & dis6<a*c.*c/2);
    V(indicies2)=dis3(indicies2)/e;
    U(indicies2)=V(indicies2);
    
    indicies3 = ( index8  & index11 ) ;
    V(indicies3)=dis1(indicies3);
    U(indicies3)=d(indicies3);
    
    indicies4 = ~(indicies1 | indicies2 | indicies3) & index2 & index9;
    V(indicies4)=0;
    U(indicies4)=d(indicies4);
    
    if index==0
        f = 0;
    else
        f = a/2*(V-c).^2+b/2*(U-d).^2+V-max(U,0);
        f = sum(sum(f));
    end
%         [V,U,f] = updateUV_dnn_lerelu(a,b,c,d,0,index);
end


function [V,U,f] = updateUV_dnn_iden(a,b,c,d,index)
    %min  a/2(x-c/a)^2+b/2(y-d/b)^2+x-y
    %s.t. x>=y,
    %V represents the solution for x, V represents the solution for y
    %for iden
    if nargin<5
        index=0;
    end
    
    index1 = a*d+a+b-b*c>0;
    index2 = ~index1;
    
    dist1 = (c+d)/(a+b);
    
    V(index1)=dis1(index1);
    U(index1)=dis1(index1);
    V(index2)=(c-1)/a;
    U(index2)=(d+1)/b;
    
    if index==0
        f = 0;
    else
        f = a/2*(V-c/a).^2+b/2*(U-d/b).^2+V-U;
        f = sum(sum(f));
    end
end
function [V,U,f] = updateUV_dnn_lerelu(a,b,c,d,arelu,index)
    %min  a/2(x-c/a)^2+b/2(y-d/b)^2+x-max(y,arelu*y)
    %s.t. x>=y,x>=arelu y
    %V represents the solution for x, V represents the solution for y
    %for leReLU, arelu=0.1
    % general, 0<arelu<1
    %if arelu=0. updateUV_dnn_lerelu equals updateUV_dnn
    if nargin<6
        index=0;
    end
    if nargin<5
        arelu=0;
    end
    if arelu<0
        [V,U,f] = updateUV_dnn_iden(a,b,c,d,index);
        return;
    end
    if arelu==0 
        [V,U,f] = updateUV_dnn(a,b,c,d,index);
        return;
    end
    
    U=zeros(size(c));
    V=zeros(size(c));
    
    r1=a*d-b*c;
    r2=arelu*a*d-b*c;
    
    dis1 = (c-1)/a;
    dis2 = (d+1)/b;
    dis3 = d+(1+arelu)/2;
    dis4= (d+arelu)/b;
    dis5 = arelu* (d+arelu)/b;
    e1=(a+b);
    dis6 = (c+d)/e1;
    dis7 = arelu* c+d ;
    e2 = arelu*arelu*a+b;
    e3=r2*r2/e2;
    e33=r1*r1/e1;
    dis8 = e33-e3;
    e4 = c/a-1/2/a-arelu*d/b-arelu*arelu/2/b-e33/2/a/b;
    
    index1 = dis1 >= dis2; 
    index2 = ~index1;
    index3 = dis3>=0;
    index4 = ~index3;
    index6 = dis6>0;
    index7 = dis1 >= dis5; 
    index8 = ~index7;
    index9 = dis7 > 0;
    index10 = ~index9;
    index11 = dis8<=0;
    index12 = dis5<0;
    index13 = ~index12;
    
%    indicies1 =  ((index4 & dis2>=0 & index8) | index3) & index1;
    indicies1 =  ( index3) & index1;
    V(indicies1)=dis1(indicies1);
    U(indicies1)=dis2(indicies1);
    
    indicies3 =  ((index3 & index12 & index2) | index4)  & index7  ;
    V(indicies3)=dis1(indicies3);
    U(indicies3)=dis5(indicies3)/arelu;
    
    indicies2 = index2 & index6 & ((index7 & (index13 | (index12 & e4>0) ) ) | (index8 & (index9 | (index10 & index11)))) ;
    V(indicies2)=dis6(indicies2);
    U(indicies2)=V(indicies2);
    
    indicies4 = ~(indicies1 | indicies2 | indicies3) & index8 & index10;
    V(indicies4)=dis7(indicies4)/e2*arelu;
    U(indicies4)=dis7(indicies4)/e2;

    
    if index==0
        f = 0;
    else
        f = a/2*(V-c/a).^2+b/2*(U-d/b).^2+V-max(U,arelu*U);
        f = sum(sum(f));
    end
end

function [W,b,U,V] = iniDNN(d_hidden,testdata)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here
layer=size(d_hidden,1)-1;
d=d_hidden;
n=size(testdata,2);
V0=testdata;
W=[];b=[];U=[];V=[];
for i=1:layer
    name=['W', num2str(i), '=randn(d(i+1),d(i))*0.01;']; % he initialization
    eval(name);
    name=['b', num2str(i), '=0.1*zeros(d(i+1),1);'];
    eval(name);
    name=['U', num2str(i), '=W',  num2str(i), '*V', num2str(i-1),'+repmat(b', num2str(i), ',1,n);'];
    eval(name);
    if i==layer
        name=['V', num2str(i), '=U',  num2str(i),';'];
        eval(name);
    else
        name=['V', num2str(i), '=max(0,U',  num2str(i),');'];
        eval(name);
    end
    name=['W=[W;W', num2str(i), '(:)];'];
    eval(name);
    name=['V=[V;V', num2str(i), '(:)];'];
    eval(name);
    name=['U=[U;U', num2str(i), '(:)];'];
    eval(name);
    name=['b=[b;b', num2str(i), '(:)];'];
    eval(name);
end

end

