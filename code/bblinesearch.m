function [Wk2,bk2,fval,kktviolation,time]= bblinesearch(W1,V0,U1,b1,xi1,rho,gamma,alphaW)
    
    tic;
    alphaW=alphaW/2;
    U11= U1 + xi1/rho;
    normV=min(1e-3,1/norm(V0,'fro'));
    Iter=100;
    [~,N] = size(U1);
    Wk=W1;bk=b1;
    VEN=[V0',ones(N,1)];
    VENTT=VEN'*VEN;
    Wk1=Wk; bk1=bk;
    Wk2=Wk; bk2=bk;
    grapho1=grarho(U11, Wk1, Wk, VEN, VENTT, bk1, bk, rho, gamma);
    fval=zeros(Iter,1);
    fval(1)=fvalWb(U11, Wk2-Wk, Wk2, V0, bk2-bk, bk2, rho, gamma, alphaW);
%     aatoc=0;
    
    for k=1:Iter
        grapho2 = grarho(U11, Wk2, Wk, VEN, VENTT, bk2, bk, rho, gamma);
%         normW=grapho2(:,1:(end-1)).*Wk2;
%         normW=sum(sum(abs(normW.*normW)))/d4/d3;
%         fprintf('%f\n',normW);
%         if k>2 && normW < 1e-6
%             break;
%         end
        deltag = grapho2 - grapho1;
        deltax = [Wk2-Wk1, bk2-bk1];
        if (mod(k,2))
            deltax = sum(sum(deltax.*deltax));
            deltag = sum(sum(deltax.*deltag));
            alpha = min(normV, abs(deltax/deltag));
%             if k>2 && deltax< 1e-3
%                 break;
%             end
%              fprintf('%f\n',deltax);
        else
            deltax = sum(sum(deltax.*deltag));
            deltag = sum(sum(deltag.*deltag));
            alpha = min(normV, abs(deltax/deltag));
        end
        if k==1
            alpha=normV;
        end
%         if mod(k,5)==0
%             fval(k)=fvalWb(U11, Wk1-Wk,  Wk1, V0, bk1-bk, bk1, rho, gamma, alphaW);
%         end
        bk1 = bk2; Wk1 = Wk2;grapho1 = grapho2;
        for j=1:2
            bk2 = bk1 - alpha * grapho2(:,end);
            Wk2 = proxl21(Wk1 - alpha * grapho2(:,1:(end-1)),alphaW*alpha);
%             aatic=tic;
            fval(k+1)=fvalWb(U11, Wk2-Wk,  Wk2, V0, bk2-bk, bk2, rho, gamma, alphaW);
             if  fval(k+1)<fval(k)*2
                break
             else
%             aatoc=aatoc+toc(aatic);
                alpha=alpha/10;
            end
        end
        normW=abs(fval(k+1)-fval(k))/fval(k+1);
%         fprintf('%f\n',normW);
        %mod(k,5)==0 &&
        if normW<1e-3
           % break;
        end
%         kktviolation = kktvio(U11, Wk2, Wk, VEN, VENTT, bk2, bk, rho, gamma, alphaW);
%         fprintf('%f\n',kktviolation);
    end
    time=toc;
    fvalend=fval(k+1);    
    kktviolation = kktvio(U11, Wk2, Wk, VEN, VENTT, bk2, bk, rho, gamma, alphaW);
end

function fval=fvalWb(U11, W3kdiff, W3, V0, b3kdiff, b3, rho, gamma, alphaW)
    [~,N] = size(U11);
    DiffU=U11-W3*V0-repmat(b3,1,N);
    fval= alphaW*columnnormp(W3) + gamma * sum(sum(W3kdiff.*W3kdiff))/2+ gamma * sum(sum(b3kdiff.*b3kdiff))/2 + rho*sum(sum(DiffU.*DiffU))/2;
end

function gra=grarho(U11, W3, W3k, VEN, VENTT, b3, b3k, rho, gamma)
    gra= -rho*U11*VEN + rho*[W3, b3]*VENTT + gamma*[W3 - W3k,b3 - b3k];
end

function gra=kktvio(U11, W3, W3k, VEN, VENTT, b3, b3k, rho, gamma, alphaW)
    gra = 0;
    [d4,d3]=size(W3);
    graW2 = grarho(U11, W3, W3k, VEN, VENTT, b3, b3k, rho, gamma);
    graW2 = graW2(:,1:(end-1));
    for i=1:d3
        normW2=norm(W3(:,d3),2);
        if normW2
            graW2d3=graW2(:,d3)+alphaW*normW2*W3(:,d3);
        else 
            graW2d3=0;
        end
        gra=gra+sum(abs(graW2d3));
    end
    gra=gra/d4/d3;
end

