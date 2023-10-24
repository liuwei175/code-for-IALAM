function [x_train,y_train,x_test,y_test,trainini_y,testini_y] = mnist_datanum(N,train_x,test_x,train_y,test_y)
    
    train_x = train_x'/255;
    test_x = test_x'/255;
    %input, training set
    %dimension
    K=10;

    index1=zeros(10,1);
    index2=zeros(10,1);
    j1=1;j2=1;
    x_train=zeros(784,N*10);
    y_train = zeros(K,N*10);
    trainini_y = zeros(N*10,1);
    x_test=zeros(784,N*2);
    y_test = zeros(K,N*2);
    testini_y = zeros(N*2,1);
    for i=1:60000
        yy=train_y(i);
        index1(yy)=index1(yy)+1;
        if index1(yy)<N+1
            x_train(:,j1)=train_x(:,i);
            y_train(yy,j1)=1;
            trainini_y(j1)=yy;
            j1=j1+1;
        end
        if j1>10*N
            break;
        end
    end
    
    for i=1:10000
        yy=test_y(i);
        index2(yy)=index2(yy)+1;
        if index2(yy)<ceil(N/5)+1
            x_test(:,j2)=test_x(:,i);
            y_test(yy,j2)=1;
            testini_y(j2)=yy;
            j2=j2+1;
        end
        if j2>N*2
            break;
        end
    end

end
