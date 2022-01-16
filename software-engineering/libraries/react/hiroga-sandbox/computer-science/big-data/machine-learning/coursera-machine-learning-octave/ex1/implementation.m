function implementation()

  data = load('ex1data1.txt');
  x = data(:,1);
  y = data(:,2);
  m = length(y); 
  X = [ones(m,1),data(:,1)];
  theta = zeros(2,1);
  
  num_iters = 1500;
  alpha = 0.01;
  