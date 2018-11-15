# Based-on-LSTM-stock-average-price-predict </br>
1. m文件是用来处理数据的，通过原始的股价数据来计算出我们需要的对数收益率和波动率</br>
2. py文件是深度学习模型程序，输入为input.mat文件和output.mat文件，第一个文件为股价和对数收益率向量，第二个文件为波动率的真实值</br>
3. 程序运行过程中会生成一个.npy文件，用于储存预测的波动率。</br>
4. plot.rar里面为画图数据文件，plot.py为画图的Python。</br>
5. stock.R为传统方法对波动率的预测。</br>
