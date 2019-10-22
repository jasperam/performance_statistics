# 业绩统计设计文稿

> [项目地址](https://github.com/jasperam/performance_statistics.git)

## 数据需求

主要需要的数据如下：

* 日数据
  1. 分仓后的持仓（已有）
  2. 分仓后的成交明细，需要有**对应的指令时间**用以给出vwap的价格（待补）
  3. T-1日的总资产（已有）
  4. 每个股票的分钟vwap数据（已有）
  5. 股票和bm的日数据（已有）
* 配置信息
  1. 模型的BM和alpha计算方式（已有）
  2. 产品的佣金费率（已有）
  3. alpha的收益调整比例（已有）

## 模型计算方式

现有的计算方式有如下几种：

> 如果用了算法交易，成交明细的价格为vwap价格加上冲击成本（3% 1bp; 3%~5% 2bps; 超过5%的不记入算法交易中）

* model每天的交易基本是market value neutral
  * 以T-1日市值做为总资产
  * 计算T日的持仓和交易pnl
  * **需要考虑加减仓时的alpha如何计算（待补）**
* ~~model每天的交易不是market value neutral~~
  * ~~原则是计算每只股票实际持有时间段的alpha，减去对应时间段的bm~~
  * ~~以T-1日的市值做为总资产计算每个股票的占比~~
  * ~~T-1日的买单，从买入时间到T日收盘的alpha算做T日的alpha~~
  * ~~T日的卖单，从昨收盘到卖出时间的alpha算做T日的alpha~~
  * ~~T日的alpha=T日的未交易持仓alpha + T-1日的买交易alpha + T日的卖交易alpha~~
* 以产品为计量单位计算
  * alpha = return - bm
* 算法交易
  * pnl = (实际成交价格 - 调整后的vwap价格) * 成交数量
  * alpha = pnl / 成交总额
* T0
  * pnl = (实际成交价格 - 收盘价) * 成交数量
  * alpha = pnl / T日可用持仓市值

## 工作流图

![](.\doc\workflow.png)