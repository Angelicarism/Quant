# Assignment 1 Report

**Team member:**

- CHENG Xinyi 	3035888449

- GUO Jing 		3035878860

- YAN Yangtian 3035888231

## Solution:

### 1. The market model

#### a.

- **Risk premium**

  Dobrynskaya (2017) showed that the winner portfolio has higher downside risk than the loser portfolio, and lower upside risk exposure, so dynamic risk premium can bring returns to some investors.

- **Mispricing story**

  Investors' wrong expectations for the future of the stock lead to the mispricing of the stock, and the individual bias further affects the group behavior bias, resulting in the momentum effect of the stock

**Reference**:
Dobrynskaya, Victoria, Dynamic Momentum and Contrarian Trading (September 22, 2017). Higher School of Economics Research Paper No. WP BRP 61/FE/2017. Available at SSRN: 

[Dobrynskaya, Victoria, Dynamic Momentum and Contrarian Trading (September 22, 2017). Higher School of Economics Research Paper No. WP BRP 61/FE/2017](https://ssrn.com/abstract=3041227)

#### b.

```python
#Cumulative Average Return
average_return=(ff.mom+1).prod() - 1
geo_average_return=average_return**(1/ff.shape[0])-1
annualize_return=(1+geo_average_return)**12-1
```

```python
#volatility
volatility=ff.mom.std() * np.sqrt(12)
```

```python
#Sharpe ratio
sharpe_ratio=((1/ff.shape[0]) * (ff.mom - ff.rf).sum() / (ff.mom).std())
annualize_sharpe_ratio=sharpe_ratio*np.sqrt(12)
```

Considering that cumulative return is more rational in financial word than simple arithmetic mean return, and by applying the above codes in python, we can get the results:

- Annualized cumulative average return: 5.92%(0.05921465451482111)

- Annualized volatility: 0.152(0.15193145926172957)

- Annualized Sharpe ratio: 0.247(0.24741705357583085)

#### c.

Here are some reasons for different performance:

1. Commission Fee: Many brokers choose to charge a fixed commission fee for each transaction, but this part is not considered in strategy. For AQR, the average amount in each transaction is much larger than that of retailers. So, the proportion commission fee takes is much less for AQR.

2. Bid-ask Spread: In reality, if you trade through brokers, there is bid-ask spread, the price you buy or sell is different from market price. For AQR, the high transaction amount will narrow down the spread and get a better performance than retailors.

3. Liquidity: when transaction volume is high, you may not be able to buy or sell the target amount of stocks the strategy suggests, especially for AQR. The average transaction price is much higher (if you buy) than expected, leading to a lower return than retailer.
   In summary, total trading cost leads to different performance, but whether AQR or retailers have better performance depends on many different factors.

#### d.

If the market goes up by 1%, the momentum will go up by  $\beta_1\%$, which is -0.2542%.

#### e.

```python
import statsmodels.formula.api as smf
smf.ols('mom ~ mkt_rf', data=ff).fit().summary()
```

By using the regression code line, the market beta of the momentum strategy is -0.2542.

#### f.

The $\alpha$ represents the excess return of our model (if our model only has $mkt-r_f$  as factor. The intercept is 0.0076(monthly $\alpha$), by applying the simple computation $(1+0.0076)^{12}-1=0.0951$, we can find that the annualized excess return is 9.51%.

#### g.

We can use t-test to test the significance of regression coefficient. 

We set a hypothesis $H_0:\beta=0$, where $\beta$ is the coefficient of  $mkt-r_f$ . The t-statistic is equal to
$$
t_{\beta}=\frac{\hat{\beta}_{}}{\sqrt{\hat{\sigma}^{2} / L_{x x}}}=\frac{\hat{\beta}_{} \sqrt{L_{x x}}}{\hat{\sigma}}=-11.00
$$
given that $t_{0.001/2,n-1}=\pm3.491$ , where $n=1147$ represents the number of samples.

It allows us to reject the null hypothesis with a confidence higher than 0.999, which means the coefficient $\beta$ is significant.

Besides, we can find the t-statistic of the intercept coefficient $t_\alpha=6.079$, which also allows us to reject the null hypothesis with a confidence higher than 0.999.

In conclusion, the statistical relations we uncovered in (b) and (c) ***are statistically significant***.

#### h.

Under the hypothesis of the single factor regression, we can conclude that the  returns will go up by $-4\beta_1\% =  -4\%*-0.2542 = 1.0168\%$. 

However, the hypothesis of the regression model is that the residual item $\varepsilon$ is Gaussian white noise, which is inconsistent with the real world, for example the SMB factor and HML facotr will appear in the residual item. 

In conclusion, the value $-4\beta_1\%$ may be be inaccurate compared with the condition in real world.

### 2. The Fama-French (1993) three-factor model

#### a.

The momentum will only be expected to go up by $\beta_1\%$ if the value of SMB and HML is constant.

However, if the $r_{mkt}$ goes up, it can affect the value of  SMB and HML, which may then affect the $r_{mom}$, and the momentum will not be expected to go up by $\beta_1\%$.

In conclusion, whether the momentum will be expected to go up by $\beta_1\%$ depends, but actually in real word, $\beta_1\%$ usually is not a precise value.

#### b.

***Small minus big (SMB)*** is the excess return that smaller market capitalization companies return versus larger companies. Since in the long-term, small-cap companies tend to see higher returns than large-cap companies. 

***High Minus Low (HML)*** is a value premium, it represents the spread in returns between companies with a high book-to-market value ratio and companies with a low book-to-market value ratio because companies with a high book to market ratio are typically considered as value stocks.

Reference: 

[Chiah M, Chai D, Zhong A, et al. A Better Model? An empirical investigation of the Fama–French five‐factor model in Australia[J]. International Review of Finance, 2016, 16(4): 595-638.](https://onlinelibrary.wiley.com/doi/full/10.1111/irfi.12099)

[Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of financial economics*, *33*(1), 3-56.](https://www.sciencedirect.com/science/article/abs/pii/0304405X93900235)

### c.

### 3.

#### 3.1

```python
smf.ols('mystery_meat ~ mkt_rf + smb + hml + rmw + cma + mom', data=my_meat).fit().summary()
```

By using the ols regression code line, we can get the direct result:
$$
\text{mystery\_meat = -0.5mkt\_rf - 0.5smb + 0.000004148hml - 0.5rmw + 0.5cma + -0.000001083mom - 0.001}
$$
But the exposure of hml and mom is too small and the t-test value of the two factors is not statistically significant, so the final regression model is:
$$
\text{mystery\_meat = -0.5mkt\_rf - 0.5smb - 0.5rmw + 0.5cma - 0.001}
$$
So the exposure of this strategy to the Fama French factors is:

- long:

  cma: 0.5

- short:

  mkt_rf: -0.5

  smb: -0.5

  rmw: -0.5

#### 3.2

From the result of the regression, we can see that in this strategy, the mystery_meat item can be explained by the above four factors($mkt\_rf, smb, rmw, cma$), theoretically there will be no $\alpha$ in the regression. Therefore, ***the intercept  value -0.0010*** is the transaction cost.
