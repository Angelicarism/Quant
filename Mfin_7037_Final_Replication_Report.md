# Mfin_7037_Final_Replication_Report

- Paper Name: EARNINGS EXTRAPOLATION AND PREDICTABLE STOCK MARKET RETURNS(Author: Hongye Guo)

- Reproduciton: Table1, Table2, Table3, Table4

- Author: CHENG XINYI 3035888449,GUO JING 3035878860,YAN YANGTIAN 3035888231

# 1. Data Documentation

To replicate the four tables in the paper, we mainly used five data files.

|                                                              | Abstract | Time Range | Shape | Used Table       |
| ------------------------------------------------------------ | -------- | ---------- | ----- | ---------------- |
| ibes.det_epsus.parquet                                       |          |            |       | table 1, table2  |
| worldscope2.wrds_ws_company.parquet                          |          |            |       | table 1          |
| comp_fundq.parquet                                           |          |            |       | table 1, table 2 |
| ff_four_factor_monthly.parquet                               |          |            |       | table 3, table 4 |
| crsp_msf.parquet（这个有用到么，我看代码里面好像读了之后就没用了 |          |            |       |                  |



# 2. Table 1 Reproduction

## 2.1 Method

For table 1, it describes the company-fiscal period counted by fiscal period end month and it contains two markets: US market and global market. Firstly, we did data preprocessing. We dropped the duplicates and the null data. Then we defined a classification function to classify months into the three different groups. After we finished the steps and got the data we wanted in both US market and global market, we concatenated them into one table and eventually did the table 1 replication.

## 2.2 Main Code

```python
#US data preprocessing
det = pd.read_parquet("ibes.det_epsus.parquet")

df = det[["cusip","anndats_act"]].dropna().drop_duplicates()
df2 = comp[["cusip","datadate","fyearq","fqtr","fyr"]].dropna().drop_duplicates()
df2["cusip"]=df2["cusip"].apply(lambda x:str(x[0:8]))
df3=pd.merge(df,df2,on="cusip")
df3["anndats_act"]=pd.to_datetime(df3["anndats_act"])
df3["datadate"]=pd.to_datetime(df3["datadate"])
df3["diff"]=(df3["anndats_act"]-df3["datadate"])/np.timedelta64(1,"D")
df3["diff"]=df3["diff"].astype(int)
df3=df3[(df3["diff"]>=0)]
df3.drop_duplicates(subset=["cusip","anndats_act"],keep="last",inplace=True)
df3["anndats_act"]=df3["anndats_act"].dt.month

def countMonth(df,classifier):
    """
    This function is used to classify month into the three different groups by 
    using modulo funcion.
    #Group1:Jan/Apr/Jul/Oct
    #Group2:Jan/Apr/Jul/Oct
    #Group3:Mar/Jun/Sep/Dec
    :param df: The dataframe
    :param classifier:The classifier used to divided dataframe into different group
    :return: Classified result
    """
    #[count,percent]
    result=[[0,0],[0,0],[0,0]]
    for subdf in df.groupby(classifier):
        if subdf[0]%3==1:
            result[0][0]+=subdf[1].shape[0]
        elif subdf[0]%3==2:
            result[1][0]+=subdf[1].shape[0]
        else:
            result[2][0]+=subdf[1].shape[0]
    result[0][1]=round((result[0][0]/df.shape[0])*100,2)
    result[1][1]=round((result[1][0]/df.shape[0])*100,2)
    result[2][1]=round((result[2][0]/df.shape[0])*100,2)
    return result

result=countMonth(df3,"fyr")
table1=pd.DataFrame({"Count":[result[0][0],result[1][0],result[2][0],df3.shape[0]],"Percent":[
        result[0][1],result[1][1],result[2][1],100]},index=["Group 1: Jan/Apr/Jul/Oct",
    "Group 2: Feb/May/Aug/Nov", "Group 3: Mar/Jun/Sep/Dec", "Total"])
```

```python
#Global data preprocessing
wsc.rename(columns={"item5352_month_of_fiscal_year_end":"fye"},inplace=True)
month_names = ['JAN', 'FEB', 'MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
month_dict = {v:f'{k:2d}' for k,v in enumerate(month_names, start=1)}
wsc["fye"] = wsc["fye"].map(month_dict)#Change month abbreviation to the digital format
wsc["fye"]=wsc["fye"].dropna().astype(int)
result1=countMonth(wsc,"fye")

table1_global=pd.DataFrame({"Count":[result1[0][0]+result[0][0],result1[1][0]+result[1][0],result1[2][0]+result[2][0],wsc.shape[0]+df3.shape[0]],"Percent":[
        round((result1[0][0]+result[0][0])/(wsc.shape[0]+df3.shape[0])*100,2),round((result1[1][0]+result[1][0])/(wsc.shape[0]+df3.shape[0])*100,2),
        round((result1[2][0]+result[2][0])/(wsc.shape[0]+df3.shape[0])*100,2),100]},index=["Group 1: Jan/Apr/Jul/Oct",
    "Group 2: Feb/May/Aug/Nov", "Group 3: Mar/Jun/Sep/Dec", "Total"])
```

## 2.3 Result

![1.png](https://github.com/Angelicarism/Quant/blob/main/1.png?raw=true)

# 3. Table 2 Reproduction

## 3.1 Method

Table 2 shows the number of company-quarter by 3 groups, where the groups are determined by the reporting month,
specifically its remainder when divided by 3. The key point is to figure out what the two filters are.

- **Rpt Lag <= 92 filter:** It selects companies which reported within 92 days.
- **Group3_FQ_end filter:** It represents companies, which reports earning in the fiscal month witch can  satisfies the requirement: # mod 3 == 0.

## 3.2 Main Code

```python
#Both filters
both=df3[(df3["diff"]<=92) & (df3["fyr"]%3==0)]
result_both=countMonth(both,"anndats_act")

#rptlag<=92 filters
rptlag=df3[df3["diff"]<=92]
result_rptlag=countMonth(rptlag,"anndats_act")

#group3_fq_end filters
group3_fq_end=df3[df3["fyr"]%3==0]
result_group3_fq_end=countMonth(group3_fq_end,"anndats_act")

#no_filter
no_filter=df3
result_nofilter=countMonth(no_filter,"anndats_act")
```

## 3.3 Result

![2.png](https://github.com/Angelicarism/Quant/blob/main/2.png?raw=true)

## 3.4 Critical Analysis in Table 1 and Table 2

- **Insufficient Data:** The volume of data we used is less than the origin paper, so the result is a slight different from the paper's table. But it does not have a significant influence, cause the point is to focus on the method not the result.

- within one month的问题

# 4. Table 3 Reproduction

## 4.1 Method

This table describes the Lead-lag relations of US monthly market returns. In this part, we used four factors market return to do the regression.

- Regression formula used in column 1, 2 and 3:
  $$
  mkt_t = \alpha + \sum_{j=1}^{4} \beta_j mkt_{nm(t, j)} + \epsilon_t
  $$

- Regression formula used in column 4:
  $$
  mkt_t = \alpha + \sum_{j=1}^{4} \beta_j mkt_{nm(t, j)} + \sum_{j=1}^{4} \gamma_j mkt_{nm(t, j)}*I_t^{nm} + \delta I_t^{nm} + \epsilon_t
  $$
  where $$ I_t^{nm} $$ is a dummy variable taking the value of 1 when month t is newsy.

- 

## 4.2 Main Code

To realize table 3, we built two functions to check the month t is newsy or non_newsy and to generatemktnm(t,j) which is the return in the jth newsy month (Jan, Apr, Jul, Oct)preceding month t.

Then we did ols regression on the grouped data.

```python
def check_states(date):
    """
    This function is used to check the month t is newsy or non_newsy
    :param date: month
    return: 1 for newsy month,0 for non_newsy month
    """
    if date.month % 3 == 1:
        return 1
    else:
        return 0
```

```python
def find_nm(i, data):
    """
    This fucntion is used to generatemktnm(t,j) which is the return in the jth 
    newsy month (Jan, Apr, Jul, Oct)preceding month t. 
    """
    for j in data.index:
        try:
            if data.loc[j,"state"] == 1:
                data.loc[j, "mkt_nm"+str(i)] = data.loc[j-3*i, "mkt"]
            elif data.loc[j-1,"state"] == 1:
                data.loc[j, "mkt_nm"+str(i)] = data.loc[j+2-3*i, "mkt"]
            else:
                data.loc[j, "mkt_nm"+str(i)] = data.loc[j+1-3*i, "mkt"]
        except Exception:
            data.loc[j, "mkt_nm"+str(i)] = np.nan
```

```python
#Regression
data["mkt_nm_total"]=data["mkt_nm1"]+data["mkt_nm2"]+data["mkt_nm3"]+data["mkt_nm4"] 
data["mkt_nm_total_state"]=data["state"]*(data["mkt_nm1"]+data["mkt_nm2"]+data["mkt_nm3"]+data["mkt_nm4"])
all_column=smf.ols('mkt ~ mkt_nm1 + mkt_nm2 + mkt_nm3 + mkt_nm4', data=data).fit(cov_type='HC1', use_t=True)
newsy_month_data=data[data["state"]==1]
newsy_month=smf.ols('mkt ~ mkt_nm1 + mkt_nm2 + mkt_nm3 + mkt_nm4', data=newsy_month_data).fit(cov_type='HC1', use_t=True)
non_newsy_month_data=data[data["state"]==0]
non_newsy_month=smf.ols('mkt ~ mkt_nm1 + mkt_nm2 + mkt_nm3 + mkt_nm4', data=non_newsy_month_data).fit(cov_type='HC1', use_t=True)

data["mkt_nm1*state"]=data["state"]*data["mkt_nm1"]
data["mkt_nm2*state"]=data["state"]*data["mkt_nm2"]
data["mkt_nm3*state"]=data["state"]*data["mkt_nm3"]
data["mkt_nm4*state"]=data["state"]*data["mkt_nm4"]
diff=smf.ols('mkt ~ mkt_nm1*state+mkt_nm2*state+ mkt_nm3*state+mkt_nm4*state +state + mkt_nm1 + mkt_nm2 + mkt_nm3 + mkt_nm4 ', data=data).fit(cov_type='HC1', use_t=True)
```

## 4.3 Result

![3.png](https://github.com/Angelicarism/Quant/blob/main/3.png?raw=true)

## 4.4 Critical Analysis

In this part, we faced problems in realizing the Difference column, we first ran regression just on  mkt_nm, the same regression as in colunm 1 , and we got the wrong result. Then we added dummy variable: state. The regression was based on 9 columns as showed in the codes.

可能需要补充下用dummy variable的理由

# 5. Table 4 Reproduction

## 5.1 Method

Table 4 is similar to table 3, and the key point is to do the regression.

- Regression formula used in column 1, 2 and 3:
  $$
  mkt_t = \alpha + \beta\sum_{j=1}^{4}mkt_{nm(t, j)} + \epsilon_t
  $$

- Regression formula used in column 4, 5, 6 and 7:
  $$
  mkt_t = \alpha + \beta_1\sum_{j=1}^{4} \beta_ mkt_{nm(t, j)} + \beta_2(\sum_{j=1}^{4}mkt_{nm(t, j)})*I_t^{nm} + \beta_3I_t^{nm} + \epsilon_t
  $$
  where $$ I_t^{nm} $$ is a dummy variable taking the value of 1 when month t is newsy.

## 5.2 Main Code

In this part, we mainly ols regression on different groups data.

```python
#Post_ww2 data
post_ww2=data[(data["month"].dt.year>=1946)&(data["month"].dt.year<=2022)]
#First_half data
first_half=data[(data["month"].dt.year>=1926)&(data["month"].dt.year<=1973)]
#Second_half data
second_half=data[(data["month"].dt.year>=1974)&(data["month"].dt.year<=2021)]

#Running regression
all_column_1=smf.ols('mkt ~ mkt_nm_total', data=data).fit(cov_type='HC1', use_t=True)
newsy_month_2=smf.ols('mkt ~ mkt_nm_total', data=newsy_month_data).fit(cov_type='HC1', use_t=True)
non_newsy_month_3=smf.ols('mkt ~ mkt_nm_total', data=non_newsy_month_data).fit(cov_type='HC1', use_t=True)
all_column_4=smf.ols('mkt ~ mkt_nm_total + mkt_nm_total_state + state', data=data).fit(cov_type='HC1', use_t=True)
post_ww2_5=smf.ols('mkt ~ mkt_nm_total + mkt_nm_total_state + state', data=post_ww2).fit(cov_type='HC1', use_t=True)
first_half_6=smf.ols('mkt ~ mkt_nm_total + mkt_nm_total_state + state', data=first_half).fit(cov_type='HC1', use_t=True)
second_half_7=smf.ols('mkt ~ mkt_nm_total + mkt_nm_total_state + state', data=second_half).fit(cov_type='HC1', use_t=True)
```

## 5.3 Result

![4.png](https://github.com/Angelicarism/Quant/blob/main/4.png?raw=true)

## 5.4 Critical Analysis

暂时没想到...