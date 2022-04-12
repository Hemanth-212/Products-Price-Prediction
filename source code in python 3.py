#importing libraries
import pandas as pd
import numpy as np
import pickle

#reading train data
df1=pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\dataset\\train.csv')

#removing unusual columns
df1=df1.drop(columns=['Product_id','Customer_name','Loyalty_customer'],axis=1)

#these are unusual columns, rows should not be removed cause of this null values
df1['instock_date']=df1['instock_date'].apply(lambda x: int(x.split('-')[1]))  
df1['charges_2 (%)']=df1['charges_2 (%)'].fillna(0)

#this is very importtant column so i filled every null value with -1 later decided to give the value 0 or 1
df1.Discount_avail=df1.Discount_avail.fillna(-1)

#we can fill the null charges_1 by median of respective product_cat for now filling them with 0 
df1.charges_1=df1.charges_1.fillna(0)

#minimun price should not be greater than selling price when discount is given
df1=df1[~(((1.13*df1.Minimum_price/df1.Selling_Price)>1)&(df1.Discount_avail==0))]
#selling price should not be greater than maximum price usually but it happens some times it can't increaced by 50 percent i picked 1.48 manually
df1=df1[~(1.48*df1.Maximum_price<df1.Selling_Price)]

# I removed the outliers when discount is 0 and max_price is almost doubled
df1=df1[~(((df1.Maximum_price/df1.Selling_Price)>1.78)&(df1.Discount_avail==0))] 

#dropping null values
df1=df1.dropna()
    
#removing unusual rows
df1=df1[~(df1.Selling_Price<=0)]
#removing outliers
df1=df1[~(((df1.Maximum_price/df1.Minimum_price)>8))]

#for normalizing
price_range=df1.Selling_Price.max()-df1.Selling_Price.min()

#adding extra average_price column
df1['avg_price']=(df1.Minimum_price+df1.Maximum_price)/(2*price_range)

#function for removing outliers of Maximum_price & Minimum_price and mapping charges_1 medians to respective Product_Category
def mapping_charges(df):
    df_out = pd.DataFrame()
    dic=dict()
    for key, subdf in df.groupby('Product_Category'):
        upper_lim = subdf['Maximum_price'].quantile(.99)
        lower_lim = subdf['Minimum_price'].quantile(.01)
        subd = subdf.copy()
        subd = subd[(subd['Maximum_price'] < upper_lim)&(subd['Minimum_price'] > lower_lim)]
        upper_lim = subdf['avg_price'].quantile(.99)
        lower_lim = subdf['avg_price'].quantile(.01)
        subd = subd[(subd['avg_price'] < upper_lim)&(subd['avg_price'] > lower_lim)]
        m = np.median(subd.charges_1)
        subd.charges_1=subd.charges_1.replace(to_replace=0,value=m)
        dic[key]=m
        reduced_df = subd
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return (df_out,dic)
var=mapping_charges(df1)
dic=var[1]


#as i am calculating based om avg_price i removed outliers
df1=df1[~(((df1.avg_price/df1.Selling_Price)>1)&(df1.Discount_avail==0))]


#creating dummies for 'Product_Category'
dummy1=pd.get_dummies(df1.Product_Category)

#concatanating dummies
df1=pd.concat([df1,dummy1],axis=1)


#creating dummies for 'Demand'
dummy2=pd.get_dummies(df1.Demand)

#concatanating dummies
df1=pd.concat([df1,dummy2],axis=1)


#dropping unusual column
df1=df1.drop(columns=['Stall_no','Product_Category','instock_date','Market_Category'],axis=1)

#storing the dataframe columns
store_col=df1.columns.values

#converting to numpy array for replacing missing values based on other values
tt=np.array(df1)
m,n=df1.shape
for i in range(m):
    if(tt[i][2]<0 and (tt[i][5]>tt[i][7])):  #when minimun price is greater than selling price discount must be given else discount is 0
        tt[i][2]=1
    elif(tt[i][2]<0):
        tt[i][2]=0
     
df1=pd.DataFrame(data=tt,columns=store_col)

#dropping used columns
df=df1.drop(columns=['Minimum_price','Maximum_price'],axis=1)

#creating input_train data
X=df.drop('Selling_Price',axis=1)
y=df['Selling_Price']
X=X.drop(columns=['Demand','charges_2 (%)'],axis=1)

##training and fitting the model
#importing necessary libraries to train data
from xgboost import XGBRegressor


mod1= XGBRegressor(eta=0.058,max_depth=23,subsample=0.5,booster='dart',gamma=0.1,alpha=1)
  
mod1.fit(np.array(X),y)
with open('C:\\Users\\BHASWANTH REDDY\\Desktop\\dataset\\model.pickle','wb') as f:
    pickle.dump(mod1,f) #storing model in pickle format



#importing test data
test=pd.read_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\dataset\\test.csv')

#storing product_ids
ids=test.Product_id

#dropping unusual columns
test=test.drop(columns=['Product_id','Customer_name','Loyalty_customer'],axis=1)

#filling null values accordingly either zero or median or mean or mode or manually picked value
test.Market_Category=test.Market_Category.fillna(0) 
test.Stall_no=test.Stall_no.fillna(0)
test.Grade=test.Grade.fillna(0)
md=test.Demand.mode()[0]
test.Demand=test.Demand.fillna(md)
test.Discount_avail=test.Discount_avail.fillna(0)
test.Maximum_price=test.Maximum_price.fillna(6500)
test.charges_1=test.charges_1.fillna(-1)
test['charges_2 (%)']=test['charges_2 (%)'].fillna(0)
test.Minimum_price=test.Minimum_price.fillna(-1)
test['instock_date']=test['instock_date'].apply(lambda x: int(x.split('-')[1])) 

###making data as we made for traing input_data
#adding column
test['avg_price']=0

#storing columns in col
col=test.columns.values

#converting test_data to numpy array
tt=np.array(test)

#filling missing values of minimum_price and charges_1
m,n=test.shape
for i in range(m):
    if(tt[i][9]<0):
        if(tt[i][10]<4000):
            tt[i][9]=tt[i][10]/1.9
        elif(tt[i][10]<8000):
            tt[i][9]=tt[i][10]/1.5     #manual picked weight through observation
        elif(tt[i][10]<12000):
            tt[i][9]=tt[i][10]/2.5
        elif(tt[i][10]<20000):
            tt[i][9]=tt[i][10]/3
        elif(tt[i][10]>22000 and tt[i][10]<23000):
            tt[i][9]=tt[i][10]/1.7
        else:
            tt[i][9]=tt[i][10]/3.3
    if(tt[i][10]<0):
        tt[i][10]=tt[i][9]*1.8
    if(tt[i][7]<0):
        tt[i][7]=dic[tt[i][3]]
    tt[i][11]=(tt[i][9]+tt[i][10])/(2*price_range)
    
     
#converting again into dataframe
test=pd.DataFrame(data=tt,columns=col)

#dropping used columns
test=test.drop(columns=['Minimum_price','Maximum_price'],axis=1)


#storing columns of product_category in prod_cat demand in dem
prod_cat=test.Product_Category
dem=test.Demand

#dropping used columns and unusual columns
test=test.drop(columns=['Demand','charges_2 (%)','instock_date','Product_Category','Stall_no','Market_Category'],axis=1)

m,n=test.shape
dum_col1=dummy1.columns.values
length1=dum_col1.size
dum_col2=dummy2.columns.values
length2=dum_col2.size

#function to find the index of particular column category
def find(arr,n):
    k=np.where(arr==n)[0]
    if(k.size==0):
        return -1
    return k[0]

#predicting selling price for every testcase and storing the predicted price in y_test
y_test=[]
test=pd.DataFrame(test)

for i in range(m):
    ar1=np.zeros((length1,1)).ravel()  #creating zero array for product_category
    ar2=np.zeros((length2,1)).ravel()  #creating zero array for demand
    f1=find(dum_col1,prod_cat[i])      #finding the position assigned for particular product_category type
    f2=find(dum_col2,dem[i])           #finding the position assigned for particular demand type
 
    if(f1!=-1):
        ar1[f1]=1
    if(f2!=-1):
        ar2[f2]=1
    arr=np.concatenate([test.loc[i].values,ar1]) 
    arr=np.concatenate([arr,ar2])               #concaenated for making input data
    p=mod1.predict(arr.reshape(1,-1))[0]        #final prediction by our trained model
    y_test.append(p)
y_test=pd.DataFrame(data=y_test,columns=['Selling_Price']) #storing results in data frame

#converting previously stored product_id of testdata to dataframe
ids=pd.DataFrame(data=ids,columns=['Product_id'])          #creating ids dataframe

#forming final dataframe
res=pd.concat([ids,y_test],axis=1)                         #clubbed to write the results
print(res)

#storing results in my desktop in csv format
res.to_csv('C:\\Users\\BHASWANTH REDDY\\Desktop\\dataset\\res.csv',index=False)   #creating final csv file in my pc

