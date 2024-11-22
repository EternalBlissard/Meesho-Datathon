import pandas as pd

categories=['Men Tshirts','Sarees','Kurtis','Women Tshirts','Women Tops & Tunics']
df=None
for i in categories:
    d2f=pd.read_csv("preds/"+i+'.csv')
    # if df==None:
    #     df=d2f
    # else:
    df=pd.concat([df,d2f])
df.fillna("default2",inplace=True)
print(df.isna().sum())
df.to_csv("submission.csv", index=False)
