# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy as sp
sns.set_style('whitegrid')

# %%
from sklearn.linear_model import LinearRegression

# %%
regr=LinearRegression()

# %%
df=pd.read_excel(r"C:\Users\hamme\OneDrive\Documents\HPLC analysis\LA release template for python script.xlsx")

# %%
df=df[(df['Sample Name']!='blank') & (df['Sample Name']!='shutdown')]
df=df.sort_values('Sample Name')
df=df.drop('Unnamed: 0',axis=1)

# %%
std_curve=df['Area'][-5:]
lac_curve=[1,0.333,0.111,0.037,0.012]

# %%
x=np.array(lac_curve).reshape(-1,1)
y=std_curve.values.reshape(-1,1)

# %%
regr.fit(x,y)
regr_int=round(float(regr.intercept_),2)
regr_co=round(float(regr.coef_),2)

# %%
# g=sns.lmplot(data=std_curve,x='[LA] (mg/mL)',y='AUC').set(title='Lactic Acid Standard Curve')
# props = dict(boxstyle='round', alpha=0.5,color=sns.color_palette()[0])
# textstr = 'y='+str(regr_co)+'x + '+str(regr_int)
# g.ax.text(0.7, 0.2, textstr, transform=g.ax.transAxes, fontsize=14, bbox=props)

# def annotate(data, **kws):
#    r, p = sp.stats.pearsonr(std_curve['[LA] (mg/mL)'],std_curve['AUC'])
#    ax = plt.gca()
#    ax.text(.8, .3, 'r={:.2f}, p={:.2g}'.format(r, p),
#            transform=ax.transAxes)
    
# g.map_dataframe(annotate)
# plt.show()

# g.savefig('Standard Curve.jpg')

# %%
#Calculating [LA] for the unknown samples
df=df[:-5]
df['[LA] (mg/mL)']=10*(df['Area']+regr_int)/regr_co

# %%
df['Sample Name']=df['Sample Name'].str.split('-')

# %%
df.insert(0,'Day',df['Sample Name'].str[0])
df.insert(1,'Sample',df['Sample Name'].str[1])
df=df.drop('Sample Name',axis=1)

# %%
df

# %%
plt.figure(figsize=(8,6))
h=sns.lineplot(data=df,x='Day',y='[LA] (mg/mL)',hue='Sample',style='Sample').set(title='Lactic Acid Release from IVR Samples')
# plt.savefig('Lactic Acid Release from IVR Samples.jpg')


# %%
excel_out_path="C:\\Users\hamme\\OneDrive\\Documents\\HPLC analysis\\script-output.xlsx"
graph_output_path="C:\\Users\hamme\\OneDrive\\Documents\\HPLC analysis"
df.to_excel(excel_out_path)

# %%



