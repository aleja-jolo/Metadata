#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from dateutil.relativedelta import relativedelta


# In[20]:


from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
def mydisplay(dfs=[], names=[]):
    html_str = ''
    if names:
        html_str += ('<tr>' + ''.join(f'<td style="text-align:center">{name}</td>' for name in names) +'</tr>')
    html_str += ('<tr>' + ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td></td>'
                                    for df in dfs) +'</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table','table style="display:inline"')
    display_html(html_str, raw=True)


# In[21]:


mdf = pd.read_pickle('Module_Temps.pkl')


# In[22]:


mdf['Time']=mdf['Time'].dt.tz_localize(None)
mdf.info()


# In[23]:


mdf = mdf[mdf['Time']<datetime.datetime(2020,5,5)]


# In[24]:


mdf['core'] = mdf['Core'].apply(lambda x: int(x[-1]))
mdf.drop(columns='Core',inplace=True)
mdf.rename(columns={'core':'Core'},inplace=True)


# In[25]:


mdf.sort_values(['Core','Rack Number','Module','Time'],inplace=True)


# In[26]:


mdf.head(50)


# In[27]:


m_df = mdf[mdf['Rack Number'].isnull()==False]
print(f'Fraction of dataframe where rack number is not null = {len(m_df)/len(mdf)}')
print(f'Length of df is {len(m_df)}')


# In[28]:


m_df.sort_values(['Core','Rack Number','Module','Time'],inplace=True)
m_df['tdiff']=m_df.groupby(['Core','Rack Number','Module'])['Time'].diff()/np.timedelta64(1,'s')


# In[29]:


mdf1 = pd.read_pickle('Module_Temps_1.pkl')
mdf1['Time']=mdf1['Time'].dt.tz_localize(None)
mdf1 = mdf1[mdf1['Time']<datetime.datetime(2020,5,5)]
mdf1.sort_values(['Core','Rack Number','Module','Time'],inplace=True)
mdf1['tdiff']=mdf1.groupby(['Core','Rack Number','Module'])['Time'].diff()/np.timedelta64(1,'s')
m_df1 = mdf1[mdf1['Rack Number'].isnull()==False]
print(f'Fraction of dataframe where rack number is not null = {len(m_df1)/len(mdf1)}')
print(f'Length of df is {len(m_df1)}')


# In[30]:


print(m_df['Rack Number'].unique())
print(m_df1['Rack Number'].unique())


# In[31]:


m_df.groupby("tdiff")['Time'].count().sort_values(ascending=False).reset_index()


# In[32]:


m_df = m_df[~((m_df['Average Cell Temperature'].isnull()==True)&(m_df['Max Cell Temperature'].isnull()==True)
    &(m_df['Min Cell Temperature'].isnull()==True))]


# In[33]:


m_df = m_df.drop_duplicates(subset=['Time','Rack Number','Module','Core'],keep='last')
m_df.sort_values(['Core','Rack Number','Module','Time'],inplace=True)
m_df['tdiff']=m_df.groupby(['Core','Rack Number','Module'])['Time'].diff()/np.timedelta64(1,'s')
#m_df = m_df[m_df['tdiff']<=360


# In[34]:


cnts = m_df.groupby("tdiff")['Time'].count().sort_values(ascending=False).reset_index()
cnts[cnts['tdiff']>3600]


# In[35]:


dcurr = m_df['Time'].min()
dend = m_df['Time'].max()
dates = []
while True:
    dates.append(dcurr)
    dcurr += relativedelta(minutes=+5)
    if dcurr.year == dend.year and dcurr.month == dend.month and dcurr.day == dend.day:  # end date
        break


# In[36]:


c = 0
gdic = {}
for i in range(len(dates)):
    if i != len(dates)-1:
        start = dates[i]
        end = dates[i+1]
    else:
        start = dates[i]
        end = start + relativedelta(minutes=+5)
    m_df.loc[(m_df['Time']>=start)&(m_df['Time']<end),'group'] = c
    gdic[c]=start
    c+=1


# In[37]:


five_df = m_df.groupby(['Core','Rack Number','Module','group']).mean().reset_index()


# In[38]:


five_df['Rack Number']=five_df['Rack Number'].astype(np.int64)


# In[39]:


for c in gdic:
    five_df.loc[(five_df['group']==c),'Time']=gdic[c]


# In[42]:


gc = five_df.groupby('group').count().reset_index()
gc[gc['Core']==gc['Core'].max()]


# In[43]:


samp_df = five_df[five_df['group']==453]
samp_df


# In[44]:


samp_df3 = samp_df.query("Core==3")
samp_df3.fillna(0,inplace=True)
samp_df3 = samp_df3[['Rack Number','Module','Average Cell Temperature']].pivot('Module','Rack Number','Average Cell Temperature')


# In[45]:


samp_df4 = samp_df.query("Core==4")
samp_df4.fillna(0,inplace=True)
samp_df4=samp_df4[['Rack Number','Module','Average Cell Temperature']].pivot('Module','Rack Number','Average Cell Temperature')


# In[46]:


sns.set(style="white",font_scale=1.2)
fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(samp_df3,linewidths=.8,cmap='plasma',annot=True)
plt.title('Core 3')
plt.yticks(rotation=0) 
plt.show()

fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(samp_df4,linewidths=.8,cmap='plasma',annot=True)
plt.title('Core 4')
plt.yticks(rotation=0) 
plt.show()


# In[47]:


ylabels = samp_df3.index.to_list()
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(12,18))
sns.heatmap(samp_df3,linewidths=.8,cmap='plasma',annot=True, ax=ax1)
ax1.set_yticklabels(ylabels,rotation=0) 
ax1.set_title('Core 3')
sns.heatmap(samp_df4,linewidths=.8,cmap='plasma',annot=True, ax=ax2)
ax2.set_yticklabels(ylabels,rotation=0) 
ax2.set_title('Core 4')
plt.show()


# In[50]:


import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[51]:


import plotly as py
import plotly.graph_objs as go


fmatrix = np.array(samp_df3)
x = samp_df3.columns.to_list()
y = samp_df3.index.to_list()

trace = go.Heatmap(
   x = x,
   y = y,
   z = fmatrix,
   type = 'heatmap',
   colorscale = 'plasma'
)
data = [trace]

layout = dict( margin=dict(l=20, r=20, t=20, b=20),
     paper_bgcolor="white",
     plot_bgcolor='rgba(0,0,0,0)')

fig = go.Figure(data = data)
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="white",
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.show()


# In[52]:


five_df3 = five_df.query("Core==3")
five_df4 = five_df.query("Core==4")


# In[53]:


five_df3


# In[54]:


data_core3 = []
t = []
for g, df in five_df3.groupby('group'):
    t.append(df.iloc[0]['Time'])
    samp = df[['Rack Number','Module','Average Cell Temperature']].pivot('Module','Rack Number','Average Cell Temperature')
    trace = go.Heatmap(
        x = samp.columns.to_list(),
        y =  samp.index.to_list(),
        z = np.array(samp),
        type = 'heatmap',
        colorscale = 'plasma',
        xgap = 3, ygap = 3
        )
    data_core3.append(trace)

steps = []
for i in range(len(data_core3)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(data_core3)},
              {"title": "Slider switchced to Timestamp: " + str(t[i])}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Time: "},
    pad={"t": 1},
    steps=steps
)]


# In[55]:


fig = go.Figure(data = data_core3)
fig.update_layout(
    margin=dict(l=30, r=30, t=40, b=40),
    paper_bgcolor="white",
    plot_bgcolor='rgba(0,0,0,0)',
    sliders=sliders,
    xaxis_title="Rack No.",
    yaxis_title="Module No."
)
fig['layout']['sliders'][0]['pad']['t']=50
fig.show()

# import dash
# import dash_core_components as dcc
# import dash_html_components as html

# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
# app.run_server(debug=True, use_reloader=False)


# In[56]:


five_df.loc[:,'CMod'] = five_df.apply(lambda x: str(x['Core'])+'-'+str(x['Module']),axis=1)
five_df


# In[57]:


samp_df = five_df[five_df['group']==453]
test = samp_df[['Rack Number','CMod','Average Cell Temperature']].pivot('CMod','Rack Number','Average Cell Temperature')
nindex = []
for i in range(3,5):
    for j in range(1,18):
        nindex.append(str(i)+'-'+str(j))
test = test.reindex(nindex)
test


# In[58]:


nindex = []
for i in range(3,5):
    for j in range(1,18):
        nindex.append(str(i)+'-'+str(j))
        
data_core = []
t = []
for g, df in five_df.groupby('group'):
    t.append(df.iloc[0]['Time'])
    samp = df[['Rack Number','CMod','Average Cell Temperature']].pivot('CMod','Rack Number','Average Cell Temperature')
    samp = samp.reindex(nindex)
    trace = go.Heatmap(
        x = samp.columns.to_list(),
        y =  samp.index.to_list(),
        z = np.array(samp),
        type = 'heatmap',
        colorscale = 'plasma',
        xgap = 3, ygap = 3
        )
    data_core.append(trace)

steps = []
for i in range(len(data_core)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(data_core)},
              {"title": "Slider Switchced to Timestamp: " + str(t[i])}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Time: "},
    pad={"t": 1},
    steps=steps
)]


# In[59]:


fig = go.Figure(data = data_core)
fig.update_layout(
    margin=dict(l=50, r=50, t=100, b=100, pad=5),
    paper_bgcolor="white",
    plot_bgcolor='rgba(0,0,0,0)',
    sliders=sliders,
    xaxis_title="Rack",
    yaxis_title="Core-Module",
    autosize=False,
    width=1000,
    height=1000
)
fig['layout']['sliders'][0]['pad']['t']=50
fig.show()


# In[63]:


sdf = pd.read_pickle('Sensors.pkl')
sdf['time']=sdf['time'].dt.tz_localize(None)
sdf['sensor'] = sdf['sensor'].astype(str)
sdf = sdf[sdf['sensor'].str.startswith('Temperature')]
sdf


# In[64]:


c = 0
gdic = {}
for i in range(len(dates)):
    if i != len(dates)-1:
        start = dates[i]
        end = dates[i+1]
    else:
        start = dates[i]
        end = start + relativedelta(minutes=+5)
    sdf.loc[(sdf['time']>=start)&(sdf['time']<end),'group'] = c
    gdic[c]=start
    c+=1


# In[73]:


sdf['Core'] = sdf['core'].apply(lambda x: int(x[-1]))
sdf['Sensor'] = sdf['sensor'].apply(lambda x: int(x[-1]))
sdf


# In[77]:


fs_df = sdf.groupby(['Core','Sensor','group']).mean().reset_index()
for c in gdic:
    fs_df.loc[(fs_df['group']==c),'Time']=gdic[c]


# In[81]:


five_df


# In[176]:


fs_df.loc[fs_df['Sensor'].isin([3,6,5,8]),'CMod']='3-0'
fs_df.loc[fs_df['Sensor'].isin([2,1,4,7]),'CMod']='4-18'

fs_df.loc[fs_df['Sensor']==3,'Rack Number']=3
fs_df.loc[fs_df['Sensor']==6,'Rack Number']=8
fs_df.loc[fs_df['Sensor']==5,'Rack Number']=12
fs_df.loc[fs_df['Sensor']==8,'Rack Number']=16

fs_df.loc[fs_df['Sensor']==2,'Rack Number']=1
fs_df.loc[fs_df['Sensor']==1,'Rack Number']=8
fs_df.loc[fs_df['Sensor']==4,'Rack Number']=11
fs_df.loc[fs_df['Sensor']==7,'Rack Number']=14


# In[177]:


fs_df


# In[178]:


fdf=five_df[['Time','CMod','Rack Number','group','Average Cell Temperature']]
fsdf = fs_df[['Time','CMod','Rack Number','group','value']]
fsdf=fsdf.rename(columns={'value':'Average Cell Temperature'})
fdf = pd.concat([fdf,fsdf],axis=0)


# In[179]:


nindex = []
for j in range(0,18):
    nindex.append('3-'+str(j))
for j in range(1,19):
    nindex.append('4-'+str(j))


# In[180]:


data_temp = []
t = []
for g, df in fdf.groupby('group'):
    t.append(df.iloc[0]['Time'])
    samp = df[['Rack Number','CMod','Average Cell Temperature']].pivot('CMod','Rack Number','Average Cell Temperature')
    samp = samp.reindex(nindex)
    trace = go.Heatmap(
        x = samp.columns.to_list(),
        y =  samp.index.to_list(),
        z = np.array(samp),
        type = 'heatmap',
        colorscale = 'plasma',
        xgap = 3, ygap = 3
        )
    data_temp.append(trace)

steps = []
for i in range(len(data_temp)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(data_temp)},
              {"title": "Slider Switchced to Timestamp: " + str(t[i])}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Time: "},
    pad={"t": 1},
    steps=steps
)]


# In[181]:


fig = go.Figure(data = data_temp)
fig.update_layout(
    margin=dict(l=50, r=50, t=100, b=100, pad=5),
    paper_bgcolor="white",
    plot_bgcolor='rgba(0,0,0,0)',
    sliders=sliders,
    xaxis_title="Rack",
    yaxis_title="Core-Module",
    autosize=False,
    width=1000,
    height=1000
)
fig['layout']['sliders'][0]['pad']['t']=50
fig.show()


# In[182]:


hdf = pd.read_pickle('HVAC.pkl')
hdf.loc[hdf['HVAC']==str(1),'Rack Number']=-1
hdf.loc[hdf['HVAC']==str(2),'Rack Number']=18


# In[183]:


c = 0
gdic = {}
for i in range(len(dates)):
    if i != len(dates)-1:
        start = dates[i]
        end = dates[i+1]
    else:
        start = dates[i]
        end = start + relativedelta(minutes=+5)
    hdf.loc[(hdf['Time']>=start)&(hdf['Time']<end),'group'] = c
    gdic[c]=start
    c+=1


# In[184]:


hdf


# In[185]:


h_df


# In[186]:


gdic


# In[187]:


h_df = hdf.groupby(['Rack Number','group']).mean().reset_index()
for c in gdic:
    h_df.loc[(h_df['group']==c),'Time']=gdic[c]


# In[188]:


nindex = []
hvac = pd.DataFrame()
for i in range(3,5):
    for j in range(1,18):
        nindex.append(str(i)+'-'+str(j))
        tmp_df = h_df.copy()
        tmp_df['CMod']=str(i)+'-'+str(j)
        hvac = pd.concat([hvac,tmp_df],axis=0)


# In[189]:


hvac = hvac[['Time','CMod','Rack Number','group','Current_Temperature']]
hvac.rename(columns={'Current_Temperature':'Average Cell Temperature'},inplace=True)
hvac.head()


# In[190]:


fdf = pd.concat([fdf,hvac],axis=0)
fdf


# In[191]:


nindex = []
for j in range(0,18):
    nindex.append('3-'+str(j))
for j in range(1,19):
    nindex.append('4-'+str(j))


# In[192]:


data_temps = []
t = []
for g, df in fdf.groupby('group'):
    t.append(df.iloc[0]['Time'])
    samp = df[['Rack Number','CMod','Average Cell Temperature']].pivot('CMod','Rack Number','Average Cell Temperature')
    samp = samp.reindex(nindex)
    trace = go.Heatmap(
        x = samp.columns.to_list(),
        y =  samp.index.to_list(),
        z = np.array(samp),
        type = 'heatmap',
        colorscale = 'plasma',
        xgap = 3, ygap = 3
        )
    data_temps.append(trace)

steps = []
for i in range(len(data_temps)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(data_temps)},
              {"title": "Slider Switchced to Timestamp: " + str(t[i])}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Time: "},
    pad={"t": 1},
    steps=steps
)]


# In[193]:


yls = []
for i in range(len(nindex)):
    if i==0:
        yls.append('sensor')
    elif i==len(nindex)-1:
        yls.append('sensor')
    else:
        yls.append(nindex[i])
yls


# In[194]:


fig = go.Figure(data = data_temps)
fig.update_layout(
    margin=dict(l=50, r=50, t=100, b=100, pad=5),
    paper_bgcolor="white",
    plot_bgcolor='rgba(0,0,0,0)',
    sliders=sliders,
    xaxis_title="Rack",
    yaxis_title="Core-Module",
    autosize=False,
    width=1000,
    height=1000,
    xaxis = dict(
        tickmode = 'array',
        tickvals = [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        ticktext = ['HVAC1','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','HVAC2']
    ),
    yaxis = dict(
        tickmode = 'array',
        tickvals = nindex,
        ticktext = yls
    )
)
fig['layout']['sliders'][0]['pad']['t']=50
fig.show()


# In[257]:


pdf = fdf.copy()
pdf['Core'] = pdf['CMod'].apply(lambda x: int(x[0]))
pdf['Module'] = pdf['CMod'].apply(lambda x: int(x[-1]) if len(x)==3 else int(x[-2:]))
pdf['x']=-100
pdf['y']=-100
pdf['z']=-100
pdf.loc[:,'Rack Number']=pdf['Rack Number']+1
pdf[['Time','Core','Rack Number','Module','Average Cell Temperature']]


# In[225]:


for c in pdf.CMod.unique():
    print(len(c))


# In[204]:


x1_df = pdf.copy()
x2_df = pdf.copy()


# In[268]:


tmp_x4 = [37.5,
482.5,
878.5,
1323.5,
1398.5,
1843.5,
1918.5,
2363.5,
2438.5,
2883.5,
2958.5,
3403.5,
3648.5,
4093.5,         
7008.5,
7453.5,
7528.5,
7973.5,
8048.5,
8493.5,
8568.5,
9013.5,
9088.5,
9533.5,
9778.5,
10223.5,
10298.5,
10743.5,
10818.5,
11263.5,
11338.5,
11783.5,
11858.5,
12303.5,
12486.5,
12931.5]         
          
core4_x = []
rng = int(len(tmp_x4)/2)
c=0
for i in range(rng):
    core4_x.append([tmp_x4[c],tmp_x4[c+1]])
    c+=2


# In[277]:


tmp_x3 = [37.5,
482.5,
878.5,
1323.5,
1398.5,
1843.5,
1918.5,
2363.5,
2438.5,
2883.5,
2958.5,
3403.5,
3648.5,
4093.5,
4168.5,
4613.5,
4688.5,
5133.5,
8048.5,
8493.5,
8568.5,
9013.5,
9088.5,
9533.5,
9778.5,
10223.5,
10298.5,
10743.5,
10818.5,
11263.5,
11338.5,
11783.5,
11858.5,
12303.5,
12486.5,
12931.5,]

core3_x = []
rng = int(len(tmp_x3)/2)
c=0
for i in range(rng):
    core3_x.append([tmp_x3[c],tmp_x3[c+1]])
    c+=2


# In[280]:


hvac_x=[[-1230,-1159],[14128.5,14199]]


# In[295]:


t4_x = []
t4_xx = []
c = 18
for i in core4_x:
    if c==16 or c==11 or c==3:
        x1 = i[0]+222.5-(85/2)
        x2 = i[0]+222.5+(85/2)
        t4_x.append([x1,x2])
    elif c==12:
        t4_xx.append([4871,4956])
    c-=1
    
t3_x = []
t3_xx = []
c = 18
for i in core3_x:
    if c==13 or c==8 or c==1:
        x1 = i[0]+222.5-(85/2)
        x2 = i[0]+222.5+(85/2)
        t3_x.append([x1,x2])
    elif c==9:
        t3_xx.append([7974,8011])
    c-=1   


# In[296]:


t3_x


# In[255]:


mod_z = []
z0 = 39.3
for i in range(17):
    if i == 0:
        z1 = round(z0,2)
        z2 = round(z1+110,2)
    else:
        z1 = round(z2+(1.9*2),2)
        z2 = round(z1+110,2)
    mod_z.append([z1,z2])
    
hvac_z = [[953,1771]]
t_z = [[1440,1546]]


# In[239]:


mod_z


# In[299]:


mod_y = [[75,730],[1270,1925]]
hvac_y =[[800,1200]]
t_y = [[38,75],[1925,1962]]
t_y3 = [[100,185]]
t_y4 = [[830,867]]


# In[293]:





# In[294]:





# In[297]:





# In[ ]:





# In[ ]:





# In[229]:


pdf['Module'].unique()


# In[222]:


pdf['Rack Number'].unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




#  # Turn off reloader if inside Jupyter


# In[414]:


#importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
flights.iplot(kind='heatmap',colorscale="Blues",title="Feature Correlation Matrix")


# In[399]:


import ipywidgets as widgets
from IPython.display import display
w = widgets.IntSlider()
display(w)


# In[394]:


samp_df3.index.to_list()


# In[318]:


flights


# In[2]:


flights = flights.pivot("month", "year", "passengers")
flights


# In[32]:


import plotly.graph_objects as go
import numpy as np
X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
values = np.sin(X*Y*Z) / (X*Y*Z)

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.show()


# In[33]:


m = np.linspace(-8,8,40)

X, Y, Z = np.meshgrid(m,m,m)
values = np.sin(X*Y*Z) / (X*Y*Z)

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.show()


# In[40]:


len(values.flatten())


# In[42]:


40**3


# In[43]:


X


# In[ ]:




