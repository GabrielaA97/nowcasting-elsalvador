"""Build the raw model panel directly from MAIN-RAW:
real series as published; nominal series divided by their deflator (IPC or IPP
ratio, column C/E of the IPC tab), reproducing MAIN-DEFLATED exactly where it is
current. Nominal series stop where the deflators stop."""
import sys, numpy as np, pandas as pd
xlsx, meta_path, map_csv, out = sys.argv[1:5]
d=pd.read_excel(xlsx,'MAIN-RAW (Fill new quarter)',header=None);b=d.iloc[5:].copy()
idx=pd.to_datetime(b[0],errors='coerce');b=b[idx.notna()]
b.index=pd.DatetimeIndex(idx[idx.notna()]).to_period('M').to_timestamp();b.columns=d.iloc[4].values
r=b.iloc[:,1:].apply(pd.to_numeric,errors='coerce')
ipc=pd.read_excel(xlsx,'IPC (Fill new quarter)',header=None);ipc=ipc[pd.to_datetime(ipc[0],errors='coerce').notna()]
ipc.index=pd.to_datetime(ipc[0]).dt.to_period('M').dt.to_timestamp()
P=ipc[[2,4]].apply(pd.to_numeric,errors='coerce').rename(columns={2:'IPC',4:'IPP'}).replace(0,np.nan).reindex(r.index)
meta=pd.read_csv(meta_path); m=pd.read_csv(map_csv).set_index('series')['best']
out_df=pd.DataFrame(index=r.index)
for c in meta.series:
    out_df[c]= r[c] if m[c]=='none' else r[c]/P[m[c]]
out_df=out_df.loc[:out_df.dropna(how='all').index.max()]
out_df.index.name='date'; out_df.reset_index().to_csv(out,index=False,date_format='%Y-%m-%d')
print('nominal:',int((m!='none').sum()),'real:',int((m=='none').sum()),'panel end',out_df.index.max().date())
