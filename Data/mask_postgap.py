"""Set to NaN the transformed value of the first observation after an internal gap
(the R script computes growth over the gap as if it were one period)."""
import sys, pandas as pd
raw_csv, tf_csv, meta_path = sys.argv[1:4]
raw=pd.read_csv(raw_csv,parse_dates=['date']).set_index('date')
tf=pd.read_csv(tf_csv,parse_dates=['date']).set_index('date')
meta=pd.read_csv(meta_path).set_index('series')
n=0
for c in tf.columns:
    if c not in raw or str(meta.loc[c,'freq']).lower()=='q': continue
    s=raw[c]; first=s.first_valid_index()
    if first is None: continue
    s=s.loc[first:]
    post=s.notna() & s.shift(1).isna()
    for dt in s.index[post]:
        if dt in tf.index and pd.notna(tf.loc[dt,c]): tf.loc[dt,c]=float('nan'); n+=1
tf.reset_index().to_csv(tf_csv,index=False,date_format='%Y-%m-%d'); print('masked post-gap values:',n)
