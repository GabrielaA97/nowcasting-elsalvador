"""
export_excel_to_csv.py
Export the MAIN-DEFLATED tab of the nowcasting Excel to the raw CSV read by
prepare_data_v2.R (Data/SLV_nowcasting_data_mq.csv).

IMPORTANT: open and save the Excel in Excel first (or run
`soffice --headless --convert-to xlsx`) so the formula tabs are recalculated;
otherwise MAIN-DEFLATED keeps old cached values.

Usage
-----
python Data/export_excel_to_csv.py <excel.xlsx> Data/meta_data_v2.csv Data/SLV_nowcasting_data_mq.csv
"""
import sys, pandas as pd
xlsx, meta_path, out = sys.argv[1:4]
end = sys.argv[4] if len(sys.argv) > 4 else None
d = pd.read_excel(xlsx, 'MAIN-DEFLATED (Drag formulas)', header=None)
body = d.iloc[5:].copy()
body.columns = ['date'] + list(d.iloc[4, 1:])
body['date'] = pd.to_datetime(body['date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
body = body.dropna(subset=['date'])
meta = pd.read_csv(meta_path)
body = body[['date'] + list(meta.series)]
for c in meta.series:
    body[c] = pd.to_numeric(body[c], errors='coerce')
if end: body = body[body.date <= end]
assert body.date.is_unique and body.date.is_monotonic_increasing
body.to_csv(out, index=False, date_format='%Y-%m-%d')
print(out, body.shape, body.date.min().date(), body.date.max().date())
