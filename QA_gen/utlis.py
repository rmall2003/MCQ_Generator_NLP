import pandas as pd

def export_csv(rows, out_path):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
