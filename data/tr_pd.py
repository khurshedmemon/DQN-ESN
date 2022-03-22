import sys
import pandas as pd
infile = sys.argv[1]
df=pd.read_csv(infile, delimiter="\t", header=None)
print(df.head(10))
df.iloc[:, -1:] = df.iloc[:, -1:].apply(pd.to_numeric, downcast='integer')
print(df.head(10))
outfile = sys.argv[2]
df.to_csv(outfile, sep="\t", index=False, header=False)