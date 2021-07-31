import pandas as pd

# calculate num_class
with open('../data/clean_header') as fin:
    for line in fin:
        header = line.strip().split('\t')
# print(header)
df = pd.read_csv('../data/clean_input_parts_0', delimiter='\t', names=header)

print(set(df['m:Rating']))
print(len(set(df['m:Rating'])))

