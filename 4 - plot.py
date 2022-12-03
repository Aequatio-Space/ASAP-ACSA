import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
filepath = "/Users/Charlie/Library/Mobile Documents/com~apple~CloudDocs/Knowledge_Engineering/train.csv2022-06-04 01:37:39 acc_evo.csv"
df1 = pd.read_csv(filepath)
df1.set_index('epoch').plot()
plt.title('Accuracy Evolution')
plt.show()