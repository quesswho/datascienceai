import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read from file
df = pd.read_csv('listings.csv')

df2 = (df[df['date'].str.contains('2022')])

print(df2['closing price'].describe().loc[['min','25%','50%','75%','max']])

fig, axs = plt.subplots(3, constrained_layout = True , figsize = (7 ,8))

rooms = np.nan_to_num(np.array(pd.to_numeric(df2['rooms'])))
axs [0].hist (df2['closing price'], bins = 25)
sp = axs[2].scatter(df2['boarea'], df2['closing price'], c = rooms , cmap = "viridis")

axs[0].set_xlabel("closing price")
axs[0].set_ylabel("N")
axs[1].scatter(df2['boarea'], df2['closing price'])
axs[1].set_xlabel("boarea")
axs[1].set_ylabel("Closing price")
axs[2].set_xlabel("boarea")
axs[2].set_ylabel("Closing price")

fig.colorbar(sp)
plt.show()
