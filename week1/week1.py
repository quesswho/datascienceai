import pandas as pd
import matplotlib.pyplot as plt

# Read from file
df = pd.read_csv('swedish_population_by_year_and_sex_1860-2022.csv')


# Cast age column to integer
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Extract age groups
children = df[df['age'] <= 14]
workforce = df[(df['age'] > 14) & (df['age'] <= 64)]
elderly = df[df['age'] > 64]

# Compute the group populations
total_children  = children[children.columns[2:]].sum()
total_workforce  = workforce[workforce.columns[2:]].sum()
total_elderly  = elderly[elderly.columns[2:]].sum()

dependency_ratio = 100*(total_children+total_elderly)/(total_workforce)

plt.plot(dependency_ratio.index, dependency_ratio.values, label='Dependency Ratio', color='b')
plt.xlabel('Year')
plt.ylabel('Dependency Ratio (%)')
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

plt.show()

# Compute the total population across all ages
total_population = total_children + total_workforce + total_elderly

# Compute the ratio of each group
ratio_children  = total_children/total_population
ratio_elderly  = total_elderly/total_population
ratio_dependent  = (total_elderly+total_children)/total_population

plt.plot(ratio_children.index, ratio_children.values, label='Fraction of Children', color='g')
plt.plot(ratio_elderly.index, ratio_elderly.values, label='Fraction of Elderly', color='r')
plt.plot(ratio_dependent.index, ratio_dependent.values, label='Fraction of Dependent Population', color='b')

plt.xlabel('Year')
plt.ylabel('Fraction of Total Population')
plt.ylim(0,0.5)
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

plt.show()