import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Import dataset
df = pd.read_csv('titanic.csv')
print(df.head())

#Overview of dataset structure
print("Dataset Dimensions (rows, columns):", df.shape)
print("\nDataset Info:\n") #\n is used to format the output in a readable manner
print(df.info())
###
#No missing values
###

#Descriptive statistics for numerical features
print("\nDescriptive Statistics:\n")
print(df.describe())
###
#Only looking at these numbers there are some observations that can be made:
#Survivor average = 0.38: about 38% of the passengers survived
#Pclass average = 2.31: slightly skewed towards 3rd class (50% = 3: at least half of the passengers were in 3rd class)
#SibSp average = 0.45: most people had no family on board
#Average ticket price = 32.09, but max = 512: highly skewed distribution
#Average boarding = 2.53, std = 0.79: most passengers boarded from Southampton (3).
###

#Frequency distribution for categorical features
print("\nSex Distribution:\n", df['Sex'].value_counts())
print("\nPclass Distribution:\n", df['Pclass'].value_counts())
print("\nEmbarked Distribution:\n", df['Embarked'].value_counts(dropna=False))

#Visualization: Survival percentage
#% calculation
survival_counts = df['Survived'].value_counts()
survival_percent = survival_counts / survival_counts.sum() * 100
#Define custom labels
labels = ['No', 'Yes'] if 0 in survival_counts.index else ['Yes', 'No']
#Define colors
colors = ['#F07F26', '#5DA5DA']
#Define font
plt.rcParams['font.family'] = 'DejaVu Serif'
#Pie chart
plt.figure(figsize=(6, 6))
plt.pie(survival_percent,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 13, 'color': 'black'})
plt.title('Survival Rate on the Titanic', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.show()

#Filtre only survivors
survivors = df[df['Survived'] == 1]
#Descriptive statistics for the main variables
age_desc = survivors['Age'].describe()
sex_counts = survivors['Sex'].value_counts()
embarked_counts = survivors['Embarked'].value_counts()
pclass_counts = survivors['Pclass'].value_counts()
sibsp_counts = survivors['SibSp'].value_counts()
parch_counts = survivors['Parch'].value_counts()
#%
sex_pct = survivors['Sex'].value_counts(normalize=True) * 100
print("Sex percentage among survivors:")
print(sex_pct)
print()
embarked_pct = survivors['Embarked'].value_counts(normalize=True) * 100
print("Embarkation port percentage among survivors:")
print(embarked_pct)
print()
pclass_pct = survivors['Pclass'].value_counts(normalize=True) * 100
print("Passenger class percentage among survivors:")
print(pclass_pct)
print()
sibsp_pct = survivors['SibSp'].value_counts(normalize=True) * 100
print("Siblings/spouses aboard percentage among survivors:")
print(sibsp_pct)
print()
parch_pct = survivors['Parch'].value_counts(normalize=True) * 100
print("Parents/children aboard percentage among survivors:")
print(parch_pct)


###VISUALIZATION###
#Survival by Sex
#Map Survived to custom labels
df['Survived'] = df['Survived'].map({0: 'No', 1: 'Yes'})
#Create crosstab (absolute values)
ct = pd.crosstab(df['Sex'], df['Survived'])
#Convert to row-wise percentages
ct_percent = ct.div(ct.sum(axis=1), axis=0) * 100
#Plot
ax = ct_percent.plot(kind='bar',
                     stacked=True,
                     color=colors,
                     edgecolor='black',
                     figsize=(8, 6))
#Add labels on each bar segment
for i, row in enumerate(ct_percent.values):
    bottom = 0
    for j, val in enumerate(row):
        ax.text(i,
                bottom + val / 2,
                f'{val:.1f}%',
                ha='center',
                va='center',
                color='white',
                fontsize=11,
                fontweight='bold')
        bottom += val
# Aesthetics
plt.title('Survival Rate by Sex (%)', fontsize=16, fontweight='bold')
plt.xlabel('Sex', fontsize=13)
plt.ylabel('Percentage (%)', fontsize=13)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Survived', title_fontsize=12, fontsize=11)
plt.tight_layout()
plt.show()
#Visualization: Age distribution by Survival
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Survived', y='Age')
#Add titles and labels
plt.title('Age Distribution by Survival', fontsize=16, fontweight='bold')
plt.xlabel('Survived', fontsize=13)
plt.ylabel('Age', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()