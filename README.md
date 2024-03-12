# LAB-ACTIVITI-2.1
import pandas as pd
import numpy as np

df = pd.read_csv('data/titanic.csv')

print(df.head())
print(df.describe())

print(df.dtypes)
print(df.info())

percentage_survived = len(df[df.Survived == 1]) / len(df) * 100.0
print(f"Percentage of passengers who survived: {percentage_survived:.2f}%")

df_grouped = df.groupby(by='Pclass')
print(df_grouped.Survived.count())
print(df_grouped.Survived.sum())
print(df_grouped.Survived.sum() / df_grouped.Survived.count())

df['age_range'] = pd.cut(df.Age, [0, 16, 65, 1e6], 3, labels=['child', 'adult', 'senior'])
print(df.age_range.describe())

df_grouped = df.groupby(by=['Pclass', 'age_range'])
print("Percentage of survivors in each group: ")
print(df_grouped.Survived.sum() / df_grouped.Survived.count() * 100)

cols_to_drop = ['PassengerId', 'Name', 'Cabin', 'Ticket']
df = df.drop(columns=cols_to_drop)

df_grouped = df.groupby(by=['Pclass', 'SibSp'])
print(df_grouped.describe())

df_imputed = df_grouped.transform(lambda grp: grp.fillna(value=grp.median()))
df_imputed[['Pclass', 'SibSp', 'Sex', 'Embarked']] = df[['Pclass', 'SibSp', 'Sex', 'Embarked']]
print(df_imputed.info())
