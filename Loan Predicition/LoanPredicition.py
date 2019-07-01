import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df.head() # cabeçalho
df['Gender'].unique() # valores únicos
df['Loan_Status'].value_counts() # quantidade de valores
sns.countplot(x='Loan_Status', data=df, palette='hls') # define gráfico com coluna x, dados e paleta
plt.show() # mostra o gráfico
plt.savefig('count_plot') # salva o gráfico. obs.: salvou mas não mostrou direito
# verifica balanceamento entre aprovados e não aprovados
count_aprov = len(df[df['Loan_Status']=='Y'])
count_not_aprov = len(df[df['Loan_Status']=='N'])
pct_of_aprov = count_aprov/(count_aprov+count_not_aprov)
print("percentage of aproved is", pct_of_aprov*100)
pct_of_not_aprov = count_not_aprov/(count_aprov+count_not_aprov)
print("percentage of not aproved", pct_of_not_aprov*100)
# estatísticas
df.groupby(['Loan_Status']).mean()  # média dos atributos numéricos
df.groupby(['Loan_Status','Gender']).count() # quantidade dos atributos categóricos. 
df.groupby(['Loan_Status','Married']).count()
# gráfico de barras
%matplotlib inline
pd.crosstab(df.Gender,df.Loan_Status).plot(kind='bar')
plt.title('Empréstimo por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Estado do Empréstimo')
plt.savefig('EmprestimoPorSexo')
# gráfico de barras empilhado - fazer para cada atributo categórico
table=pd.crosstab(df.Gender,df.Loan_Status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Sexo x Emprestimo')
plt.xlabel('Sexo')
plt.ylabel('Proporção de Empréstimo')
plt.savefig('Prop_sexo_emprestimo')
#
table=pd.crosstab(df.Dependents,df.Loan_Status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Dependentes x Emprestimo')
plt.xlabel('Dependentes')
plt.ylabel('Proporção de Empréstimo')
plt.savefig('Prop_dependentes_emprestimo')
#
table=pd.crosstab(df.Education,df.Loan_Status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Educação x Emprestimo')
plt.xlabel('Educação')
plt.ylabel('Proporção de Empréstimo')
plt.savefig('Prop_educacao_emprestimo')
#
table=pd.crosstab(df.Self_Employed,df.Loan_Status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Autônomo x Emprestimo')
plt.xlabel('Autônomo')
plt.ylabel('Proporção de Empréstimo')
plt.savefig('Prop_autonomo_emprestimo')
#
table=pd.crosstab(df.Property_Area,df.Loan_Status)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Propriedade x Emprestimo')
plt.xlabel('Propriedade')
plt.ylabel('Proporção de Empréstimo')
plt.savefig('Prop_propriedade_emprestimo')
# categorical missing values
df['Gender'] = df['Gender'].fillna('Unknown')
df['Married'] = df['Married'].fillna('Unknown')
df['Dependents'] = df['Dependents'].fillna('Unknown')
df['Education'] = df['Education'].fillna('Unknown')
df['Self_Employed'] = df['Self_Employed'].fillna('Unknown')
df['Property_Area'] = df['Property_Area'].fillna('Unknown')
df['Loan_Status'] = df.Loan_Status.replace('N', 0)
df['Loan_Status'] = df.Loan_Status.replace('Y', 1)
# continuous missing values
df['Credit_History'].apply(str)
df['Credit_History'] = df['Credit_History'].fillna('Unknown')
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
# dummy variables
cat_vars=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
for var in cat_vars:
    cat_list = pd.get_dummies(df[var], prefix=var)
    df1=df.join(cat_list)
    df=df1
# remove columns
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
df_final=df[to_keep]
del df_final['Loan_ID']
df_final.columns.values
# df_final.head()
# sampling with SMOTE
X = df_final.loc[:, df_final.columns != 'Loan_Status']
y = df_final.loc[:, df_final.columns == 'Loan_Status']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0) # não precisa : já está dividido no exercício
columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train.values.ravel())
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Loan_Status'])
# numbers of sampling
print("length of oversampled data is ",len(os_data_X))
print("Number of no Loan in oversampled data",len(os_data_y[os_data_y['Loan_Status']==0]))
print("Number of Loan",len(os_data_y[os_data_y['Loan_Status']==1]))
print("Proportion of no Loan data in oversampled data is ",len(os_data_y[os_data_y['Loan_Status']==0])/len(os_data_X))
print("Proportion of Loan data in oversampled data is ",len(os_data_y[os_data_y['Loan_Status']==1])/len(os_data_X))
# RFE Recursive Feature Elimination
df_final_vars=df_final.columns.values.tolist()
y=['Loan_Status']
X=[i for i in df_final_vars if i not in y]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#
x_col = pd.DataFrame({'Col_name': X_train.columns})
sel_col_index = x_col.index[rfe.support_].tolist()
print(sel_col_index)
sel_col = x_col['Col_name'].iloc[sel_col_index]
print(sel_col)
#X=os_data_X[sel_col]
X=pd.DataFrame(data=os_data_X,columns=sel_col)
y=os_data_y['Loan_Status']