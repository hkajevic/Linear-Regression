# Pandas modul za manipulaciju nad podacime
import pandas as pd

# Pyplot i seaborn za vizuelizaciju podataka
import matplotlib.pyplot as plt
import seaborn as sb

# Numpy za rad sa visedimenzionalnim nizovima
import numpy as np

# Ograniacvanje sirine
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)

# Uvoz ugradjenog algoritma za linearnu regresiju
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Citanje podataka
data = pd.read_csv('car_purchase.csv')

# Ispis prvih i poslednjih 5 redova
print("PRVIH 5 REDOVA")
print(data.head())
print("POSLEDNJIH 5 REDOVA")
print(data.tail())

# Bazicne informacije o dataframu
print("OSNOVNE INFORMACIJE")
print(data.info())

# Statistika numerickih atributa
print("STATISTIKA NUMERICKIH ATRIBUTA")
print(data.describe())

# Statistika kategorickih atributa
print("STATISTIKA KATEGORICKIH ATRIBUTA")
print(data.describe(include=[object]))

# Zavisnost maksimalnog iznosa od godina klijenta
X = data.loc[:, ['age']]
y = data['max_purchase_amount']
plt.figure('Maximalna zarada - Godine')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
edgecolors='black', linewidths=2)

plt.ylabel('Maxiamalna zarada', fontsize=13)
plt.xlabel('Godiine klijenta', fontsize=13)
plt.title('Maximalna zarada - Godine')


# Zavisnost maksimalnog iznosa od godisnje zarade klijenta
X = data.loc[:, ['annual_salary']]
y = data['max_purchase_amount']
plt.figure('Maximalna zarada - Godisnja zarada')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
edgecolors='black', linewidths=2)
plt.ylabel('Maximalna zarada ', fontsize=13)
plt.xlabel('Godisnja zarada', fontsize=13)
plt.title('Maximalna zarada - Godisnja zarada')



# Zavisnost maksimalnog iznosa od godisnje duga na kreditnoj kartici
X = data.loc[:, ['credit_card_debt']]
y = data['max_purchase_amount']
plt.figure('Maximalna zarada - Dug')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
edgecolors='black', linewidths=2)
plt.ylabel('Maximalna zarada', fontsize=13)
plt.xlabel('Dug na kreditnoj kartici', fontsize=13)
plt.title('Maximalna zarada - Dug')


# Zavisnost maksimalnog iznosa od ukupnog imetka musterije
X = data.loc[:, ['net_worth']]
y = data['max_purchase_amount']
plt.figure('Maximalna zarada - Ukupni imetak')
plt.scatter(X, y, s=23, c='red', marker='o', alpha=0.7,
edgecolors='black', linewidths=2)
plt.ylabel('Maximalna zarada', fontsize=13)
plt.xlabel('Ukupni imetak', fontsize=13)
plt.title('Maximalna zarada - Ukupni imetak')
plt.tight_layout()
plt.show()

data = data.drop(columns=['customer_id'])
data['gender'].replace('F', 0, inplace=True)
data['gender'].replace('M', 1, inplace=True)


Y = data['max_purchase_amount']
X = data.drop(columns=['max_purchase_amount'])
sc=StandardScaler()
X_transform = sc.fit_transform(X)

def predicted_y(weight, x, intercept):
    y_lst = []
    for i in range(len(x)):
        y_lst.append(weight @ x[i] + intercept)
    return np.array(y_lst)


# linearni gubitak
def loss(y, y_predicted):
    n = len(y)
    s = 0
    for i in range(n):
        s += (y[i] - y_predicted[i]) ** 2
    return (1 / n) * s


def dldw(x, y, y_predicted):
    s = 0
    n = len(y)
    for i in range(n):
        s += -x[i] * (y[i] - y_predicted[i])
    return (2 / n) * s


# izvod
def dldb(y, y_predicted):
    n = len(y)
    s = 0
    for i in range(len(y)):
        s += -(y[i] - y_predicted[i])
    return (2 / n) * s


# gradijentna funkcija
def gradient_descent(x, y):
    weight_vector = np.random.randn(x.shape[1])
    intercept = 0
    epoch = 2000
    n = len(x)
    linear_loss = []
    learning_rate = 0.001

    for i in range(epoch):
        y_predicted = predicted_y(weight_vector, x, intercept)

        weight_vector = weight_vector - learning_rate * dldw(x, y, y_predicted)

        intercept = intercept - learning_rate * dldb(y, y_predicted)
        linear_loss.append(loss(y, y_predicted))

    plt.plot(np.arange(1, epoch), linear_loss[1:])
    plt.xlabel("number of epoch")
    plt.ylabel("loss")

    return weight_vector, intercept

w,b=gradient_descent(X_transform,Y)


def predict(inp):
    y_lst=[]
    for i in range(len(inp)):
        y_lst.append(w@inp[i]+b)
    return np.array(y_lst)

y_pred=predict(X_transform)

plt.figure()
plt.title('My Linear Regression')
plt.scatter(Y, y_pred)
plt.xlabel('y_actual')
plt.ylabel('y_pred')


df_pred=pd.DataFrame()
df_pred["y_actual"]=Y
df_pred["y_predicted"]=np.round(y_pred,1)

reg = LinearRegression()
reg.fit(X_transform, y)

plt.figure()
plt.title('Ugradjeni model')
plt.scatter(Y.values, reg.predict(X_transform), color="r")
plt.xlabel('y_actual')
plt.ylabel('y_pred')
plt.figure(1)


print("KOEFICIJENTI")
print("weight:", w)
print("bias:", b)
print("=========")
print("weight:", reg.coef_)
print("bias:", reg.intercept_)

print("\nGRESKA")
print(mean_squared_error(Y, y_pred))
print("=========")
print( mean_squared_error(Y, reg.predict(X_transform)))

old_c = reg.coef_
old_i = reg.intercept_
print("\nSCORE")
reg.coef_ = w
reg.intercept_ = b
print(reg.score(X_transform, Y))
print("=========")
reg.coef_ = old_c
reg.intercept_ = old_i
print(reg.score(X_transform, Y))

plt.show()
