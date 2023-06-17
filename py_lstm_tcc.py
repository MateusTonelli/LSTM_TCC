import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Função utilizada para transformar dados de serie temporal em dados de aprendizado supervisionado
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]        
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

# Carregando arquivo de texto e transformando o mesmo em uma tabela
df = pd.read_csv('household_power_consumption.txt', sep=';', dayfirst=True,
                 parse_dates={'dt' : ['Date', 'Time']}, low_memory=False, na_values=['nan','?'], index_col='dt')

df.shape

print(df.isnull().sum())

# Substituindo os dados nan pela media de cada propriedade (coluna)
df = df.fillna(df.mean())

print(df.isnull().sum())

# Desenhando gráfico da distribuição das informações
i = 1
cols=[0, 1, 3, 4, 5, 6]
plt.figure(figsize=(20, 25))
for col in cols:
    plt.subplot(len(cols), 1, i)
    plt.plot(df.values[:, col])
    plt.title(df.columns[col] + ' Representação da fonte de dados', y=0.75, loc='left')
    i += 1
plt.show()

f= plt.figure(figsize=(21,3))

ax=f.add_subplot(133)
sns.set_theme(style="white")
cmap = sns.color_palette("Blues")
sns.heatmap(df.corr(), cmap=cmap, vmin=-1, vmax=1, annot=True)
plt.title('Correlação dos atributos', size=12)
plt.show()

df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_1','Sub_metering_3']]

df = df.drop(['Voltage'],axis=1)  

values = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
r = list(range(df.shape[1]+1, 2*df.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)
reframed.head()

# Divizão das informações sendo 5000 dados para treinar a base e restante para teste.
values = reframed.values
n_train_time = 5000
train = values[:n_train_time, :]
test = values[n_train_time:, :]
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# Configurando a LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Rodando a LSTM
history = model.fit(train_x, train_y, epochs=50, batch_size=70, validation_data=(test_x, test_y), verbose=2, shuffle=False)

# Gráfico da perda
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('perda')
plt.ylabel('perda')
plt.xlabel('época')
plt.legend(['treino', 'teste'], loc='upper right')
plt.show()

size = df.shape[1]

# Teste
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], size))

# Removendo transformações de escala dos dados para apresentação do resultado (Pervisão)
inv_yhat = np.concatenate((yhat, test_x[:, 1-size:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# Removendo transformações de escala dos dados para apresentação do resultado (Valor real)
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x[:, 1-size:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Calculo do erro quadrático médio
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Teste RMSE: %.3f' % rmse)

# Apresentando os dados reais e de pervisão em uma tabela para analise
aa=[x for x in range(5000)]
plt.figure(figsize=(25,10)) 
plt.plot(aa, inv_y[:5000], marker='.', label="Atual")
plt.plot(aa, inv_yhat[:5000], 'r', label="Previsão")
plt.ylabel(df.columns[0], size=15)
plt.xlabel('Analise da predição dos 5000 primeiros minutos', size=15)
plt.legend(fontsize=15)
plt.show()