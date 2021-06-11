# Projeto 1 - Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile
# Dataset: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

### 1 - Definindo o problema de negócio:
# Prever se o usuário fará o download de um aplicativo ou não após clicar 
# em um anúncio, de modo a prever a ocorrência de fraudes, com base 
# em cliques que não fizeram o download.



### 2 - Coleta e visualização dos dados:
# Configurando o diretório de trabalho
setwd("C:/Projetos_DataScience/Projeto1-Deteccao-Fraudes-Cliques-Propaganda-AppMobile")
# Verificar o diretório:
getwd()

# Carregando os pacotes:
library(data.table)
library(ggplot2)
library(corrplot)
library(dplyr)
library(lubridate)
library(gridExtra)
library(caTools)
library(caret)
library(ROSE)
library(rpart)
library(randomForest)
library(C50)
library(e1071)
library(knitr)
library(rmarkdown)

# Carregando os dados:
dados <- fread("train_sample.csv", stringsAsFactors = F, sep = ",", header =T)

# Verificando se há valores missing:
sum(is.na(dados))
# Não há valores nulos

# Visualizando os dados:
head(dados)
View(dados)
str(dados)

# Dicionário dos dados:
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded

#Note that ip, app, device, os, and channel are encoded.



### 3 - Pré processamento dos dados:
# Modificando as variaveis de tempo para o formato correto:
dados$click_time = ymd_hms(dados$click_time) 
dados$attributed_time = ymd_hms(dados$attributed_time)

# Modificando a variável is_attributed para fator, necessário para
# criar o modelo de classificação
dados$is_attributed <- as.factor(dados$is_attributed)

# Visualizando o formato das variáveis:
str(dados)

# Verificando se há valores missing após a transformação dos dados:
sum(is.na(dados))

# Retirando a coluna attributed_time do dataset por ter muitos valores NA:
dados$attributed_time <- NULL


### 4 - Análise exploratória:
## Verificando a quantidade de valores unicos:
sapply(dados, function(x) length(unique(x))) %>% 
  sort(decreasing = TRUE)

## Verificando a quantidade de cliques do mesmo IP:
ip <- dados$ip %>%
  table() %>%                   # Verifica a quantidade de valores unicos
  sort(decreasing = TRUE) %>%   # Ordena por ordem decrescente
  head(n = 10L) %>%             # Exibe os 10 IPs que mais fizeram cliques
  as.data.frame()               # Converte para formato data.frame
colnames(ip) <- c('ip_x', 'ip_y')

# Visualizando graficamente
graph1 <- ggplot(ip, aes(x = ip_x, y = ip_y)) +
  geom_bar(stat="identity", fill = 'steelblue4') +
  ggtitle("IPs que mais clicaram em anuncios")+
  xlab("Endereço IP") + 
  ylab("Numero de cliques")
graph1


## Verificando os apps mais acessados:
app <- dados$app %>%
  table() %>%                   # Verifica a quantidade de valores unicos
  sort(decreasing = TRUE) %>%   # Ordena por ordem decrescente
  head(n = 10L) %>%             # Exibe os 10 apps mais acessados
  as.data.frame()               # Converte para formato data.frame
colnames(app) <- c('app_x', 'app_y')

# Visualizando graficamente
graph2 <- ggplot(app, aes(x = app_x, y = app_y)) +
  geom_bar(stat="identity", fill = 'steelblue4') +
  ggtitle("Apps mais acessados")+
  xlab("App") + 
  ylab("Quantidade")
graph2


## Verificando os dispositivos mais usados:
device <- dados$device %>%
  table() %>%                   # Verifica a quantidade de valores unicos
  sort(decreasing = TRUE) %>%   # Ordena por ordem decrescente
  head(n = 5L) %>%              # Exibe os 5 dispositivos mais usados
  as.data.frame()               # Converte para formato data.frame
colnames(device) <- c('device_x', 'device_y')

# Visualizando graficamente
graph3 <- ggplot(device, aes(x = device_x, y = device_y)) +
  geom_bar(stat="identity", fill = 'steelblue4') +
  ggtitle("Dispositivo mais usado")+
  xlab("Dispositivo") + 
  ylab("Quantidade")
graph3


## Verificando o sistema operacional mais usados:
os <- dados$os %>%
  table() %>%                   # Verifica a quantidade de valores unicos
  sort(decreasing = TRUE) %>%   # Ordena por ordem decrescente
  head(n = 10L) %>%             # Exibe os 10 SO mais usados
  as.data.frame()               # Converte para formato data.frame
colnames(os) <- c('os_x', 'os_y')

# Visualizando graficamente
graph4 <- ggplot(os, aes(x = os_x, y = os_y)) +
  geom_bar(stat="identity", fill = 'steelblue4') +
  ggtitle("S.O. mais usado")+
  xlab("S.O") + 
  ylab("Quantidade")
graph4


## Verificando o canal mais usados:
channel <- dados$channel %>%
  table() %>%                   # Verifica a quantidade de valores unicos
  sort(decreasing = TRUE) %>%   # Ordena por ordem decrescente
  head(n = 10L) %>%             # Exibe os 10 canais mais utilizados
  as.data.frame()               # Converte para formato data.frame
colnames(channel) <- c('ch_x', 'ch_y')

# Visualizando graficamente
graph5 <- ggplot(channel, aes(x = ch_x, y = ch_y)) +
  geom_bar(stat="identity", fill = 'steelblue4') +
  ggtitle("Canais mais utilizados")+
  xlab("Canais") + 
  ylab("Quantidade")
graph5

# Plotando os gráficos simultaneos:
grid.arrange(graph1, graph2, graph3, graph4, graph5)


## Verificando o balanceamento dos dados:
prop.table(table(dados$is_attributed))*100
# Mais de 99% dos dados é referente a IPs que não fizeram o download do app.
# Ou seja, a classe está completamente desbalanceada.

# Verificando o resultado graficamente:
barplot(table(dados$is_attributed),
        main = "Balanceamento dos dados",
        xlab = '0 = Não fez dowload    1 = Fez download',
        ylab = 'Quantidade',
        ylim = c (0,100e3))


## Verificando a correlação entre os dados:
# Transformando as variáveis para numeric:
dados_cor <- dados %>%
  mutate_if(is.factor, as.integer)

str(dados_cor)

cor(dados_cor[, -c('click_time')]) %>%
  cor(method = 'spearman') %>%
  corrplot(method = 'number', diag = FALSE, type = 'lower')
# O IP apresenta uma correlação positiva de 66% com a variável target.

## Verificando o padrão cliques e download por horario:
dados_time <- dados[, c('click_time','is_attributed')]
str(dados_time)
head(dados_time)

#Simbolos especiais para data.table:
?.N

# Verificando o numero de cliques para cada hora:
hora <- hour(dados_time$click_time)

numero_cliques <- dados_time[, .N, by = hora]
numero_cliques

# Plotando gráfico de numero de cliques por hora:
graph6 <- ggplot(numero_cliques, aes(x = hora, y = N)) +
  geom_bar(stat="identity", fill="steelblue4")+
  ggtitle("Numero de cliques por hora")+
  xlab("Hora") + 
  ylab("Numero de cliques")
graph6


# Verificando o numero de download para cada hora:
dados_time$hora <- hour(dados_time$click_time)
numero_download <- dados_time[is_attributed == 1, .N,
                              by = c('hora','is_attributed')]
numero_download

# Plotando gráfico de numero de cliques por hora:
graph7 <- ggplot(numero_download, aes(x = hora, y = N)) +
  geom_bar(stat="identity", fill="steelblue4")+
  ggtitle("Numero de dowload por hora")+
  xlab("Hora") + 
  ylab("Quantidade de dowload")
graph7


# Plotando os dois gráficos simultaneos:
grid.arrange(graph6, graph7)
# O menor numero de cliques ocorre proximo das 20hrs. O numero de downloads segue
# o mesmo padrão.


# 5 - Construção do modelo preditivo:
# Dividindo os dados em treino (70%) e teste (30%)
set.seed(0)
amostra <- sample.split(dados$is_attributed, SplitRatio = 0.7)
dados_treino <- subset(dados, amostra == TRUE)
dados_teste <- subset(dados, amostra == FALSE)

str(dados_treino)
str(dados_teste)

# Verificando a proporção dos dados de treino e teste:
prop.table(table(dados_treino$is_attributed))*100
prop.table(table(dados_teste$is_attributed))*100

# Criando o modelo sem balanceamento dos dados:
?rpart
modelo1 <- rpart (data = dados_treino,
                is_attributed ~ .,
                control = rpart.control(cp = .0005))
print(modelo1)

# Previsão
previsao1 <- predict(modelo1, dados_teste, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao1, dados_teste$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste$is_attributed, previsao1, plotit = T, col = "red")

# Acurácia = 0.9979
# Score AUC = 0.603

# Se analisarmos apenas a acuracia do modelo, estaria excelente (99%).
# Mais analisando a curva AUC, temos o valor de 0.603. Podemos melhorar
# o modelo balanceando os dados.

## 6 - Balanceando os dados para criação do modelo:
# Aplicando ROSE em dados de treino e checando a proporção de classes
?ROSE

# Alterando o tipo da variavél click_time para numérica:
dados_treino$click_time <- as.numeric(dados_treino$click_time)
dados_teste$click_time <- as.numeric(dados_teste$click_time)
# Verificando os dados
str(dados_treino)
str(dados_teste)

# Balanceando os dados de treino
rose_treino <- ROSE(is_attributed ~.,
                    data = dados_treino, seed = 1)$data
prop.table(table(rose_treino$is_attributed))*100
# Agora os dados estão com uma proporção de ~50% para as duas classes.

# Balanceando os dados de teste
rose_teste <- ROSE(is_attributed ~.,
                    data = dados_teste, seed = 1)$data
prop.table(table(rose_teste$is_attributed))*100
# Agora os dados estão com uma proporção de ~50% para as duas classes.

# Criando um novo modelo com os dados balanceados:
modelo2 <- rpart (data = rose_treino,
                  is_attributed ~ .,
                  control = rpart.control(cp = .0005))

# Previsão
previsao2 <- predict(modelo2, rose_teste, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao2, rose_teste$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(rose_teste$is_attributed, previsao2, plotit = T, col = "red")

# Acurácia = 0.8651
# Score AUC = 0.865

# Diminuimos a acuracia do modelo, porém, o score AUC aumentou significativamente.
# Portanto, o modelo2 é melhor que o modelo1, indicando que os dados precisam estar
# Balanceados para uma melhor performance do modelo.


# 7 - Otimizando o modelo:
# 1) Verificando outros métodos de balanceamento dos dados:
## Over sampling:
# Balanceando dos dados de treino:
?ovun.sample
dados_treino_over <- ovun.sample(is_attributed ~ ., data = dados_treino, 
                          method = "over")$data
prop.table(table(dados_treino_over$is_attributed))*100

# Balanceando os dados de teste
dados_teste_over <- ovun.sample(is_attributed ~ ., data = dados_teste, 
                                method = "over")$data
prop.table(table(dados_teste_over$is_attributed))*100

# Criando um novo modelo com os dados balanceados:
modelo3 <- rpart (data = dados_treino_over,
                  is_attributed ~ .,
                  control = rpart.control(cp = .0005))

# Previsão
previsao3 <- predict(modelo3, dados_teste_over, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao3, dados_teste_over$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste_over$is_attributed, previsao3, plotit = T, col = "red")

# Acurácia = 0.9319
# Score AUC = 0.932



## Under sampling:
## Balanceando dos dados de treino:
dados_treino_under <- ovun.sample(is_attributed ~ ., data = dados_treino, 
                                 method = "under")$data
prop.table(table(dados_treino_under$is_attributed))*100

# Balanceando os dados de teste
dados_teste_under <- ovun.sample(is_attributed ~ ., data = dados_teste, 
                                method = "under")$data
prop.table(table(dados_teste_under$is_attributed))*100

# Criando um novo modelo com os dados balanceados:
modelo4 <- rpart (data = dados_treino_under,
                  is_attributed ~ .,
                  control = rpart.control(cp = .0005))

# Previsão
previsao4 <- predict(modelo4, dados_teste_under, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao4, dados_teste_under$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste_under$is_attributed, previsao4, plotit = T, col = "red")

# Acurácia = 0.9111
# Score AUC = 0.911



# Both sampling
# Balanceando dos dados de treino:
dados_treino_both <- ovun.sample(is_attributed ~ ., data = dados_treino, 
                                  method = "both")$data
prop.table(table(dados_treino_both$is_attributed))*100

# Balanceando os dados de teste
dados_teste_both <- ovun.sample(is_attributed ~ ., data = dados_teste, 
                                 method = "both")$data
prop.table(table(dados_teste_both$is_attributed))*100

# Criando um novo modelo com os dados balanceados:
modelo5 <- rpart (data = dados_treino_both,
                  is_attributed ~ .,
                  control = rpart.control(cp = .0005))

# Previsão
previsao5 <- predict(modelo5, dados_teste_both, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao5, dados_teste_both$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste_both$is_attributed, previsao5, plotit = T, col = "red")

# Acurácia = 0.931
# Score AUC = 0.931


## Score dos 4 modelos de balanceamento:
# ROSE = 
    # Acurácia = 0.8651
    # Score AUC = 0.865

# OVER SAMPLING = 
    # Acurácia = 0.9319
    # Score AUC = 0.932

# UNDER SAMPLING = 
    # Acurácia = 0.9111
    # Score AUC = 0.911

# BOTH SAMPLING = 
    # Acurácia = 0.93
    # Score AUC = 0.931
  
# O balanceamento utilizando over sampling e both sampling apresentaram um
# melhor resultado, com valores proximos um do outro, porém, o balanceamento,
# utilizando over sampling não diminui tanto o tamanho do dataset em relação
# aos demais métodos de balanceamento, por isso, será utilizado o modelo de
# over sampling.


#2) Testando outros algoritmos de classificação:
str(dados_teste_over)
str(dados_treino_over)

#2.1) Utilizando o random forest:
library(randomForest)

# Criando o modelo
?randomForest
modelo6 <- randomForest(data = dados_treino_over,
                  is_attributed ~ .,
                  ntree = 100,
                  nodesize = 10)
print(modelo6)

# Previsão
previsao6 <- predict(modelo6, dados_teste_over, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao6, dados_teste_over$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste_over$is_attributed, previsao6, plotit = T, col = "red")

# Acurácia = 0.6702
# Score AUC = 0.67

#2.2) Utilizando o C50:
library(C50)

# Criando o modelo
?C5.0
modelo7 <- C5.0(data = dados_treino_over,
                        is_attributed ~ .)
print(modelo7)

# Previsão
previsao7 <- predict(modelo7, dados_teste_over, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao7, dados_teste_over$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste_over$is_attributed, previsao7, plotit = T, col = "red")

# Acurácia = 0.7982
# Score AUC = 0.798


#2.3) Utilizando o NaiveBayes:
library(e1071)

# Criando o modelo
?naiveBayes
modelo8 <- naiveBayes(data = dados_treino_over,
                is_attributed ~ .)
print(modelo8)

# Previsão
previsao8 <- predict(modelo8, dados_teste_over, type = 'class')

# Criando uma confusion matrix:
caret::confusionMatrix(previsao8, dados_teste_over$is_attributed,
                       positive = '1')

# Criando a Curva ROC para encontrar a métrica AUC:
roc.curve(dados_teste_over$is_attributed, previsao8, plotit = T, col = "red")

# Acurácia = 0.7761
# Score AUC = 0.776


### Conclusão:
# O melhor resultado foi utilizando o algoritmo RPART com os dados balanceados
# pelo algoritmo ROSE, utilizando o método de Over Sampling. Nessas condições, foi
# obtido uma acurácia de 93,19% e um Score AUC de 93,20%.