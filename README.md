# CRIAÇÂO DE UMA REDE NEURAL - CNN - USANDO A BASE CIFAR10 

Componentes do Grupo:
 - Gabriel Franco - 2403394
 - Klaus Dieter Bernhard V Bartels - 2402752
 - Luiz Henrique Juvenal - 2401275
  

Segue abaixo o que foi feito...

O CIFAR-10 é um conjunto de dados amplamente utilizado para treinar e testar modelos de aprendizado de máquina, especialmente redes neurais convolucionais (CNNs). Ele contém 60.000 imagens coloridas de 32x32 pixels, divididas em 10 categorias: avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio e caminhão.

Criamos um notebook em Python


# Passo 1: Importamos TensorFlow e Keras para modelagem da CNN, além de matplotlib para visualizar dados.

# Passo 2: Carregamos o Dataset CIFAR-10 .

Esse Dataset é um conjunto de imagens de 32x32 pixels divididas em 10 categorias. Normalizamos os pixels para facilitar o treinamento. normalizamos os valores dos pixels (/255) das imagens para ficarem entre 0 e 1. Isso melhora o desempenho da rede, evitando que valores altos prejudiquem o aprendizado. Já quando usamos o One-hot encoding , serve como coversão... por exemplo, se a imagem for de um "carro" (classe 1), a conversão transforma isso em: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] . Isso facilita o treinamento em classificação multiclasse.

# Carregar dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizar os dados (entre 0 e 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Converter rótulos para formato categórico (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Passo 3: Criamos a Arquitetura da Rede Neural 
Aqui usamos uma CNN com três camadas convolucionais seguidas de camadas densas. Usamos as três camadas convolucionais com BatchNormalization e Dropout para estabilizar o aprendizado. Ou seja , Criamos um modelo sequencial, onde cada camada é adicionada uma após a outra.

Na primeira camada:

Criamos uma camada convolucional com 64 filtros de tamanho 3x3. Onde:
- ReLU é a função de ativação usada para evitar valores negativos.
- Padding='same' mantém o tamanho da imagem igual após a convolução.
- input_shape=(32, 32, 3) define que nossa entrada tem 32x32 pixels e 3 canais de cor (RGB).
- Batch Normalization estabiliza o treinamento normalizando as ativações.Isso acelera o aprendizado e melhora a generalização.
- MaxPooling reduz o tamanho da imagem pela metade, selecionando apenas os valores mais importantes.
- Dropout desativa 20% dos neurônios aleatoriamente para evitar overfitting.

Na segunda camada:

Criamos mais uma camada convolucional com 128 filtros. Onde:
- Batch Normalization para estabilizar.
- MaxPooling reduz mais a imagem.
- Dropout aumenta para 30%, já que a rede está ficando maior.

Na terceira camada:

Agora temos 256 filtros, permitindo capturar padrões mais complexos.
O Dropout sobe para 40%, reforçando a prevenção contra overfitting.
Na sequencia fazemos "Flatten" , ou seja, transformamos nossa matriz em um vetor unidimensional, essencial para conectar com camadas densas.


Em seguida na camada totalmente conectada...

- 512 neurônios formam uma camada totalmente conectada.
- ReLU é usado para evitar valores negativos.
- Dropout alto (50%) para evitar memorizar os dados.

Camada de Saída para 10 Classes:

- A última camada tem 10 neurônios, correspondendo às 10 classes do CIFAR-10.
- A ativação softmax converte valores em probabilidades (exemplo: carro = 92% de chance).

# Passo 4: Compilar e Treinar o Modelo. Definimos o otimizador, a função de perda e métricas de avaliação, onde:

Na compilação:

- Adam é um otimizador avançado que ajusta automaticamente a taxa de aprendizado.
- A função de perda categorical_crossentropy é usada para classificação multiclasse.
- A métrica accuracy mede a acurácia do modelo.

No treinamento do Modelo:

- Treinamos por 30 épocas (cada época é um ciclo completo pelos dados).
- Batch size de 64 regula quantas imagens são processadas de cada vez.
- Validation data permite ver o desempenho no conjunto de teste.


# Treinamento do modelo sendo demonstrado abaixo
history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))
Epoch 1/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 474s 601ms/step - accuracy: 0.3701 - loss: 2.1156 - val_accuracy: 0.5570 - val_loss: 1.3271
Epoch 2/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 514s 617ms/step - accuracy: 0.5771 - loss: 1.2065 - val_accuracy: 0.6610 - val_loss: 0.9731
Epoch 3/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 491s 603ms/step - accuracy: 0.6442 - loss: 1.0031 - val_accuracy: 0.6838 - val_loss: 0.8970
Epoch 4/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 507s 611ms/step - accuracy: 0.6885 - loss: 0.8875 - val_accuracy: 0.7328 - val_loss: 0.7563
Epoch 5/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 497s 604ms/step - accuracy: 0.7184 - loss: 0.8009 - val_accuracy: 0.7532 - val_loss: 0.7152
Epoch 6/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 508s 612ms/step - accuracy: 0.7373 - loss: 0.7385 - val_accuracy: 0.7419 - val_loss: 0.7364
Epoch 7/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 518s 633ms/step - accuracy: 0.7577 - loss: 0.6865 - val_accuracy: 0.7323 - val_loss: 0.7795
Epoch 8/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 481s 607ms/step - accuracy: 0.7741 - loss: 0.6416 - val_accuracy: 0.7740 - val_loss: 0.6737
Epoch 9/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 529s 641ms/step - accuracy: 0.7873 - loss: 0.6056 - val_accuracy: 0.7602 - val_loss: 0.7086
Epoch 10/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 467s 597ms/step - accuracy: 0.7946 - loss: 0.5797 - val_accuracy: 0.7611 - val_loss: 0.7220
Epoch 11/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 502s 598ms/step - accuracy: 0.8093 - loss: 0.5393 - val_accuracy: 0.7484 - val_loss: 0.7405
Epoch 12/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 520s 620ms/step - accuracy: 0.8220 - loss: 0.5038 - val_accuracy: 0.7661 - val_loss: 0.6969
Epoch 13/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 465s 595ms/step - accuracy: 0.8282 - loss: 0.4927 - val_accuracy: 0.8007 - val_loss: 0.5914
Epoch 14/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 504s 597ms/step - accuracy: 0.8369 - loss: 0.4648 - val_accuracy: 0.8084 - val_loss: 0.5657
Epoch 15/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 502s 598ms/step - accuracy: 0.8428 - loss: 0.4447 - val_accuracy: 0.8097 - val_loss: 0.5738
Epoch 16/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 465s 595ms/step - accuracy: 0.8368 - loss: 0.4586 - val_accuracy: 0.8010 - val_loss: 0.5946
Epoch 17/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 509s 604ms/step - accuracy: 0.8594 - loss: 0.4017 - val_accuracy: 0.8165 - val_loss: 0.5507
Epoch 18/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 498s 599ms/step - accuracy: 0.8625 - loss: 0.3902 - val_accuracy: 0.7456 - val_loss: 0.8120
Epoch 19/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 465s 595ms/step - accuracy: 0.8623 - loss: 0.3871 - val_accuracy: 0.8038 - val_loss: 0.5885
Epoch 20/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 503s 596ms/step - accuracy: 0.8680 - loss: 0.3746 - val_accuracy: 0.8135 - val_loss: 0.5687
Epoch 21/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 473s 605ms/step - accuracy: 0.8749 - loss: 0.3523 - val_accuracy: 0.8313 - val_loss: 0.5182
Epoch 22/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 497s 599ms/step - accuracy: 0.8784 - loss: 0.3466 - val_accuracy: 0.8058 - val_loss: 0.6027
Epoch 23/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 466s 596ms/step - accuracy: 0.8801 - loss: 0.3347 - val_accuracy: 0.8276 - val_loss: 0.5202
Epoch 24/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 503s 597ms/step - accuracy: 0.8850 - loss: 0.3207 - val_accuracy: 0.8224 - val_loss: 0.5436
Epoch 25/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 466s 596ms/step - accuracy: 0.8895 - loss: 0.3147 - val_accuracy: 0.7940 - val_loss: 0.6608
Epoch 26/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 498s 592ms/step - accuracy: 0.8913 - loss: 0.3065 - val_accuracy: 0.8055 - val_loss: 0.6171
Epoch 27/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 506s 597ms/step - accuracy: 0.8939 - loss: 0.2988 - val_accuracy: 0.8343 - val_loss: 0.5128
Epoch 28/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 491s 628ms/step - accuracy: 0.8955 - loss: 0.2952 - val_accuracy: 0.8261 - val_loss: 0.5510
Epoch 29/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 479s 599ms/step - accuracy: 0.8974 - loss: 0.2857 - val_accuracy: 0.8343 - val_loss: 0.5258
Epoch 30/30
782/782 ━━━━━━━━━━━━━━━━━━━━ 497s 593ms/step - accuracy: 0.8975 - loss: 0.2828 - val_accuracy: 0.8353 - val_loss: 0.5407


# Passo 5: Avaliação do Modelo , onde após o treinamento, podemos testar a acurácia:

Se o modelo estiver superajustado, a acurácia do teste será bem menor que a do treino. Abaixo o codigo mostrando a acurácia...

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Acurácia no conjunto de teste: {test_acc:.2f}')

313/313 ━━━━━━━━━━━━━━━━━━━━ 22s 72ms/step - accuracy: 0.8370 - loss: 0.5425

# Acurácia no conjunto de teste: 0.84


# Resultados 

Após os ajustes, conseguimos 84% de acurácia! Isso foi possível graças a:

- Batch Normalization para estabilizar o aprendizado.
- Dropout alto para evitar overfitting.
- Camadas profundas para capturar detalhes mais complexos.
- Otimização com Adam para um treinamento mais eficiente.

Grato

Klaus , Gabriel e Luiz Henrique
