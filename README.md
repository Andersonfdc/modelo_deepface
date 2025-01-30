[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://<your-custom-subdomain>.streamlit.app](https://modelodeepfacetic.streamlit.app))

<div align="center">
  <img src="logo.jpeg" width="200" />
</div>

Este projeto tem como objetivo classificar imagens faciais em duas categorias: familiares e amigos. Para isso, utiliza técnicas avançadas de aprendizado de máquina e visão computacional aplicadas ao reconhecimento facial. No contexto do modelo, considera-se familiares como pessoas de ascendência asiática, enquanto amigos são pessoas de outras etnias.

## Tecnologias e Bibliotecas Utilizadas

- 🤖 **DeepFace**: Extração de embeddings faciais utilizando o modelo Facenet.

- 🦾 **TensorFlow/Keras**: Construção e treinamento de um modelo de classificação baseado em redes neurais artificiais.

- 🏜️ **OpenCV**: Processamento de imagens e detecção de rostos.

- 📈 **Matplotlib e Seaborn**: Visualização de resultados, incluindo a matriz de confusão.

- 💻 **Streamlit**: Interface interativa para upload de imagens e visualização dos resultados.

- 📃 **FPDF**: Geração de relatórios em PDF com estatísticas do modelo.
<br>
<div align="center"">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" width="40" style="margin: 8px;"/> 
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/anaconda/anaconda-original.svg" width="40" style="margin: 8px;" />
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/opencv/opencv-original-wordmark.svg" width="40" style="margin: 8px;" />
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original-wordmark.svg" width="40" style="margin: 8px;"/>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" width="40" style="margin: 8px;"/>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg"  width="40" style="margin: 8px;"/>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/tensorflow/tensorflow-original.svg" width="40" style="margin: 8px;" />
        <img src="https://icon.icepanel.io/Technology/svg/Matplotlib.svg" width="40" style="margin: 8px;" />
        <img src="https://icon.icepanel.io/Technology/svg/Keras.svg" width="40" style="margin: 8px;" />     
        <img src="https://icon.icepanel.io/Technology/svg/scikit-learn.svg" width="40" style="margin: 8px;" /> 
</div>   

## Fluxo de Trabalho

- **Carregamento dos Dados**: As imagens são carregadas de diretórios de treino e validação.

- **Extração de Embeddings**: Cada imagem é processada pelo DeepFace, extraindo uma representação vetorial do rosto.

- **Construção e Treinamento do Modelo**: Uma rede neural densa com camadas `fully connected`, função de ativação `ReLU` e `dropout` para evitar overfitting.

- **Validação e Avaliação**:

    * Matriz de confusão para análise dos erros do modelo.

    * Relatório de classificação (índices de precisão, recall e F1-score).

- **Geração de Relatório**: Exportação dos resultados em formato PDF.

## Aplicabilidade

O projeto pode ser expandido para diversas aplicações, como controle de acesso, segurança digital e organização automática de álbuns de fotos baseados em grupos sociais.
    
