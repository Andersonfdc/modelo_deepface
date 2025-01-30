import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from deepface import DeepFace
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import cv2
from fpdf import FPDF
import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
from matplotlib.animation import FuncAnimation
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Função para suprimir avisos do TensorFlow
def suppress_tf_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

suppress_tf_warnings()

# Configuração do Streamlit
st.set_page_config(page_title="Reconhecimento Facial", page_icon=":beginner:")
st.title("Classificação de Imagens - Amigos vs Familiares")
st.sidebar.image("logo.jpeg", use_column_width=True)
st.sidebar.title("Projeto de Machine Learning")

descricao_projeto1 = """
Este projeto tem como objetivo classificar imagens faciais em duas categorias: familiares e amigos. Para isso, utiliza técnicas avançadas de aprendizado de máquina e visão computacional aplicadas ao reconhecimento facial. No contexto do modelo, considera-se familiares como pessoas de ascendência asiática, enquanto amigos são pessoas de outras etnias.

## Tecnologias e Bibliotecas Utilizadas

- 🤖 **DeepFace**: Extração de embeddings faciais utilizando o modelo Facenet.

- 🦾 **TensorFlow/Keras**: Construção e treinamento de um modelo de classificação baseado em redes neurais artificiais.

- 🏜️ **OpenCV**: Processamento de imagens e detecção de rostos.

- 📈 **Matplotlib e Seaborn**: Visualização de resultados, incluindo a matriz de confusão.

- 💻 **Streamlit**: Interface interativa para upload de imagens e visualização dos resultados.

- 📃 **FPDF**: Geração de relatórios em PDF com estatísticas do modelo.

"""
# Exibir a descrição na barra lateral
st.sidebar.write(descricao_projeto1)

icons = '''
    <div style="text-align: center;">
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
    </div                   
'''
st.sidebar.markdown(icons, unsafe_allow_html=True)

descricao_projeto2 = """

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
"""

# Exibir a descrição na barra lateral
st.sidebar.write(descricao_projeto2)

# Adicionando o link para o LinkedIn
linkedin_icon = '''
    <a href="https://www.linkedin.com/in/andersonfdc/" target="_blank" style="text-decoration: none; color: white; font-weight: bold; display: flex; align-items: center;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/linkedin/linkedin-original.svg" width="20" style="margin-right: 8px;"/> 
        LinkedIn
    </a>
'''
st.sidebar.markdown(linkedin_icon, unsafe_allow_html=True)

class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, status_placeholder, progress_bar, plot_placeholder, matrix_placeholder, validation_data, validation_labels, class_names):
        self.status_placeholder = status_placeholder
        self.progress_bar = progress_bar
        self.plot_placeholder = plot_placeholder
        self.matrix_placeholder = matrix_placeholder
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.class_names = class_names
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    def on_train_begin(self, logs=None):
        self.status_placeholder.write("\U0001F680 Treinamento do modelo em andamento...")
        self.progress_bar.progress(0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history['accuracy'].append(logs.get('accuracy', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))

        df = pd.DataFrame({
            'Época': range(1, len(self.history['accuracy']) + 1),
            'Acurácia Treinamento': self.history['accuracy'],
            'Acurácia Validação': self.history['val_accuracy'],
            'Perda Treinamento': self.history['loss'],
            'Perda Validação': self.history['val_loss']
        })
        
        self.plot_placeholder.line_chart(df.set_index('Época'))
        self.progress_bar.progress((epoch + 1) / 80)

        y_pred = np.argmax(self.model.predict(self.validation_data), axis=1)
        y_true = np.argmax(self.validation_labels, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusão')
        self.matrix_placeholder.pyplot(fig)

    def on_train_end(self, logs=None):
        self.status_placeholder.success("\U0001F3C6 Treinamento concluído com sucesso!")
        self.progress_bar.empty()

# Diretórios de treino e validação
train_dir = './data/train'
validation_dir = './data/validation'

# Validar se os diretórios existem
if not os.path.exists(train_dir) or not os.listdir(train_dir):
    st.error("O diretório de treino está vazio ou não existe.")
    st.stop()
if not os.path.exists(validation_dir) or not os.listdir(validation_dir):
    st.error("O diretório de validação está vazio ou não existe.")
    st.stop()
    
@st.cache_data
def prepare_data(directory):
    embeddings = []
    labels = []
    classes = os.listdir(directory)
    classes = [c for c in classes if not c.startswith(".")]

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            try:
                embedding_result = DeepFace.represent(img_path=file_path, model_name="Facenet", enforce_detection=False)
                embedding = embedding_result[0]['embedding']
                embeddings.append(embedding)
                labels.append(label)
            except Exception as e:
                st.warning(f"Erro ao processar {file_path}: {e}")

    embeddings = np.array(embeddings, dtype="float32")
    labels = np.array(labels, dtype="int32")

    return embeddings, labels, classes

with st.spinner("Carregando dados de treino e validação..."):
    train_data, train_labels, train_classes = prepare_data(train_dir)
    validation_data, validation_labels, validation_classes = prepare_data(validation_dir)

if train_classes != validation_classes:
    st.error("As classes em treino e validação não correspondem.")
    st.stop()

train_labels = to_categorical(train_labels, num_classes=len(train_classes))
validation_labels = to_categorical(validation_labels, num_classes=len(validation_classes))

def build_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model

model = build_model(input_shape=(128,), num_classes=len(train_classes))

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Criando um espaço para o gráfico
status_placeholder = st.empty()
progress_bar = st.progress(0)
plot_placeholder = st.empty()
matrix_placeholder = st.empty()

# Treinar o modelo
st.write("Treinando o modelo...")
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

with st.spinner("Treinando... Isso pode levar algum tempo."):
    history = model.fit(
        train_data, train_labels,
        batch_size=8,
        epochs=80,
        validation_data=(validation_data, validation_labels),
        callbacks=[StreamlitCallback(status_placeholder, progress_bar, plot_placeholder, matrix_placeholder, validation_data, validation_labels, train_classes), early_stopping]
    )

model.save('face_recognition_model.h5')
st.success("Modelo treinado")


# Exibir histórico de treinamento
st.subheader("Resultados do Treinamento")
def plot_training_history(history,save_path="training_history.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.plot(history.history['loss'], label='Perda Treinamento')
    plt.plot(history.history['val_loss'], label='Perda Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Métrica')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    st.pyplot(plt)

plot_training_history(history)

def animate_training_history(history, save_path="training_animation.gif"):
    # Extrair dados do histórico
    epochs = np.arange(1, len(history.history['accuracy']) + 1)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Criar a figura e os eixos
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1.1)  # Ajuste o limite superior se necessário
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Métrica')
    ax.grid(True)

    # Adicionar as linhas para as métricas
    train_acc_line, = ax.plot([], [], label='Acurácia Treinamento', color='blue')
    val_acc_line, = ax.plot([], [], label='Acurácia Validação', color='orange')
    train_loss_line, = ax.plot([], [], label='Perda Treinamento', color='green')
    val_loss_line, = ax.plot([], [], label='Perda Validação', color='red')
    ax.legend()

    # Função de atualização da animação
    def update(epoch):
        train_acc_line.set_data(epochs[:epoch], acc[:epoch])
        val_acc_line.set_data(epochs[:epoch], val_acc[:epoch])
        train_loss_line.set_data(epochs[:epoch], loss[:epoch])
        val_loss_line.set_data(epochs[:epoch], val_loss[:epoch])
        return train_acc_line, val_acc_line, train_loss_line, val_loss_line

    # Criar animação
    anim = FuncAnimation(fig, update, frames=len(epochs) + 1, interval=300, blit=True)

    # Salvar como GIF
    anim.save(save_path, writer='pillow', fps=2)  # Salva como GIF
    plt.close(fig)

    # Exibir no Streamlit
    st.image(save_path)

# Exemplo de uso no Streamlit
if 'history' in locals():  # Certifique-se de que você já treinou um modelo antes
    if st.button("Gerar Animação do Treinamento"):
        st.write("Gerando animação...")
        animate_training_history(history)

# Função para salvar imagem na pasta correspondente
def save_image(image_path, label):
    dest_dir = os.path.join(train_dir, label)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(image_path))
      
    # Se a imagem já existir, remove antes de sobrescrever
    if os.path.exists(dest_path):
        os.remove(dest_path)
    
    os.rename(image_path, dest_path)
    return dest_path

# Matriz de Confusão
def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig(save_path)
    # st.pyplot(plt)
        
# Avaliação
# st.write("Avaliando o modelo...")
y_pred = np.argmax(model.predict(validation_data), axis=1)
y_true = np.argmax(validation_labels, axis=1)
# plot_confusion_matrix(y_true, y_pred, train_classes)
# st.text(classification_report(y_true, y_pred, target_names=train_classes))

def display_classification_report(y_true, y_pred, class_names):
    # Gerar o relatório como dicionário
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Converter para DataFrame
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    # Exibir no Streamlit como tabela interativa
    st.dataframe(report_df)
    
    # # Aplicar estilos com pandas Styler
    # styled_df = report_df.style.background_gradient(subset=["precision", "recall", "f1-score"])

    # # Exibir no Streamlit
    # st.write(styled_df.to_html(), unsafe_allow_html=True)
    st.write()

display_classification_report(y_true, y_pred,train_classes)


def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Relatório de Treinamento e Avaliação", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt="Matriz de Confusão:", ln=True)
    pdf.image("confusion_matrix.png", x=10, y=None, w=180)  # Adiciona a imagem ao PDF
    pdf.ln(10)
    
    # Adicionar relatório de classificação
    pdf.cell(200, 10, txt="Relatório de Classificação:", ln=True)
    report = classification_report(y_true, y_pred, target_names=train_classes)
    pdf.multi_cell(0, 10, txt=report)
    pdf.ln(10)

    # Adicionar Histórico de Treinamento
    pdf.cell(200, 10, txt="Histórico de Treinamento:", ln=True)
    pdf.image("training_history.png", x=10, y=None, w=180)  # Adiciona a imagem ao PDF
    
    pdf.output("relatorio_treinamento.pdf")
    st.success("Relatório PDF gerado: relatorio_treinamento.pdf")

if st.button("Gerar Relatório PDF"):
    with st.spinner("Gerando relatório em PDF..."):
        generate_pdf_report()
        report_path="./relatorio_treinamento.pdf"
        st.download_button("Download Relatório PDF", open(report_path, "rb"), file_name="relatorio.pdf")

def classify_image(image_path, model_path='face_recognition_model.h5'):
    model = tf.keras.models.load_model(model_path)
    
    # Obter o embedding com DeepFace
    embedding_result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
    
    if embedding_result and isinstance(embedding_result, list) and 'embedding' in embedding_result[0]:
        embedding = embedding_result[0]['embedding']  # Extraia o vetor de embedding
    else:
        raise ValueError("Falha ao gerar o embedding da imagem.")
    
    # Converter para numpy array e fazer a predição
    prediction = model.predict(np.expand_dims(embedding, axis=0))
    return train_classes[np.argmax(prediction)]
        
st.subheader("Upload da Imagem")
# Seleção de entrada
option = st.radio("Opções de entrada:", ("Upload de imagem", "Tirar foto com a câmera"))

uploaded_file = None
captured_image = None

if option == "Upload de imagem":
    uploaded_file = st.file_uploader("Envie uma imagem para análise:", type=["jpg", "jpeg", "png"])

if option == "Tirar foto com a câmera":
    uploaded_file = st.camera_input(label="Clique no botão abaixo para capturar uma foto")

def size_img(image_path):
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    return img_h, img_w


if uploaded_file or captured_image:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analisando a imagem..."):
        # Salvar e carregar a imagem
        if uploaded_file:
            image = Image.open(uploaded_file)
        elif captured_image is not None:
            image = Image.open(uploaded_file)
    
        image.save("temp_test_image.jpg")
        img_path="temp_test_image.jpg"
        result = DeepFace.analyze(img_path="temp_test_image.jpg", actions=['emotion'], enforce_detection=False)
        img_h, img_w = size_img(img_path)
        st.image(image, caption=f"Imagem Enviada {img_w}x{img_h}", use_column_width=True)
        st.write("Análise da Imagem:")
        # st.json(result)
        
        # Processar dados de emoção
        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        face_confidence = result[0]['face_confidence']
        region = result[0]['region']
        
        # 1. Gráfico de barras das emoções
        emotion_df = pd.DataFrame(emotions.items(), columns=['Emoção', 'Probabilidade'])
        emotion_df.sort_values(by='Probabilidade', ascending=False, inplace=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(emotion_df['Emoção'], emotion_df['Probabilidade'], color='skyblue')
        ax.set_title("Análise das Emoções")
        ax.set_xlabel("Emoções")
        ax.set_ylabel("Probabilidade (%)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # 2. Destaque da emoção dominante
        st.markdown(f"**Emoção Dominante:** {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2f}%)")
        st.markdown(f"**Confiança na detecção do rosto:** {face_confidence:.2f}")

        # 3. Destaque visual na região detectada
        if region:
            draw = ImageDraw.Draw(image)
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            st.image(image, caption="Rosto Detectado", use_column_width=True)
        else:
            st.warning("Nenhuma região de rosto detectada na imagem enviada.")

                
    st.subheader("Classificação e Resultados")
    try:
       
        with st.spinner("Classificando a imagem..."):
            resultado = classify_image(uploaded_file.name)
            st.success(f"A imagem pertence à classe: {resultado}")
            saved_path = save_image(uploaded_file.name, resultado)
            print(f"Imagem salva em: {saved_path}")
            
    except Exception as e:
        st.error(f"Erro durante a classificação: {e}")

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def get_occlusion_params(image_path, block_ratio=0.1, stride_ratio=0.05):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")
    
    img_h, img_w = img.shape[:2]
    min_dim = min(img_h, img_w)  # Usa a menor dimensão para escalar
    
    block_size = max(10, int(min_dim * block_ratio))  # Evita valores muito pequenos
    stride = max(5, int(min_dim * stride_ratio))

    return block_size, stride



def occlusion_map(image_path, block_size, stride, model_path='face_recognition_model.h5'):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape
    
    faces = detect_faces(img)
    
    # Obter embedding original
    embedding_result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
    if not embedding_result or 'embedding' not in embedding_result[0]:
        raise ValueError("Falha ao gerar o embedding da imagem.")
    original_embedding = np.array(embedding_result[0]['embedding']).reshape(1, -1)
    
    # Carregar modelo
    model = tf.keras.models.load_model(model_path)
    original_prediction = model.predict(original_embedding, verbose=0)
    original_class = np.argmax(original_prediction)

    # Criar mapa de impacto
    heatmap = np.zeros((img_h, img_w))
    total_steps = ((img_h // stride) + 1) * ((img_w // stride) + 1)
    with tqdm.tqdm(total=total_steps, desc="Gerando Occlusion Map", unit="bloco") as pbar:
        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                occluded_img = img.copy()
                occluded_img[y:y+block_size, x:x+block_size] = (0, 0, 0)  # Aplica bloqueio preto
                
                # Salvar temporário e obter embedding modificado
                temp_path = "temp_occlusion.jpg"
                cv2.imwrite(temp_path, cv2.cvtColor(occluded_img, cv2.COLOR_RGB2BGR))
                embedding_result = DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=False)
                if embedding_result and 'embedding' in embedding_result[0]:
                    modified_embedding = np.array(embedding_result[0]['embedding']).reshape(1, -1)
                    modified_prediction = model.predict(modified_embedding, verbose=0)
                    impact = original_prediction[0][original_class] - modified_prediction[0][original_class]
                else:
                    impact = 0  # Caso não consiga obter embedding, não altera
                
                # Atualiza heatmap
                heatmap[y:y+block_size, x:x+block_size] = impact
                pbar.update(1)  # Atualiza barra de progresso

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Desenhar retângulo ao redor do rosto
    for (x, y, w, h) in faces:
        cv2.rectangle(superimposed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    cv2.imwrite("occlusion_map.jpg", cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    st.image(superimposed_img, caption="Mapa de Oclusão", use_column_width=True)

if uploaded_file and st.button("Gerar Mapa de Oclusão"):
    with st.spinner("Gerando mapa de oclusão...O tempo irá variar de acordo com o tamanho da imagem"):
        path = os.path.join(train_dir,resultado,uploaded_file.name)
        print(f"Imagem salva em: {path}")
        try:
            path = os.path.join(train_dir,resultado,uploaded_file.name)
            print(f"Imagem salva em: {path}")
            block_size, stride = get_occlusion_params(path)
            occlusion_map("temp_test_image.jpg",block_size, stride)
        except ValueError as e:
            path = os.path.join(train_dir,resultado,uploaded_file.name)
            print(f"Imagem salva em: {path}")
            print(e)
            st.warning("Não foi possível carregar a imagem 🙁.")
            col1, col2, col3 = st.columns([1, 2, 1])  # Ajuste a proporção conforme necessário
            with col2:
                st.image("luffy.gif")
    

