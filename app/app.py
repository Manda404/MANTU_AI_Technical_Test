import os
import joblib
import pickle
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from datetime import datetime
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🔬 Medical Report Classification Using NLP and Machine Learning",  # Titre de la page
    page_icon="🩺",  # Icône affichée dans l'onglet du navigateur
    layout="wide"  # Mise en page large pour maximiser l'espace
)

# Titre de l'application avec emojis
st.title("🔬 Medical Report Classification Using NLP and Machine Learning")
st.markdown("### 📂 **Mantu - Data Science Technical Assessment**")

# Fonction pour afficher la barre latérale
def display_sidebar():
    st.sidebar.header("🔬 **Classification de Rapports Médicaux**")
    
    # Affichage de l'image de l'entreprise
    st.sidebar.image(
        'https://careers.mantu.com/build/assets/Mantu-careers_thumbnail-X-b281b1af.png', 
        use_container_width=True
    )
    
    # Informations sur la version de l'application
    st.sidebar.markdown(
        """
        <div style="text-align: center; font-size: 18px; font-weight: bold; margin-top: 10px; color: #000000;">
            📅 Version de l'application : 2.0.0
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Informations sur le projet
    st.sidebar.markdown(
        """
        <hr style="margin-top: 10px; margin-bottom: 10px; border: 1px solid #ddd;" />
        <h3 style="text-align: center; font-size: 20px; color: #000000;">
            📝 Informations sur le projet
        </h3>
        <div style="font-size: 16px; line-height: 1.6; color: #000000;">
            <b>👨‍💻 Auteur :</b> Rostand Surel<br/>
            <b>🗓️ Date de création :</b> {creation_date}<br/>
            <b>🔄 Dernière mise à jour :</b> {last_update}<br/>
            <b>📂 Code source :</b> <a href="https://github.com/Manda404" target="_blank" style="color: #000000;">GitHub</a><br/>
            <b>🔗 Profil LinkedIn :</b> <a href="https://www.linkedin.com/in/rostand-surel/" target="_blank" style="color: #000000;">Voir le profil</a>
        </div>
        """.format(
            creation_date="2025-01-28",
            last_update=datetime.now().strftime("%Y-%m-%d")
        ), 
        unsafe_allow_html=True
    )
    
    # Lien vers des ressources utiles
    st.sidebar.markdown("""
    <hr style="border: 1px solid #ddd;">
    <h3 style="text-align: center; font-size: 18px; color: #000000;">🔗 Ressources utiles</h3>
    <ul style="color: #000000;">
        <li><a href="https://scikit-learn.org/stable/" target="_blank" style="color: #000000;">Documentation Scikit-Learn</a></li>
        <li><a href="https://pandas.pydata.org/docs/" target="_blank" style="color: #000000;">Documentation Pandas</a></li>
        <li><a href="https://docs.streamlit.io/" target="_blank" style="color: #000000;">Documentation Streamlit</a></li>
    </ul>
    """, unsafe_allow_html=True)
    
    # Séparation visuelle
    st.sidebar.markdown("---")

# Charger le dataset avec la nouvelle méthode de mise en cache
@st.cache_data
def load_dataset(file_path,nom_fichier='medical-abstracts-dataset.csv'):
    """
    :param file_path: Chemin vers le fichier CSV.
    :return: DataFrame Pandas contenant les données chargées.
    """
    try:
        # Charger le dataset
        data = pd.read_csv(os.path.join(file_path, nom_fichier))
        return data

    except FileNotFoundError:
        print(f"Erreur : le fichier spécifié à '{file_path}' est introuvable.")
        return None
    except pd.errors.EmptyDataError:
        print("Erreur : le fichier est vide.")
        return None
    except pd.errors.ParserError:
        print("Erreur : problème de parsing lors de la lecture du fichier.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
        return None

# Fonction pour afficher la présentation en Markdown
def display_presentation():
    st.markdown(
        """
        ## **Project Objective**  
        
        ### 📂 **Dataset**  
        - Analyze a dataset comprising **2,286 medical reports** extracted from a HuggingFace subset.  
        - Medical reports are classified into the following categories:  
          - **Neoplasms**  
          - **Digestive System Diseases**  
          - **Nervous System Diseases**  
          - **Cardiovascular Diseases**  
          - **General Pathological Conditions**  
        
        ### 🎯 **Goal**  
        - Develop a **Python-based solution** capable of predicting the associated medical department for any given report.  
        
        ### 🔬 **Approach**  
        - Implement multiple **text classification techniques**, including but not limited to:  
          - Traditional machine learning algorithms (e.g., Logistic Regression, SVM).  
          - Deep learning models such as **LSTM**.  
          - Transformer-based models like **BERT** for state-of-the-art performance.  
        - Select and evaluate the **best-performing models** based on relevant metrics.  
        
        ### 🛠 **Interface Streamlit**
        
        - The application is structured into four main tabs:
          1. **🏠 Présentation** - Overview of the project and objectives.
          2. **📈 Exploratory Data Analysis** - Visual analysis and insights from the dataset.
          3. **🏋️‍♂️ Global Performance** - Evaluation of models and their performances.
          4. **🏃‍♂️ Prediction** - Predict medical report categories using the best-trained model.
        
        - The **sidebar** contains:
          - The application logo and title.
          - Version information.
          - Project details such as author, creation date, and last update.
        
        - The main sections are displayed dynamically within the respective tabs, allowing for easy navigation and interaction.
        """
    )

# Initialiser `report_nouveau` dans st.session_state si ce n'est pas déjà fait
if 'report_nouveau' not in st.session_state:
    st.session_state['report_nouveau'] = {
        'Accuracy': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'F1 Score': 0.0
    }

# Fonction pour lire le fichier CSV et retourner les données sous forme de dictionnaire
def lire_metrics(path, nom_fichier='metrics.csv'):
    """
    Lit le fichier CSV contenant les métriques et retourne les données sous forme de dictionnaire.
    
    Args:
        path (str): Le chemin du fichier CSV.
        nom_fichier (str): Le nom du fichier CSV (par défaut 'metrics.csv').
    
    Returns:
        dict: Les données du fichier CSV sous forme de dictionnaire.
    """
    # Créer le chemin complet du fichier
    chemin_complet = os.path.join(path, nom_fichier)
    
    # Lire le fichier CSV
    df = pd.read_csv(chemin_complet)
    
    # Convertir le DataFrame en dictionnaire
    report_dict = df.to_dict(orient='records')[0]  # On prend le premier enregistrement
    return report_dict


# Fonction pour créer une jauge circulaire
def create_gauge(name, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,  # Conversion en pourcentage
        number={'suffix': "%"},
        title={'text': name, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value * 100  # Conversion en pourcentage
            }
        },
        delta={'reference': 50, 'position': "top"}
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=250,
        width=250
    )
    return fig

# Fonction principale pour l'affichage et la comparaison des performances
@st.cache_data
def display_global_performance(path_rapport):
    st.title("Performance Globale")
    st.subheader("Métriques de performance")

    # Chargement des métriques actuelles
    report_actuelle = lire_metrics(path_rapport)

    # Création des graphiques pour chaque métrique
    figures = [create_gauge(name, value) for name, value in report_actuelle.items()]

    with st.expander("Détails et explications des métriques", expanded=True):
        st.markdown("""
        **Accuracy (Exactitude)** — Cette métrique mesure la proportion de prédictions correctes parmi toutes les prédictions effectuées.
        """, unsafe_allow_html=True)
        st.markdown("""
        **Precision (Précision)** — Cette métrique mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives faites par le modèle.
        """, unsafe_allow_html=True)
        st.markdown("""
        **Recall (Rappel)** — Aussi appelé sensibilité, le rappel mesure la proportion d'instances positives réelles qui ont été correctement identifiées par le modèle.
        """, unsafe_allow_html=True)
        st.markdown("""
        **F1 Score** — La moyenne harmonique de la précision et du rappel, qui équilibre les deux métriques. Il est utile lorsqu'il est important de trouver un équilibre entre la précision et le rappel.
        """, unsafe_allow_html=True)

        # Affichage des graphiques dans les colonnes
        col1, col2, col3, col4 = st.columns(4)
        for col, fig in zip([col1, col2, col3, col4], figures):
            col.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Implication des métriques dans notre classification de rapports médicaux", expanded=True):
        st.markdown(
            """
            ## Implication des métriques dans notre problème de classification de rapports médicaux
            
            Dans notre projet de classification de rapports médicaux, les métriques d’évaluation jouent un rôle essentiel pour mesurer la qualité de nos modèles de classification. Voici comment chacune d'elles s’applique à notre contexte :
            
            ### 1️⃣ Accuracy (Exactitude)
            - Mesure la proportion de rapports médicaux correctement classifiés sur l’ensemble des prédictions.
            - **Exemple** : Si nous avons 100 rapports et que 90 sont correctement classés, l’accuracy est de 90%.
            - **Problème potentiel** : Un modèle qui prédit toujours la classe majoritaire peut avoir une haute accuracy sans être efficace.
            
            ### 2️⃣ Precision (Précision)
            - Indique combien de rapports prédits dans une catégorie spécifique sont réellement corrects.
            - **Exemple** : Sur 50 rapports classés comme "Neoplasms", 40 sont corrects → Précision = 40/50 = 80%.
            - **Importance** : Éviter les faux positifs est crucial, par exemple, pour ne pas diagnostiquer à tort une maladie grave.
            
            ### 3️⃣ Recall (Rappel)
            - Mesure la proportion de rapports d’une catégorie spécifique bien détectés.
            - **Exemple** : Sur 60 rapports de "Cardiovascular Diseases", le modèle n’en identifie que 45 → Rappel = 45/60 = 75%.
            - **Importance** : Un rappel élevé est essentiel pour ne pas rater des diagnostics critiques.
            
            ### 4️⃣ F1 Score
            - C’est un **compromis entre précision et rappel**, surtout utile en cas de déséquilibre des classes.
            - **Exemple** : Si un modèle a une précision de 80% et un rappel de 75%, son F1 Score sera d’environ 77%.
            - **Importance** : Assure un bon équilibre et évite de favoriser une métrique au détriment de l’autre.
            
            ---
            
            ## Conclusion et choix des métriques adaptées
            - **Si nous voulons minimiser les faux positifs** (éviter un mauvais diagnostic) → **Privilégier la précision**.
            - **Si nous voulons minimiser les faux négatifs** (éviter de rater un diagnostic important) → **Privilégier le rappel**.
            - **Si nous cherchons un compromis équilibré** → **Utiliser le F1 Score**.
            - **L’accuracy peut être trompeuse**, donc elle sera utilisée avec prudence.
            
            💡 **Notre choix** : Nous privilégions le **F1 Score** pour garantir un bon compromis entre rappel et précision, tout en surveillant les autres métriques pour ajuster nos modèles.
            """
        )


# Fonction pour charger le pipeline sauvegardé
@st.cache_data
def load_saved_pipeline(nom_fichier, chemin):
    """
    Charge un pipeline sauvegardé à partir d'un fichier.

    Args:
        nom_fichier (str): Nom du fichier contenant le modèle sauvegardé.
        chemin (str): Répertoire où se trouve le fichier.

    Returns:
        Pipeline: Le pipeline chargé depuis le fichier.
    """
    chemin_complet = os.path.join(chemin, nom_fichier)
    
    try:
        pipeline = joblib.load(chemin_complet)
        #st.write(f"✅ Modèle chargé avec succès depuis {chemin_complet}")
        return pipeline
    except FileNotFoundError:
        st.write(f"❌ Erreur : Aucun modèle trouvé à {chemin_complet}")
        return None
    except Exception as e:
        st.write(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Fonction pour afficher l'onglet de prédiction
def display_prediction(pipeline):
    st.title("Prédiction")
    option = st.selectbox("Choisissez la méthode de saisie des données :", ["Saisie manuelle", "Charger un fichier CSV"], placeholder="Sélectionnez la méthode de saisie...",index=None)
    if option == "Saisie manuelle":
        st.subheader("Saisie manuelle des données")
        input_data = st.text_area("📝 Collez un résumé médical pour la prédiction :", "")
        if input_data.strip():  # Vérifie que l'entrée n'est pas vide
            if st.button("Prédire"):
                # Prédiction de la classe
                y_pred = pipeline.predict([input_data])[0]

                # Prédiction des probabilités associées à chaque classe
                y_proba = pipeline.predict_proba([input_data])[0]

                # Récupérer la probabilité associée à la classe prédite
                class_index = list(pipeline.classes_).index(y_pred)
                confidence = y_proba[class_index] * 100  # Convertir en pourcentage

                # Affichage du résultat
                st.success(f"🔍 Classe prédite : **{y_pred}**")
                st.info(f"📊 Probabilité de cette classe : **{confidence:.2f}%**")
        else:
            st.warning("⚠️ Veuillez entrer un texte avant de lancer la prédiction.")

    elif option == "Charger un fichier CSV":
        st.subheader("Chargement de données à partir d'un fichier CSV")
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Aperçu des données chargées :")
            st.dataframe(data.head(),use_container_width=True)
            # Vérifier si la colonne cleaned_medical_abstract existe
            if "cleaned_medical_abstract" in data.columns:
                if st.button("Prédire pour tous les textes"):
                    with st.spinner("⏳ Prédiction en cours, veuillez patienter..."):

                        # Prédire les classes
                        predictions = pipeline.predict(data["cleaned_medical_abstract"])
                        probabilities = pipeline.predict_proba(data["cleaned_medical_abstract"]).max(axis=1) * 100  # Probabilité max
                        
                        # Ajouter les colonnes de prédiction
                        data["label_predicted"] = predictions
                        data["probability"] = probabilities
                        
                        # Affichage des résultats
                        st.write("✅ Prédictions effectuées avec succès !")
                        st.dataframe(data.head(), use_container_width=True)
                        
                        # Permettre le téléchargement du fichier mis à jour
                        csv = data.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Télécharger les résultats", csv, "predictions.csv", "text/csv")
            else:
                st.error("❌ Le fichier CSV doit contenir une colonne 'cleaned_medical_abstract'.")


# Fonction pour tracer la distribution
def plot_label_distribution(data, column_name):
    label_counts = data[column_name].value_counts()
    label_percentage = label_counts / label_counts.sum() * 100

    data_pie = pd.DataFrame({
        'Label': label_counts.index,
        'Count': label_counts.values,
        'Percentage': label_percentage.values
    })
    
    # Définir une palette de couleurs
    colors = sns.color_palette("viridis", len(data_pie)).as_hex()
    
    # Création de la figure
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type': 'domain'}, {'type': 'xy'}]],
                        subplot_titles=['Pie Chart 🥧', 'Bar Chart 📊'])

    # Ajout du pie chart
    fig.add_trace(
        go.Pie(
            labels=data_pie['Label'], 
            values=data_pie['Count'],
            name='Labels', 
            marker=dict(colors=colors),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Ajout du bar chart
    fig.add_trace(
        go.Bar(
            x=data_pie['Label'], 
            y=data_pie['Count'], 
            name='Labels', 
            marker=dict(color=colors),
            text=[f"{count} ({percentage:.1f}%)" for count, percentage in zip(data_pie['Count'], data_pie['Percentage'])],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'Distribution of Labels in "{column_name}"',
        title_x=0.5
    )
    
    return fig

# Fonction pour tracer la distribution des longueurs de texte
def plot_text_length_box_swarm_px(df, length_column='original_length', label_column='condition_label'):
    fig = px.box(df, x=label_column, y=length_column, points="all", color=label_column, 
                 title="📏 Distribution des longueurs de texte par catégorie médicale 📊", 
                 labels={length_column: "Longueur du Texte ✍️", label_column: "Condition Médicale 🏥"}, 
                 template="simple_white")
    return fig

def add_text_length_column(df, text_column='medical_abstract', length_column='original_length'):
    """
    Ajoute une colonne spécifiée au dataset avec la longueur des textes d'une colonne donnée.
    
    Args:
    - df (pd.DataFrame): Le DataFrame contenant les données.
    - text_column (str): Nom de la colonne contenant les textes.
    - length_column (str): Nom de la colonne où sera stockée la longueur des textes.
    
    Returns:
    - pd.DataFrame: DataFrame avec la nouvelle colonne contenant les longueurs des textes.
    """
    if text_column not in df.columns:
        raise ValueError(f"La colonne '{text_column}' n'existe pas dans le DataFrame.")
    
    df[length_column] = df[text_column].apply(lambda x: len(str(x)))
    return df

# Fonction pour calculer les statistiques descriptives
def text_length_statistics(df, length_column='original_length', label_column='condition_label'):
    stats_df = df.groupby(label_column)[length_column].describe().reset_index()
    stats_df.columns = ['Condition Médicale', 'Nombre de textes', 'Moyenne', 'Écart-Type', 'Min', 'Q1 (25%)', 'Médiane (50%)', 'Q3 (75%)', 'Max']
    return stats_df

# Fonction pour afficher l'histogramme des longueurs de texte
def plot_text_length_histogram(df, length_column='original_length', label_column='condition_label'):
    fig = px.histogram(df, x=length_column, color=label_column, barmode="overlay", nbins=50,
                       title="📏 Distribution des longueurs de texte par catégorie médicale 📊",
                       labels={length_column: "Longueur du Texte ✍️", label_column: "Condition Médicale 🏥"},
                       template="simple_white")
    return fig

# Fonction pour afficher le Box + Swarm plot
def plot_text_length_box_swarm(df, length_column='original_length', label_column='condition_label'):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=label_column, y=length_column, showfliers=False, width=0.5, boxprops={'facecolor':'lightgray'}, ax=ax)
    sns.swarmplot(data=df, x=label_column, y=length_column, color='black', alpha=0.7, size=4, ax=ax)
    ax.set_title("📊 Distribution des longueurs de texte par catégorie médicale (Box + Swarm)")
    ax.set_xlabel("Condition Médicale 🏥")
    ax.set_ylabel("Longueur du Texte ✍️")
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig


# Fonction pour afficher l'onglet EDA
def display_eda(data):
    st.title("Exploratory Data Analysis (EDA)")

    st.header("_Analyse des Données Médicales Report_ 🏥 :bar_chart: :clipboard: :blue[📊] :orange[🔥]", divider='rainbow')

    st.subheader("🔬 **Visualisation du Jeu de Données Médicales** 📊🏥")

    st.dataframe(
        data.head(), 
        use_container_width=True,  # Ajuste la largeur à celle du conteneur
        #height=300  # Ajuste la hauteur pour éviter le scroll excessif
    )
    # Colonne cible à exclure
    text_column, length_column, target_column ='medical_abstract', 'original_length', 'condition_label'

    # Affichage des visualisations dans un expander
    with st.expander("📊 Show Label Distribution Graphs 🔍"):
        fig = plot_label_distribution(data, target_column)
        st.plotly_chart(fig)

    # Sélection des colonnes pour la distribution des longueurs de texte
    with st.expander("📏 Show Text Length Distribution 📊"):
        data = add_text_length_column(data, text_column='medical_abstract', length_column='original_length')  # Ajout de la colonne '
        fig_text_length = plot_text_length_box_swarm_px(data, length_column, target_column)
        st.plotly_chart(fig_text_length)

    # Affichage des statistiques descriptives
    with st.expander("📈 Show Text Length Statistics 🏥"):
        stats_df = text_length_statistics(data, length_column, target_column)
        st.dataframe(stats_df, use_container_width=True)

    with st.expander("📊 Show Text Length Histogram 📏"):
        fig_hist = plot_text_length_histogram(data, length_column, target_column)
        st.plotly_chart(fig_hist)
    
    with st.expander("📊 Show Text Length Box + Swarm 📏"):
        fig_swarm = plot_text_length_box_swarm(data, length_column, target_column)
        st.pyplot(fig_swarm)

# Interface principale
def main():
    dataset_path = "/Users/surelmanda/Downloads/My-Projects-Data/Project-Data-Scientist/Machine-Leaning-Local/MANTU_AI_Technical_Test/data/original_data"
    path_rapport = '/Users/surelmanda/Downloads/My-Projects-Data/Project-Data-Scientist/Machine-Leaning-Local/MANTU_AI_Technical_Test/metrics'
    nom_fichier_pipeline = "best_pipeline.pkl"
    path_model = "/Users/surelmanda/Downloads/My-Projects-Data/Project-Data-Scientist/Machine-Leaning-Local/MANTU_AI_Technical_Test/model"

    data = load_dataset(dataset_path)
    display_sidebar()
    # model = load_model(model_path)
    tab1, tab2, tab3, tab4 = st.tabs(["1 - 🏠 Présentation", "2 - 📈 Exploratory Data Analysis", "3 - 🏋️‍♂️ Global Performance", "4 - 🏃‍♂️ Prediction"])
    with tab1:
        display_presentation()
    with tab2:
         display_eda(data)
         #st.write('Okey!!!!')   
    with tab3:
         display_global_performance(path_rapport)
    with tab4:
         pipeline = load_saved_pipeline(nom_fichier_pipeline, path_model)
         display_prediction(pipeline)


if __name__ == "__main__":
    main()
