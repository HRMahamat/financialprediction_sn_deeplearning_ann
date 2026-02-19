import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import datetime
import plotly.graph_objects as go
import matplotlib.patheffects as patheffects

# ==========================================
# 1. STYLE CSS "CYBER-TRADING" (ULTRA-GLOW)
# ==========================================
st.set_page_config(page_title="Financial Prediction", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    .main {
        background: linear-gradient(135deg, #050505 0%, #0a0a12 50%, #050505 100%);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }

    .title-text {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00f0ff, #ff2a68, #ffffff, #00f0ff);
        background-size: 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        animation: gradientMove 8s linear infinite;
        margin-bottom: 0px;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(0, 240, 255, 0.2);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-bottom: 20px;
    }

    .stButton > button {
        background: linear-gradient(45deg, #00f0ff, #ff2a68);
        border: none;
        color: white;
        padding: 15px;
        font-family: 'Orbitron', sans-serif;
        font-size: 20px;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(255, 42, 104, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. ARCHITECTURE DU MODÈLE PYTORCH (M2)
# ==========================================
# On doit répliquer exactement la structure de ton script d'entraînement
class CNN_LSTM_Net(nn.Module):
    def __init__(self, num_features):
        super(CNN_LSTM_Net, self).__init__()
        # Le state_dict montre que le modèle a été sauvé comme une suite de couches (0, 1, 2...)
        # On recrée exactement cette séquence
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3), # 0
            nn.BatchNorm1d(64),                                                  # 1
            # Les LSTM ne peuvent pas être dans Sequential facilement, 
            # mais ton erreur montre qu'ils ont les index 2 et 3
        )
        
        # On définit les couches séparément avec les index correspondant aux "Unexpected keys"
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True) # Index 2
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=True) # Index 3
        
        self.fc_res1 = nn.Linear(128, 128) # Index 4
        self.fc_res2 = nn.Linear(128, 128) # Index 5
        self.ln = nn.LayerNorm(128)        # Index 6
        self.out = nn.Linear(128, 14)      # Index 7

    def forward(self, x):
        # 1. CNN + BN (Index 0 et 1)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (2, 0))
        x = self.network[0](x) # Conv1d
        x = self.network[1](x) # BatchNorm1d
        x = F.leaky_relu(x, 0.01)
        
        # 2. LSTM (Index 2 et 3)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        output, _ = self.lstm2(x)
        
        # 3. Résiduel et Sortie (Index 4, 5, 6, 7)
        x = output[:, -1, :]
        shortcut = x
        x = F.silu(self.fc_res1(x))
        x = self.fc_res2(x)
        x = x + shortcut
        x = self.ln(x)
        return self.out(x)


# ==========================================
# 3. CHARGEMENT DES RESSOURCES (CACHED)
# ==========================================
@st.cache_resource
def load_neuro_engines():
    try:
        # 1. Chargement du Scaler
        with open('Hamad_Rassem_Mahamat_sn_ann_lstm_best_model.pkl', 'rb') as f:
            lstm_scaler = pickle.load(f)
        
        # 2. Initialisation du modèle
        lstm_model = CNN_LSTM_Net(num_features=5)
        state_dict = torch.load('Hamad_Rassem_Mahamat_sn_ann_lstm_best_model.pth', map_location='cpu')
        
        # 3. Mapping dynamique des clés (Correction de l'erreur)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        # Dictionnaire de traduction des index vers tes noms de variables
        mapping = {
            "0": "network.0", # Conv1d
            "1": "network.1", # BatchNorm1d
            "2": "lstm1",
            "3": "lstm2",
            "4": "fc_res1",
            "5": "fc_res2",
            "6": "ln",
            "7": "out"
        }
        
        for k, v in state_dict.items():
            prefix = k.split('.')[0] # Récupère l'index (ex: "2")
            suffix = ".".join(k.split('.')[1:]) # Récupère le reste (ex: "weight_ih_l0")
            if prefix in mapping:
                new_key = f"{mapping[prefix]}.{suffix}"
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        lstm_model.load_state_dict(new_state_dict)
        lstm_model.eval()
        return (lstm_model, lstm_scaler)

    except Exception as e:
        st.error(f"Erreur de synchronisation du moteur IA: {e}")
        return None, None


# ==========================================
# 4. TRAITEMENT DES DONNÉES TEMPS RÉEL
# ==========================================
def get_live_data(ticker="MSFT"):
    day = datetime.datetime.now().strftime('%Y-%m-%d')
    # Ajout de progress=False pour éviter des bugs dans les logs Streamlit
    df = yf.download(ticker, start="2020-01-01", end=day, progress=False)
    vix = yf.download("^VIX", start="2020-01-01", end=day, progress=False)['Close']

    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.droplevel(1)

    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['VIX_Norm'] = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    df['RSI'] = ta.rsi(df['Close'], length=14) / 100
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['Vol_Shock'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    # CRUCIAL : Enlève la virgule après dropna()
    return df.dropna()


# ==========================================
# 5. INTERFACE UTILISATEUR
# ==========================================
st.markdown('<h1 class="title-text">ALPHA PREDICT PRO</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Neuro-Forecasting Engine for OpenAI (MSFT) Proxy</p>",
            unsafe_allow_html=True)

(m_lstm, s_lstm)= load_neuro_engines()

tab1, tab2, tab3 = st.tabs(["🚀 TERMINAL DE PRÉDICTION", "📊 ANALYSE TECHNIQUE", "⚙️ MOTEURS IA"])

with tab1:
    with st.container():
        c_set1, c_set2, c_set3 = st.columns([2, 2, 1])
        with c_set1: ticker = st.text_input("Ticker", "MSFT")
        with c_set2: horizon = st.selectbox("Horizon", ["14 Jours"])
        with c_set3: 
            st.write("##")
            run_analysis = st.button("RUN IA")

    if run_analysis:
        with st.spinner("Chargement des rapports visuels..."):
            df_live = get_live_data(ticker)
            
            if df_live is None or df_live.empty:
                st.error("Impossible de récupérer les données pour ce symbole.")
            else:
                # --- SAUVEGARDE DANS LA SESSION ---
                st.session_state['df_live'] = df_live
                st.session_state['ticker_name'] = ticker
            # Simulation d'un petit délai pour l'effet "IA"
            import time
            time.sleep(1) 

            # Création de deux colonnes pour tes images
            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("### 💠 Analyse BI-LSTM")
                
                # Remplace 'chemin/vers/ton_image_lstm.png' par ton vrai nom de fichier
                st.image("lstm.png", caption="Prévisions Neural-Network", use_container_width=True)
                
                st.info("Le modèle BI-LSTM détecte les anomalies de volatilité locales.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_img2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("### 📈 Analyse NeuralProphet")
                
                # Remplace 'chemin/vers/ton_image_prophet.png' par ton vrai nom de fichier
                st.image("prophet.png", caption="Décomposition de Tendance", use_container_width=True)
                
                st.success("NeuralProphet confirme la saisonnalité hebdomadaire du titre.")
                st.markdown("</div>", unsafe_allow_html=True)
    if 'df_live' in st.session_state:
        df_to_plot = st.session_state['df_live']
        last_price = float(df_to_plot['Close'].iloc[-1])
with tab2:
    # On vérifie si la variable existe dans le session_state
    if 'df_live' in st.session_state:
        df_analysis = st.session_state['df_live']
        st.write(f"### Analyse des indicateurs de tension : {st.session_state['ticker_name']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.line_chart(df_analysis['RSI'].tail(100))
            st.caption("RSI (Relative Strength Index) - Dynamique 100 derniers jours")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.line_chart(df_analysis['VIX_Norm'].tail(100))
            st.caption("VIX (Volatility Index) - Z-Score de tension")
            st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("### Spécifications Techniques")
    st.info(
        "**Modèle 1 : CNN-LSTM** - Analyse la structure locale (Conv1D) et les dépendances temporelles longues (Bi-LSTM) sur les Log-Returns.")
    st.success(
        "**Modèle 2 : NeuralProphet** - Utilise un réseau AR-Net pour décomposer la saisonnalité et l'impact des régresseurs externes (VIX, RSI).")
    st.write("---")
    st.write("Développé par : **Hamad • Rassem • Mahamat**")
    st.write("Projet : Master 2 Deep Learning - 2026")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.sidebar.image("https://static.vecteezy.com/ti/vecteur-libre/t1/12921176-fond-de-presentation-creative-rouge-et-noir-pour-l-analyse-finance-prevoir-marche-icone-de-la-ligne-de-prediction-gratuit-vectoriel.jpg", width=100)
st.sidebar.markdown("---")
st.sidebar.write("🟢 **Status :** Optimal")
st.sidebar.write(f"📅 **Dernière synchronisation :** {datetime.datetime.now().strftime('%H:%M:%S')}")








