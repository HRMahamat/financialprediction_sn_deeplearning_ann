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
from neuralprophet import NeuralProphet
import matplotlib.patheffects as patheffects

# ==========================================
# 1. STYLE CSS "CYBER-TRADING" (ULTRA-GLOW)
# ==========================================
st.set_page_config(page_title="Alpha Predict Pro", layout="wide", page_icon="📈")

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
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=True)
        self.fc_res1 = nn.Linear(128, 128)
        self.fc_res2 = nn.Linear(128, 128)
        self.ln = nn.LayerNorm(128)
        self.out = nn.Linear(128, 14)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.pad(x, (2, 0))
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        output, _ = self.lstm2(x)
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
    # 1. LSTM Engine
    try:
        with open('Hamad_Rassem_Mahamat_sn_ann_lstm_best_model.pkl', 'rb') as f:
            lstm_scaler = pickle.load(f)
        lstm_model = CNN_LSTM_Net(num_features=5)
        lstm_model.load_state_dict(torch.load('Hamad_Rassem_Mahamat_sn_ann_lstm_best_model.pth', map_location='cpu'))
        lstm_model.eval()
    except Exception as e:
        st.error(f"Erreur LSTM: {e}")
        lstm_model, lstm_scaler = None, None

    # 2. NeuralProphet Engine
    try:
        with open('Hamad_Rassem_Mahamat_sn_ann_neuralprophet_best_model.pkl', 'rb') as f:
            prophet_model = pickle.load(f)
    except Exception as e:
        st.error(f"Erreur Prophet: {e}")
        prophet_model = None

    return (lstm_model, lstm_scaler), prophet_model


# ==========================================
# 4. TRAITEMENT DES DONNÉES TEMPS RÉEL
# ==========================================
def get_live_data(ticker="MSFT"):
    day = datetime.datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start="2020-01-01", end=day)
    vix = yf.download("^VIX", start="2020-01-01", end=day)['Close']

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)

    # Feature Engineering (Identique au training)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['VIX_Norm'] = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    df['RSI'] = ta.rsi(df['Close'], length=14) / 100
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['Vol_Shock'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    # Pour Prophet (Besoin de colonnes spécifiques)
    df_prophet = df.copy().reset_index()
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df_prophet['Vol_Mom'] = df_prophet['Volume'].pct_change()

    return df.dropna(), df_prophet.dropna()


# ==========================================
# 5. INTERFACE UTILISATEUR
# ==========================================
st.markdown('<h1 class="title-text">ALPHA PREDICT PRO</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Neuro-Forecasting Engine for OpenAI (MSFT) Proxy</p>",
            unsafe_allow_html=True)

(m_lstm, s_lstm), m_prophet = load_neuro_engines()

tab1, tab2, tab3 = st.tabs(["🚀 TERMINAL DE PRÉDICTION", "📊 ANALYSE TECHNIQUE", "⚙️ MOTEURS IA"])

with tab1:
    col_set1, col_set2 = st.columns([1, 3])

    with col_set1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("### Configuration")
        ticker = st.text_input("Symbole Boursier", "MSFT")
        period = st.selectbox("Horizon de prédiction", ["14 Jours (2 Semaines)"])
        confidence = st.slider("Intervalle de Confiance", 0.80, 0.99, 0.95)

        run_analysis = st.button("LANCER L'IA")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_analysis:
        with st.spinner("Extraction des signaux de marché..."):
            df_live, df_prophet = get_live_data(ticker)
            last_price = df_live['Close'].iloc[-1]

        col_res1, col_res2 = st.columns(2)

        # --- LOGIQUE PRÉDICTION LSTM ---
        features = ['Log_Ret', 'VIX_Norm', 'RSI', 'MACD', 'Vol_Shock']
        scaled_input = s_lstm.transform(df_live[features])
        last_seq = torch.FloatTensor(scaled_input[-60:]).unsqueeze(0)

        with torch.no_grad():
            preds_log_ret = m_lstm(last_seq).numpy()[0]

        # Dénormalisation (Calcul cumulatif des Log-Returns)
        future_prices_lstm = [last_price * np.exp(np.sum(preds_log_ret[:i + 1])) for i in range(14)]

        # --- LOGIQUE PRÉDICTION PROPHET ---
        future_df = m_prophet.make_future_dataframe(df_prophet, periods=14, n_historic_predictions=False)
        # On doit ajouter les regresseurs dans le futur (approximation par les dernières valeurs connues)
        for reg in ['RSI', 'MACD', 'VIX_Norm', 'Vol_Mom']:
            future_df[reg] = df_prophet[reg].iloc[-1]

        forecast = m_prophet.predict(future_df)
        future_prices_prophet = forecast.iloc[-1][[f'yhat{i}' for i in range(1, 15)]].values

        with col_res1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"#### ⚡ Score Neuro-CNN-LSTM")
            st.title(f"{future_prices_lstm[-1]:.2f} $")
            delta = ((future_prices_lstm[-1] / last_price) - 1) * 100
            st.metric("Variation attendue", f"{delta:.2f}%", delta_color="normal")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_res2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"#### 🔮 Score NeuralProphet")
            st.title(f"{future_prices_prophet[-1]:.2f} $")
            delta_p = ((future_prices_prophet[-1] / last_price) - 1) * 100
            st.metric("Variation attendue", f"{delta_p:.2f}%", delta_color="normal")
            st.markdown("</div>", unsafe_allow_html=True)

        # GRAPHIQUE COMPARATIF
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig = go.Figure()
        days = [f"J+{i}" for i in range(1, 15)]

        fig.add_trace(go.Scatter(x=days, y=future_prices_lstm, name='Hybrid CNN-LSTM',
                                 line=dict(color='#00f0ff', width=4, dash='solid')))
        fig.add_trace(go.Scatter(x=days, y=future_prices_prophet, name='AR-Net Prophet',
                                 line=dict(color='#ff2a68', width=4, dash='dot')))

        fig.update_layout(title="Trajectoire de Prix Anticipée (14 Jours)",
                          template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Orbitron"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    if 'df_live' in locals() or 'df_live' in globals():
        st.write("### Analyse des Indicateurs de Tension")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df_live['RSI'].tail(50))
            st.caption("RSI (Relative Strength Index) - Normalisé")
        with c2:
            st.line_chart(df_live['VIX_Norm'].tail(50))
            st.caption("VIX (Volatility Index) - Z-Score")

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
st.sidebar.image("https://img.icons8.com/nolan/512/ai.png", width=100)
st.sidebar.markdown("---")
st.sidebar.write("🟢 **Status Engine :** Optimal")
st.sidebar.write(f"📅 **Dernière Synchro :** {datetime.datetime.now().strftime('%H:%M:%S')}")