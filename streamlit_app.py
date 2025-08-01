import streamlit as st
import boto3
import pickle
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="🎬 Movie Engagement Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.mlb = None
    st.session_state.ohe_user = None
    st.session_state.prediction_history = []

@st.cache_data
def load_model_artifacts():
    """Load model artifacts from S3"""
    try:
        os.environ['AWS_PROFILE'] = 'degen-mlops'
        session = boto3.Session(profile_name='degen-mlops')
        s3_client = session.client('s3')
        
        bucket = 'phase3-mlops-source-bucket-degen-1'
        model_path = 'models/randomforestclassifier'
        
        artifacts = ['model.pkl', 'scaler.pkl', 'multilabel_binarizer.pkl', 'ohe_user.pkl']
        
        # Download artifacts
        for artifact in artifacts:
            s3_key = f"{model_path}/{artifact}"
            local_path = f"/tmp/{artifact}"
            s3_client.download_file(bucket, s3_key, local_path)
        
        # Load artifacts
        with open('/tmp/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('/tmp/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('/tmp/multilabel_binarizer.pkl', 'rb') as f:
            mlb = pickle.load(f)
        
        with open('/tmp/ohe_user.pkl', 'rb') as f:
            ohe_user = pickle.load(f)
        
        return model, scaler, mlb, ohe_user, True
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, False

def make_prediction(age, gender, occupation, genres, release_year):
    """Make prediction using loaded model"""
    try:
        # Create feature vector
        data = {
            'ReleaseYear': [release_year],
            'Gender': [gender],
            'Age': [age],
            'Occupation': [occupation]
        }
        
        df = pd.DataFrame(data)
        
        # Process genres
        genre_features = st.session_state.mlb.transform([genres])
        genre_df = pd.DataFrame(genre_features, columns=st.session_state.mlb.classes_)
        
        # Process user features
        user_features = st.session_state.ohe_user.transform(df[['Gender', 'Age', 'Occupation']])
        user_df = pd.DataFrame(user_features, columns=st.session_state.ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']))
        
        # Combine features
        feature_df = pd.concat([df[['ReleaseYear']], genre_df, user_df], axis=1)
        
        # Scale release year
        feature_df['ReleaseYear'] = st.session_state.scaler.transform(feature_df[['ReleaseYear']])
        
        # Make prediction
        prediction_proba = st.session_state.model.predict_proba(feature_df)[0]
        prediction = st.session_state.model.predict(feature_df)[0]
        
        return prediction_proba[1], prediction, prediction_proba
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Engagement Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict whether a user will engage with a movie based on their profile and movie characteristics")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading ML model from S3..."):
            model, scaler, mlb, ohe_user, success = load_model_artifacts()
            if success:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.mlb = mlb
                st.session_state.ohe_user = ohe_user
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model")
                return
    
    # Sidebar for inputs
    st.sidebar.header(" User & Movie Details")
    
    # User inputs
    st.sidebar.subheader("👤 User Profile")
    age = st.sidebar.slider("Age", 7, 73, 25)
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    
    occupations = [
        "administrator", "artist", "doctor", "educator", "engineer", 
        "entertainment", "executive", "healthcare", "homemaker", "lawyer",
        "librarian", "marketing", "none", "other", "programmer", 
        "retired", "salesman", "scientist", "student", "technician", "writer"
    ]
    occupation = st.sidebar.selectbox("Occupation", occupations, index=occupations.index("student"))
    
    # Movie inputs
    st.sidebar.subheader("🎬 Movie Details")
    release_year = st.sidebar.slider("Release Year", 1920, 2000, 1995)
    
    available_genres = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
        "Thriller", "War", "Western", "unknown"
    ]
    
    genres = st.sidebar.multiselect(
        "Movie Genres", 
        available_genres, 
        default=["Action", "Comedy"]
    )
    
    if not genres:
        st.sidebar.warning("Please select at least one genre")
        return
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Engagement", type="primary"):
        with st.spinner("Making prediction..."):
            prob, pred, full_proba = make_prediction(age, gender, occupation, genres, release_year)
            
            if prob is not None:
                # Store in history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'age': age,
                    'gender': gender,
                    'occupation': occupation,
                    'genres': ', '.join(genres),
                    'release_year': release_year,
                    'probability': prob,
                    'prediction': pred
                })
                
                # Main prediction display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction card
                    engagement_text = "Will Engage! 🎉" if pred == 1 else "Won't Engage 😐"
                    color = "#4CAF50" if pred == 1 else "#FF5722"
                    
                    st.markdown(f"""
                    <div style="background: {color}; padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;">
                        <h2>{engagement_text}</h2>
                        <h1>{prob:.1%}</h1>
                        <p>Engagement Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Engagement Score"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                st.subheader("Prediction Details")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(" User Age", f"{age} years")
                
                with col2:
                    st.metric(" Genres", f"{len(genres)} selected")
                
                with col3:
                    st.metric(" Movie Era", f"{release_year}s")
                
                with col4:
                    confidence = max(full_proba)
                    st.metric(" Confidence", f"{confidence:.1%}")
                
                # Feature importance (mock)
                st.subheader("Key Factors")
                factors = {
                    'Age Group': np.random.uniform(0.1, 0.3),
                    'Movie Genres': np.random.uniform(0.2, 0.4),
                    'Occupation': np.random.uniform(0.1, 0.25),
                    'Release Year': np.random.uniform(0.05, 0.2),
                    'Gender': np.random.uniform(0.05, 0.15)
                }
                
                fig = px.bar(
                    x=list(factors.values()),
                    y=list(factors.keys()),
                    orientation='h',
                    title="Feature Importance",
                    color=list(factors.values()),
                    color_continuous_scale="viridis"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.subheader("Prediction History")
        
        # Convert to DataFrame
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        # Show recent predictions
        st.dataframe(
            df_history[['timestamp', 'age', 'gender', 'occupation', 'genres', 'release_year', 'probability', 'prediction']].tail(10),
            use_container_width=True
        )
        
        # Probability distribution
        if len(df_history) > 1:
            fig = px.histogram(
                df_history, 
                x='probability', 
                nbins=20,
                title="Distribution of Engagement Probabilities",
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    with st.expander("Model Information"):
        if st.session_state.model_loaded:
            st.write(f"**Model Type:** {type(st.session_state.model).__name__}")
            st.write(f"**Available Genres:** {len(st.session_state.mlb.classes_)}")
            st.write(f"**User Features:** {len(st.session_state.ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']))}")
            st.write(f"**Total Features:** {len(st.session_state.mlb.classes_) + len(st.session_state.ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation'])) + 1}")

if __name__ == "__main__":
    main()