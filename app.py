import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import os
from PIL import Image
import io
import base64
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile

# === CLASS NAMES (EXACTLY AS IN TRAINING) ===
CLASS_NAMES = {
    'c0': 'Safe Driving',
    'c1': 'Texting - Right',
    'c2': 'Talking Phone - Right',
    'c3': 'Texting - Left',
    'c4': 'Talking Phone - Left',
    'c5': 'Operating Radio',
    'c6': 'Drinking',
    'c7': 'Reaching Behind',
    'c8': 'Hair & Makeup',
    'c9': 'Talking to Passenger'
}

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Driver Behavior Analysis",
    page_icon="🚗",
    layout="wide"
)

# === CSS STYLING ===
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .risk-safe {
        background-color: #28a745;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #dc3545;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-moderate {
        background-color: #ffc107;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === SOUND ALERT FUNCTION ===
def play_alert_sound(risk_level):
    """Generate and play alert sound based on risk level"""
    try:
        duration_ms = 500
        if risk_level == "HIGH RISK":
            frequency = 1000  # High pitch for danger
            sound = Sine(frequency).to_audio_segment(duration=duration_ms)
        elif risk_level == "MODERATE RISK":
            frequency = 700  # Medium pitch for warning
            sound = Sine(frequency).to_audio_segment(duration=duration_ms)
        else:
            return  # No sound for safe driving
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sound.export(f.name, format="wav")
            # Read and encode audio
            with open(f.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Play audio using HTML audio tag
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)
            
            # Clean up
            os.unlink(f.name)
    except Exception as e:
        st.warning(f"Could not play sound: {str(e)}")

# === FEATURE EXTRACTION FUNCTIONS (EXACT MATCH WITH TRAINING) ===
@st.cache_resource
def load_vgg_model():
    """Load VGG16 model for feature extraction - EXACTLY AS IN TRAINING"""
    return VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

def extract_handcrafted_features(img):
    """
    Extract HOG + LBP features from image
    EXACT PARAMETERS FROM TRAINING NOTEBOOK
    
    CRITICAL: Image MUST be resized to 160x160 FIRST!
    """
    # ⚠️ CRITICAL FIX: Resize to 160x160 BEFORE feature extraction
    img_resized = cv2.resize(img, (160, 160))
    
    # Convert to grayscale
    gray = rgb2gray(img_resized)
    
    # HOG features - EXACT parameters from training
    # With 160x160 image and pixels_per_cell=(16,16):
    # Cells per dimension = 160/16 = 10
    # Features = (10-1)*(10-1)*2*2*9 = 2,916
    hog_features = hog(
        gray,
        pixels_per_cell=(16, 16),  # MUST MATCH TRAINING
        cells_per_block=(2, 2),     # MUST MATCH TRAINING
        orientations=9,              # MUST MATCH TRAINING
        feature_vector=True
    )
    
    # LBP features - EXACT parameters from training
    # 26 bins for uniform LBP
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # Combine features: 2,916 + 26 = 2,942 features
    combined = np.concatenate([hog_features, lbp_hist])
    
    return combined

def extract_deep_features(img, vgg_model):
    """
    Extract deep features using VGG16
    EXACT METHOD FROM TRAINING NOTEBOOK
    
    CRITICAL: Image MUST be resized to 160x160!
    """
    # ⚠️ CRITICAL FIX: Resize to exact training size
    img_resized = cv2.resize(img, (160, 160))  # MUST BE 160x160
    
    # Prepare for VGG16
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)
    
    # Extract features
    # Output shape: (1, 5, 5, 512)
    features = vgg_model.predict(img_preprocessed, verbose=0)
    
    # Flatten: 5*5*512 = 12,800 features
    return features.flatten()

def generate_cam_heatmap(img, vgg_model):
    """Generate Class Activation Map heatmap"""
    # Resize to model input size
    img_resized = cv2.resize(img, (160, 160))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)
    
    # Get last conv layer output
    last_conv_output = vgg_model.predict(img_preprocessed, verbose=0)
    
    # Average across feature maps (shape: 1, 5, 5, 512 -> 5, 5)
    heatmap = np.mean(last_conv_output[0], axis=-1)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    
    # Resize to original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    return heatmap_resized, overlay, img

def create_prediction_pipeline(img, model, scaler, pca, lda, vgg_model):
    """
    Complete prediction pipeline
    EXACT FEATURE DIMENSIONS FROM TRAINING
    
    Feature breakdown:
    - Handcrafted (HOG + LBP): 2,942 features
    - Deep (VGG16): 12,800 features
    - Total: 15,742 features
    """
    # Extract handcrafted features (2,942 features)
    # NOW includes resize to 160x160 inside the function
    handcrafted_feat = extract_handcrafted_features(img).reshape(1, -1)
    
    # Extract deep features (12,800 features from 5*5*512)
    # Also resizes to 160x160 inside the function
    deep_feat = extract_deep_features(img, vgg_model).reshape(1, -1)
    
    # Combine features (2,942 + 12,800 = 15,742 total features)
    combined_feat = np.concatenate([handcrafted_feat, deep_feat], axis=1)
    
    # Debug: Print feature dimensions
    st.info(f"🔍 Feature dimensions: Handcrafted={handcrafted_feat.shape[1]}, Deep={deep_feat.shape[1]}, Total={combined_feat.shape[1]}")
    
    # Apply preprocessing pipeline
    scaled_feat = scaler.transform(combined_feat)
    pca_feat = pca.transform(scaled_feat)
    lda_feat = lda.transform(pca_feat)
    
    # Predict
    prediction = model.predict(lda_feat)[0]
    probabilities = model.predict_proba(lda_feat)[0]
    
    # Get class name
    class_label = f'c{prediction}'
    class_name = CLASS_NAMES[class_label]
    confidence = probabilities[prediction]
    
    return class_name, confidence, probabilities

# === MAIN APP ===
def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Driver Behavior Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a driver image
        2. Click **Analyze** button
        3. View results with:
           - Predicted behavior
           - Confidence level
           - Risk assessment
           - Attention heatmap
           - Probability distribution
        4. **Sound alert** for risky behaviors
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Information")
        st.markdown("""
        - **Model**: Random Forest
        - **Accuracy**: 95.4%
        - **Features**: HOG + LBP + VGG16
        - **Classes**: 10 behaviors
        - **Training Data**: State Farm Dataset
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Feature Dimensions")
        st.markdown("""
        - **Image Size**: 160×160 pixels
        - **HOG**: 2,916 features
        - **LBP**: 26 features
        - **Handcrafted Total**: 2,942
        - **VGG16 Deep**: 12,800 features
        - **Combined Total**: 15,742
        - **After PCA**: 100 components
        - **After LDA**: 9 components
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a driver image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a driver"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        if img_array.shape[-1] == 4:  # RGBA to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"Uploaded Image (Original: {img_array.shape[1]}×{img_array.shape[0]})", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Driver Behavior", use_container_width=True):
            with st.spinner('🔄 Loading models and analyzing...'):
                try:
                    # Check if model files exist
                    if not os.path.exists('preprocessing_objects.pkl') or not os.path.exists('best_model_xgboost.pkl'):
                        st.error("⚠️ Model files not found! Please ensure the following files are in the same directory:")
                        st.markdown("""
                        - `preprocessing_objects.pkl`
                        - `best_model_xgboost.pkl`
                        
                        Run the training notebook first to generate these files.
                        """)
                        return
                    
                    # Load models
                    with open('preprocessing_objects.pkl', 'rb') as f:
                        prep_objects = pickle.load(f)
                    
                    scaler = prep_objects['scaler']
                    pca = prep_objects['pca']
                    lda = prep_objects['lda']
                    
                    with open('best_model_xgboost.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    vgg_model = load_vgg_model()
                    
                    st.success("✅ Models loaded successfully!")
                    
                    # Make prediction
                    with st.spinner('🧠 Analyzing behavior...'):
                        class_name, confidence, probabilities = create_prediction_pipeline(
                            img_array, model, scaler, pca, lda, vgg_model
                        )
                    
                    # Generate heatmap
                    with st.spinner('🔥 Generating attention heatmap...'):
                        heatmap, overlay, original = generate_cam_heatmap(img_array, vgg_model)
                    
                    # Determine risk level
                    if 'Safe' in class_name:
                        risk_emoji = "✅"
                        risk_level = "SAFE"
                        risk_class = "risk-safe"
                    elif any(word in class_name.lower() for word in ['texting', 'phone']):
                        risk_emoji = "⚠️"
                        risk_level = "HIGH RISK"
                        risk_class = "risk-high"
                    else:
                        risk_emoji = "🔶"
                        risk_level = "MODERATE RISK"
                        risk_class = "risk-moderate"
                    
                    # Play sound alert
                    play_alert_sound(risk_level)
                    
                    # === DISPLAY RESULTS ===
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    st.markdown(f"### {risk_emoji} **{risk_level}**")
                    
                    # Main prediction box
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Predicted Behavior")
                        st.markdown(f"### **{class_name}**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'<div class="{risk_class}">{risk_emoji} {risk_level}</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🔥 Attention Heatmap")
                        st.markdown("**What the model focuses on:**")
                        st.image(overlay, caption="Model Focus Areas", use_column_width=True)
                    
                    with col2:
                        st.markdown("### 📈 Top 5 Predictions")
                        top5_indices = np.argsort(probabilities)[-5:][::-1]
                        top5_classes = [CLASS_NAMES[f'c{i}'] for i in top5_indices]
                        top5_probs = probabilities[top5_indices]
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['#ff4444' if i == 0 else '#ffaaaa' for i in range(len(top5_classes))]
                        bars = ax.bar(range(len(top5_classes)), top5_probs, color=colors, alpha=0.8)
                        ax.set_xticks(range(len(top5_classes)))
                        ax.set_xticklabels(top5_classes, rotation=45, ha='right')
                        ax.set_ylabel('Probability', fontsize=12)
                        ax.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
                        ax.set_ylim(0, max(top5_probs) * 1.2)
                        ax.grid(axis='y', alpha=0.3)
                        
                        for bar, prob in zip(bars, top5_probs):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                                   f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    # Detailed probability breakdown
                    st.markdown("### 📋 Detailed Probability Breakdown")
                    
                    sorted_indices = np.argsort(probabilities)[::-1]
                    
                    prob_data = []
                    for i, idx in enumerate(sorted_indices):
                        class_name_detail = CLASS_NAMES[f'c{idx}']
                        prob = probabilities[idx]
                        rank = ["1ST 🥇", "2ND 🥈", "3RD 🥉"][i] if i < 3 else f"{i+1}."
                        prob_data.append({
                            "Rank": rank,
                            "Behavior": class_name_detail,
                            "Probability": f"{prob:.4f}",
                            "Percentage": f"{prob*100:.1f}%"
                        })
                    
                    st.table(prob_data)
                    
                    # Success message
                    st.success("✅ Analysis completed successfully!")
                    
                    # Model info box
                    with st.expander("📊 Model Information"):
                        st.markdown("""
                         Training SVM...
                        ✅ SVM Accuracy: 0.930 (93.0%)

                        Training Random Forest...
                        ✅ Random Forest Accuracy: 0.934 (93.4%)

                        Training XGBoost...
                        ✅ XGBoost Accuracy: 0.937 (93.7%)

                        Training AdaBoost...
                        ✅ AdaBoost Accuracy: 0.728 (72.8%)

                         Training Bagging...
                        ✅ Bagging Accuracy: 0.927 (92.7%)

                        ============================================================
                        MODEL PERFORMANCE SUMMARY
                        ============================================================
                        SVM            : 0.930 (93.0%)
                        Random Forest  : 0.934 (93.4%)
                        XGBoost        : 0.937 (93.7%)
                        AdaBoost       : 0.728 (72.8%)
                        Bagging        : 0.927 (92.7%)

                        🏆 BEST MODEL: XGBoost
                         Best Accuracy: 0.937 (93.7%)
                        💾Best model saved as: best_model_xgboost.pkl
                        ✅ Model training completed successfully! 
                        **Features Used**:
                        - Handcrafted (HOG + LBP): 2,942 features
                        - Deep Learning (VGG16): 12,800 features
                        - **Total**: 15,742 features
                        
                        **Dimensionality Reduction**:
                        - PCA → 100 components
                        - LDA → 9 components
                        
                        **Dataset**: State Farm (10 classes)  
                        **Processing**: Real-time capable
                        
                        **Image Processing**:
                        - All images resized to 160×160 pixels
                        - RGB color space
                        - Normalized for VGG16
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.markdown("""
                    **Possible solutions:**
                    1. Ensure all model files are present in the same directory
                    2. Check image format (JPG/PNG)
                    3. Verify image contains a driver
                    4. Make sure the model was trained with the same feature extraction parameters
                    5. Check that image is resized to 160×160 before feature extraction
                    """)

if __name__ == "__main__":
    main()