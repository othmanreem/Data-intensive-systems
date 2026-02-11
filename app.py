import gradio as gr
import pandas as pd
import pickle
import os

# Get directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local paths - models loaded from A3/models/ directory
MODEL_PATH = os.path.join(SCRIPT_DIR, "A3/models/champion_model_final_2.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(SCRIPT_DIR, "A3/models/final_champion_model_A3.pkl")
DATA_PATH = os.path.join(SCRIPT_DIR, "A3/A3_Data/train_dataset.csv")

model = None
FEATURE_NAMES = None
MODEL_METRICS = None

# Classification model
classification_model = None
CLASSIFICATION_FEATURE_NAMES = None
CLASSIFICATION_CLASSES = None
CLASSIFICATION_METRICS = None

BODY_REGION_RECOMMENDATIONS = {
    'Upper Body': "Focus on shoulder mobility, thoracic spine extension, and keeping your head neutral.",
    'Lower Body': "Work on hip mobility, ankle dorsiflexion, and knee tracking over toes."
}


def load_champion_model():
    global model, FEATURE_NAMES, MODEL_METRICS
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading champion model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        
        model = artifact["model"]
        FEATURE_NAMES = artifact["feature_columns"]
        MODEL_METRICS = artifact.get("test_metrics", {})
        
        print(f"Model loaded: {len(FEATURE_NAMES)} features")
        print(f"Test R2: {MODEL_METRICS.get('r2', 'N/A')}")
        return True
    
    print(f"Champion model not found at {MODEL_PATH}")
    return False


def load_classification_model():
    global classification_model, CLASSIFICATION_FEATURE_NAMES, CLASSIFICATION_CLASSES, CLASSIFICATION_METRICS
    
    if os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Loading classification model from {CLASSIFICATION_MODEL_PATH}")
        with open(CLASSIFICATION_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        
        classification_model = artifact["model"]
        CLASSIFICATION_FEATURE_NAMES = artifact["feature_columns"]
        CLASSIFICATION_CLASSES = artifact["classes"]
        CLASSIFICATION_METRICS = artifact.get("test_metrics", {})
        
        print(f"Classification model loaded: {len(CLASSIFICATION_FEATURE_NAMES)} features")
        print(f"Classes: {CLASSIFICATION_CLASSES}")
        return True
    
    print(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
    return False


load_champion_model()
load_classification_model()


def predict_score(*feature_values):
    if model is None:
        return "Error", "Model not loaded", ""
    
    features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    raw_score = model.predict(features_df)[0]
    score = max(0, min(1, raw_score)) * 100

    if score >= 80:
        interpretation = "Excellent, great squat form"
    elif score >= 60:
        interpretation = "Good, minor improvements needed"
    elif score >= 40:
        interpretation = "Average, a lot of areas to work on"
    else:
        interpretation = "Needs work, focus on proper form"

    r2 = MODEL_METRICS.get('r2', 'N/A')
    correlation = MODEL_METRICS.get('correlation', 'N/A')
    r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
    corr_str = f"{correlation:.4f}" if isinstance(correlation, (int, float)) else str(correlation)
    
    details = f"""
### Prediction Details
- **Raw Model Output:** {raw_score:.4f}
- **Normalized Score:** {score:.1f}%
- **Assessment:** {interpretation}

### Model Performance
- **Test R-squared:** {r2_str}
- **Test Correlation:** {corr_str}

*Lower deviation values = better form*
    """

    return f"{score:.1f}%", interpretation, details


def predict_weakest_link(*feature_values):
    if classification_model is None:
        return "Error", "Model not loaded", ""
    
    features_df = pd.DataFrame([feature_values], columns=CLASSIFICATION_FEATURE_NAMES)
    
    prediction = classification_model.predict(features_df)[0]
    probabilities = classification_model.predict_proba(features_df)[0]
    
    class_probs = list(zip(CLASSIFICATION_CLASSES, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    
    confidence = max(probabilities) * 100
    recommendation = BODY_REGION_RECOMMENDATIONS.get(prediction, "Focus on exercises that strengthen this region.")
    
    accuracy = CLASSIFICATION_METRICS.get('accuracy', 'N/A')
    f1_weighted = CLASSIFICATION_METRICS.get('f1_weighted', 'N/A')
    acc_str = f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else str(accuracy)
    f1_str = f"{f1_weighted:.2%}" if isinstance(f1_weighted, (int, float)) else str(f1_weighted)
    
    predictions_list = "\n".join([f"{i+1}. **{cp[0]}** - {cp[1]*100:.1f}%" for i, cp in enumerate(class_probs)])
    
    details = f"""
### Prediction Details
- **Predicted Body Region:** {prediction}
- **Confidence:** {confidence:.1f}%

### Probability Distribution
{predictions_list}

### Recommendation
{recommendation}

### Model Performance
- **Test Accuracy:** {acc_str}
- **Test F1 (weighted):** {f1_str}
    """
    
    return prediction, f"Confidence: {confidence:.1f}%", details


def load_example():
    if FEATURE_NAMES is None:
        return [0.5] * 35
    
    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        available_features = [f for f in FEATURE_NAMES if f in df.columns]
        sample = df[available_features].sample(1).values[0]
        return [float(x) for x in sample]
    except Exception as e:
        print(f"Error loading example: {e}")
        return [0.5] * len(FEATURE_NAMES)


def load_classification_example():
    if CLASSIFICATION_FEATURE_NAMES is None:
        return [0.5] * 40
    
    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        available_features = [f for f in CLASSIFICATION_FEATURE_NAMES if f in df.columns]
        sample = df[available_features].sample(1).values[0]
        return [float(x) for x in sample]
    except Exception as e:
        print(f"Error loading classification example: {e}")
        return [0.5] * len(CLASSIFICATION_FEATURE_NAMES)


def create_interface():
    if FEATURE_NAMES is None:
        return gr.Interface(
            fn=lambda: "Model not loaded",
            inputs=[],
            outputs="text",
            title="Error: Model not loaded"
        )
    
    inputs = []
    for name in FEATURE_NAMES:
        slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label=name.replace("_", " "))
        inputs.append(slider)

    classification_inputs = []
    if CLASSIFICATION_FEATURE_NAMES is not None:
        for name in CLASSIFICATION_FEATURE_NAMES:
            slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label=name.replace("_", " "))
            classification_inputs.append(slider)

    description = """
## Deep Squat Movement Assessment

**How to use:**
1. Adjust the sliders to input deviation values (0 = no deviation, 1 = maximum deviation)
2. Click "Submit" to get your predicted score
3. Or click "Load Random Example" to test with real data

**Score Interpretation:**
- 80-100%: Excellent form
- 60-79%: Good form  
- 40-59%: Average form
- 0-39%: Needs improvement
"""

    classification_description = """
## Body Region Classification

**How to use:**
1. Adjust the sliders to input deviation values (0 = no deviation, 1 = maximum deviation)
2. Click "Predict Body Region" to identify where to focus improvements
3. Or click "Load Random Example" to test with real data

**Body Regions:** Upper Body, Lower Body
"""

    angle_features = [n for n in FEATURE_NAMES if "Angle" in n]
    nasm_features = [n for n in FEATURE_NAMES if "NASM" in n]
    time_features = [n for n in FEATURE_NAMES if "Time" in n]
    
    angle_indices = [FEATURE_NAMES.index(f) for f in angle_features]
    nasm_indices = [FEATURE_NAMES.index(f) for f in nasm_features]
    time_indices = [FEATURE_NAMES.index(f) for f in time_features]

    if CLASSIFICATION_FEATURE_NAMES is not None:
        class_angle_features = [n for n in CLASSIFICATION_FEATURE_NAMES if "Angle" in n]
        class_nasm_features = [n for n in CLASSIFICATION_FEATURE_NAMES if "NASM" in n]
        class_time_features = [n for n in CLASSIFICATION_FEATURE_NAMES if "Time" in n]
        class_angle_indices = [CLASSIFICATION_FEATURE_NAMES.index(f) for f in class_angle_features]
        class_nasm_indices = [CLASSIFICATION_FEATURE_NAMES.index(f) for f in class_nasm_features]
        class_time_indices = [CLASSIFICATION_FEATURE_NAMES.index(f) for f in class_time_features]

    with gr.Blocks(title="Deep Squat Assessment") as demo:
        gr.Markdown("# Deep Squat Movement Assessment")
        
        with gr.Tabs():
            with gr.TabItem("Movement Scoring"):
                gr.Markdown(description)

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Input Features")
                        gr.Markdown(f"*{len(FEATURE_NAMES)} features loaded from champion model*")
                        gr.Markdown("*Deviation values: 0 = perfect, 1 = maximum deviation*")

                        with gr.Tabs():
                            with gr.TabItem(f"Angle Deviations ({len(angle_indices)})"):
                                for idx in angle_indices:
                                    inputs[idx].render()

                            with gr.TabItem(f"NASM Deviations ({len(nasm_indices)})"):
                                for idx in nasm_indices:
                                    inputs[idx].render()

                            with gr.TabItem(f"Time Deviations ({len(time_indices)})"):
                                for idx in time_indices:
                                    inputs[idx].render()

                    with gr.Column(scale=1):
                        gr.Markdown("### Results")
                        score_output = gr.Textbox(label="Predicted Score")
                        interp_output = gr.Textbox(label="Assessment")
                        details_output = gr.Markdown(label="Details")

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    example_btn = gr.Button("Load Random Example")
                    clear_btn = gr.Button("Clear")

                submit_btn.click(fn=predict_score, inputs=inputs, outputs=[score_output, interp_output, details_output])
                example_btn.click(fn=load_example, inputs=[], outputs=inputs)
                clear_btn.click(
                    fn=lambda: [0.5] * len(FEATURE_NAMES) + ["", "", ""],
                    inputs=[],
                    outputs=inputs + [score_output, interp_output, details_output],
                )

            if CLASSIFICATION_FEATURE_NAMES is not None:
                with gr.TabItem("Body Region Classification"):
                    gr.Markdown(classification_description)

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### Input Features")
                            gr.Markdown(f"*{len(CLASSIFICATION_FEATURE_NAMES)} features for classification*")
                            gr.Markdown("*Deviation values: 0 = perfect, 1 = maximum deviation*")

                            with gr.Tabs():
                                with gr.TabItem(f"Angle Deviations ({len(class_angle_indices)})"):
                                    for idx in class_angle_indices:
                                        classification_inputs[idx].render()

                                with gr.TabItem(f"NASM Deviations ({len(class_nasm_indices)})"):
                                    for idx in class_nasm_indices:
                                        classification_inputs[idx].render()

                                with gr.TabItem(f"Time Deviations ({len(class_time_indices)})"):
                                    for idx in class_time_indices:
                                        classification_inputs[idx].render()

                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            class_output = gr.Textbox(label="Predicted Body Region")
                            class_interp_output = gr.Textbox(label="Confidence")
                            class_details_output = gr.Markdown(label="Details")

                    with gr.Row():
                        class_submit_btn = gr.Button("Predict Body Region", variant="primary")
                        class_example_btn = gr.Button("Load Random Example")
                        class_clear_btn = gr.Button("Clear")

                    class_submit_btn.click(fn=predict_weakest_link, inputs=classification_inputs, outputs=[class_output, class_interp_output, class_details_output])
                    class_example_btn.click(fn=load_classification_example, inputs=[], outputs=classification_inputs)
                    class_clear_btn.click(
                        fn=lambda: [0.5] * len(CLASSIFICATION_FEATURE_NAMES) + ["", "", ""],
                        inputs=[],
                        outputs=classification_inputs + [class_output, class_interp_output, class_details_output],
                    )
    
    return demo


demo = create_interface()

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
