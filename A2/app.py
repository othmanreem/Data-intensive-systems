import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os

MODEL_PATH = "models/champion_model_final_2.pkl"
DATA_PATH = "A2_dataset.csv"

model = None
FEATURE_NAMES = None
MODEL_METRICS = None


def load_champion_model():
    global model, FEATURE_NAMES, MODEL_METRICS
    
    possible_paths = [
        MODEL_PATH,
        "A2/models/champion_model_final_2.pkl",
        "../A2/models/champion_model_final_2.pkl",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading champion model from {path}")
            with open(path, "rb") as f:
                artifact = pickle.load(f)
            
            model = artifact["model"]
            FEATURE_NAMES = artifact["feature_columns"]
            MODEL_METRICS = artifact.get("test_metrics", {})
            
            print(f"model loaded successfully")
            print(f"Features: {len(FEATURE_NAMES)} columns")
            print(f"Test R2: {MODEL_METRICS.get('r2', 'N/A')}")
            return True
    
    print("champion model not found")
    return False


load_champion_model()


# prediction function
def predict_score(*feature_values):
    if model is None:
        return "Error", "Model not loaded"
    
    # Convert inputs to dataframe with correct feature names
    features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    
    raw_score = model.predict(features_df)[0]

    # score to valid range and change to %
    score = max(0, min(1, raw_score)) * 100

    if score >= 80:
        interpretation = "Excellent, great squat form"
    elif score >= 60:
        interpretation = "Good, minor improvements needed"
    elif score >= 40:
        interpretation = "Average, a lot of areas to work on"
    else:
        interpretation = "Needs work, focus on proper form"

    # Create output
    r2 = MODEL_METRICS.get('r2', 'N/A')
    correlation = MODEL_METRICS.get('correlation', 'N/A')
    
    # Format metrics
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


# load example for tesitng
def load_example():
    if FEATURE_NAMES is None:
        return [0.5] * 35
    
    try:
        possible_paths = [
            DATA_PATH,
            "A2/A2_dataset.csv",
            "../A2/A2_dataset.csv",
            "../Datasets_all/A2_dataset_80.csv",
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break

        # Get a random row with only the features we need
        available_features = [f for f in FEATURE_NAMES if f in df.columns]
        sample = df[available_features].sample(1).values[0]
        # Convert to float list to ensure proper types for Gradio sliders
        return [float(x) for x in sample]
    except Exception as e:
        print(f"Error loading example: {e}")
        return [0.5] * len(FEATURE_NAMES)


# create gradio interface
def create_interface():
    if FEATURE_NAMES is None:
        return gr.Interface(
            fn=lambda: "Model not loaded",
            inputs=[],
            outputs="text",
            title="Error: Model not loaded"
        )
    
    # Create input sliders for features
    inputs = []
    for name in FEATURE_NAMES:
        slider = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.01,
            label=name.replace("_", " "),
        )
        inputs.append(slider)

    # Build the interface
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

    # features into categories
    angle_features = [n for n in FEATURE_NAMES if "Angle" in n]
    nasm_features = [n for n in FEATURE_NAMES if "NASM" in n]
    time_features = [n for n in FEATURE_NAMES if "Time" in n]
    
    # Get indices for each category
    angle_indices = [FEATURE_NAMES.index(f) for f in angle_features]
    nasm_indices = [FEATURE_NAMES.index(f) for f in nasm_features]
    time_indices = [FEATURE_NAMES.index(f) for f in time_features]

    # Create the main interface
    with gr.Blocks(title="Deep Squat Assessment") as demo:
        gr.Markdown("# Deep Squat Movement Assessment")
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

        submit_btn.click(
            fn=predict_score,
            inputs=inputs,
            outputs=[score_output, interp_output, details_output],
        )

        example_btn.click(
            fn=load_example, 
            inputs=[], 
            outputs=inputs
        )

        clear_btn.click(
            fn=lambda: [0.5] * len(FEATURE_NAMES) + ["", "", ""],
            inputs=[],
            outputs=inputs + [score_output, interp_output, details_output],
        )
    
    return demo


# Create the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
