from PIL import Image
import gradio as gr
from controlnet_aux import OpenposeDetector
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any

# Load OpenPose detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

def extract_joint_positions(openpose_result) -> Dict[str, Any]:
    """Extract joint positions from OpenPose result."""
    # OpenPose returns a PIL Image with encoded pose data
    # We need to access the pose data from the result
    if hasattr(openpose_result, 'pose_keypoints_2d'):
        # If OpenPose returns structured data
        return {
            "keypoints": openpose_result.pose_keypoints_2d,
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Fallback: extract from image if possible
        return {
            "keypoints": [],
            "timestamp": datetime.now().isoformat(),
            "note": "No structured pose data available"
        }

def save_to_csv(joint_data: Dict[str, Any], filename: str = None) -> str:
    """Save joint positions to CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_data_{timestamp}.csv"

    filepath = os.path.join("pose_outputs", filename)
    os.makedirs("pose_outputs", exist_ok=True)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Joint", "X", "Y", "Confidence"])

        keypoints = joint_data.get("keypoints", [])
        if keypoints and isinstance(keypoints, list):
            # OpenPose format: [x1, y1, c1, x2, y2, c2, ...]
            joint_names = [
                "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                "LHeel", "RBigToe", "RSmallToe", "RHeel"
            ]

            for i in range(0, len(keypoints), 3):
                if i + 2 < len(keypoints):
                    joint_idx = i // 3
                    joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"Joint_{joint_idx}"
                    writer.writerow([
                        joint_name,
                        keypoints[i],      # X
                        keypoints[i + 1],  # Y
                        keypoints[i + 2]   # Confidence
                    ])

        writer.writerow(["Timestamp", joint_data.get("timestamp", "")])

    return filepath

def save_to_json(joint_data: Dict[str, Any], filename: str = None) -> str:
    """Save joint positions to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_data_{timestamp}.json"

    filepath = os.path.join("pose_outputs", filename)
    os.makedirs("pose_outputs", exist_ok=True)

    with open(filepath, 'w') as jsonfile:
        json.dump(joint_data, jsonfile, indent=2)

    return filepath

def generate_pose(image, use_openpose=True, save_outputs=True):
    img = image.convert("RGB")
    if use_openpose:
        result = openpose(img)
    else:
        result = img
    if not isinstance(result, Image.Image):
        result = Image.fromarray(result)

    # Extract and save pose data if OpenPose was used
    joint_data = {}
    if use_openpose and save_outputs:
        joint_data = extract_joint_positions(result)
        csv_path = save_to_csv(joint_data)
        json_path = save_to_json(joint_data)
        joint_data["csv_path"] = csv_path
        joint_data["json_path"] = json_path

    return result, joint_data

# Gradio UI with pose data outputs
def format_pose_output(joint_data: Dict[str, Any]) -> str:
    """Format pose data for display."""
    if not joint_data or not joint_data.get("keypoints"):
        return "No pose data available."

    output = "### Joint Positions\n\n"
    output += f"**Timestamp:** {joint_data.get('timestamp', 'N/A')}\n\n"

    keypoints = joint_data.get("keypoints", [])
    if keypoints and isinstance(keypoints, list):
        joint_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
            "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
            "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
            "LHeel", "RBigToe", "RSmallToe", "RHeel"
        ]

        output += "| Joint | X | Y | Confidence |\n"
        output += "|-------|---|---|------------|\n"

        for i in range(0, min(len(keypoints), 72), 3):
            if i + 2 < len(keypoints):
                joint_idx = i // 3
                joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"Joint_{joint_idx}"
                x = keypoints[i]
                y = keypoints[i + 1]
                confidence = keypoints[i + 2]
                output += f"| {joint_name} | {x:.1f} | {y:.1f} | {confidence:.3f} |\n"

    output += f"\n**CSV File:** `{joint_data.get('csv_path', 'N/A')}`\n"
    output += f"**JSON File:** `{joint_data.get('json_path', 'N/A')}`\n"

    return output

def process_and_display(image, use_openpose=True):
    """Process image and return pose output with data files."""
    result, joint_data = generate_pose(image, use_openpose, save_outputs=True)

    if use_openpose and joint_data:
        pose_info = format_pose_output(joint_data)
        return result, pose_info
    else:
        return result, "Pose data extraction skipped."

# Gradio UI with pose data outputs
demo = gr.Interface(
    fn=process_and_display,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Checkbox(value=True, label="Use OpenPose (default: true)"),
    ],
    outputs=[
        gr.Image(type="pil", label="Pose Output"),
        gr.Textbox(label="Pose Data", lines=10)
    ],
    title="OpenPose Pose Generator",
    description="Generate full body pose including face and hands. Extracts and stores joint positions in CSV and JSON formats."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
