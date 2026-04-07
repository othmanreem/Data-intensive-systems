from PIL import Image
import gradio as gr
from controlnet_aux import OpenposeDetector
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Load OpenPose detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# OpenPose joint mapping (COCO format - 18 joints)
JOINT_NAMES = [
    "Nose",           # 0
    "Neck",           # 1
    "RShoulder",      # 2
    "RElbow",         # 3
    "RWrist",         # 4
    "LShoulder",      # 5
    "LElbow",         # 6
    "LWrist",         # 7
    "RHip",           # 8
    "RKnee",          # 9
    "RAnkle",         # 10
    "LHip",           # 11
    "LKnee",          # 12
    "LAnkle",         # 13
    "REye",           # 14
    "LEye",           # 15
    "REar",           # 16
    "LEar"            # 17
]

def extract_joint_positions_from_detect_poses(pose_results: List[Any]) -> Dict[str, Any]:
    """Extract joint positions from OpenPose detect_poses result."""
    all_poses = []

    for idx, pose in enumerate(pose_results):
        body = pose.body
        keypoints = []

        for joint_idx, keypoint in enumerate(body.keypoints):
            if keypoint is not None:
                keypoints.append({
                    "x": keypoint.x,
                    "y": keypoint.y,
                    "score": getattr(keypoint, 'score', 0.0),
                    "name": JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"Joint_{joint_idx}"
                })
            else:
                keypoints.append({
                    "x": None,
                    "y": None,
                    "score": None,
                    "name": JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"Joint_{joint_idx}"
                })

        all_poses.append({
            "pose_id": idx,
            "total_score": body.total_score,
            "total_parts": body.total_parts,
            "keypoints": keypoints
        })

    return {
        "poses": all_poses,
        "timestamp": datetime.now().isoformat(),
        "joint_names": JOINT_NAMES
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
        writer.writerow(["Pose_ID", "Joint", "X", "Y", "Confidence", "Visible"])

        poses = joint_data.get("poses", [])
        for pose in poses:
            pose_id = pose.get("pose_id", 0)
            for kp in pose.get("keypoints", []):
                x = kp.get("x")
                y = kp.get("y")
                score = kp.get("score")
                name = kp.get("name", "Unknown")

                visible = "Yes" if x is not None and y is not None else "No"

                writer.writerow([
                    pose_id,
                    name,
                    f"{x:.2f}" if x is not None else "N/A",
                    f"{y:.2f}" if y is not None else "N/A",
                    f"{score:.3f}" if score is not None else "N/A",
                    visible
                ])

        writer.writerow([])
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

def generate_pose(image, use_openpose=True, save_outputs=True, include_hands=False, include_face=False):
    """Generate pose estimation and extract joint positions."""
    img = image.convert("RGB")

    if use_openpose:
        # Convert PIL Image to numpy array for detect_poses
        img_array = np.array(img)

        # Use detect_poses to get structured data
        pose_results = openpose.detect_poses(
            img_array,
            include_hand=include_hands,
            include_face=include_face
        )

        # Extract joint positions from pose results
        joint_data = extract_joint_positions_from_detect_poses(pose_results)

        # Generate the annotated image
        result = openpose(img)

        # Save pose data if requested
        if save_outputs:
            csv_path = save_to_csv(joint_data)
            json_path = save_to_json(joint_data)
            joint_data["csv_path"] = csv_path
            joint_data["json_path"] = json_path
    else:
        result = img
        joint_data = {
            "poses": [],
            "timestamp": datetime.now().isoformat(),
            "note": "OpenPose disabled - no pose data extracted"
        }

    if not isinstance(result, Image.Image):
        result = Image.fromarray(result)

    return result, joint_data

def format_pose_output(joint_data: Dict[str, Any]) -> str:
    """Format pose data for display in Gradio."""
    if not joint_data.get("poses"):
        return "No pose data available.\n\n" + \
               f"**Timestamp:** {joint_data.get('timestamp', 'N/A')}\n" + \
               f"**CSV File:** `{joint_data.get('csv_path', 'N/A')}`\n" + \
               f"**JSON File:** `{joint_data.get('json_path', 'N/A')}`"

    output = "### Detected Poses\n\n"
    output += f"**Timestamp:** {joint_data.get('timestamp', 'N/A')}\n\n"

    for pose in joint_data.get("poses", []):
        output += f"#### Pose #{pose.get('pose_id', 0)}\n"
        output += f"- **Total Score:** {pose.get('total_score', 0):.3f}\n"
        output += f"- **Total Parts:** {pose.get('total_parts', 0)}\n\n"

        output += "| Joint | X | Y | Confidence | Visible |\n"
        output += "|-------|---|---|------------|---------|\n"

        for kp in pose.get("keypoints", []):
            name = kp.get("name", "Unknown")
            x = kp.get("x")
            y = kp.get("y")
            score = kp.get("score")

            x_str = f"{x:.1f}" if x is not None else "N/A"
            y_str = f"{y:.1f}" if y is not None else "N/A"
            score_str = f"{score:.3f}" if score is not None else "N/A"
            visible = "Yes" if x is not None and y is not None else "No"

            output += f"| {name} | {x_str} | {y_str} | {score_str} | {visible} |\n"

        output += "\n"

    output += f"**CSV File:** `{joint_data.get('csv_path', 'N/A')}`\n"
    output += f"**JSON File:** `{joint_data.get('json_path', 'N/A')}`\n"

    return output

def process_and_display(image, use_openpose=True, include_hands=False, include_face=False):
    """Process image and return pose output with data files."""
    result, joint_data = generate_pose(
        image,
        use_openpose=use_openpose,
        save_outputs=True,
        include_hands=include_hands,
        include_face=include_face
    )

    pose_info = format_pose_output(joint_data)
    return result, pose_info

# Gradio UI
demo = gr.Interface(
    fn=process_and_display,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Checkbox(value=True, label="Use OpenPose (default: true)"),
        gr.Checkbox(value=False, label="Include Hands"),
        gr.Checkbox(value=False, label="Include Face"),
    ],
    outputs=[
        gr.Image(type="pil", label="Pose Output"),
        gr.Textbox(label="Pose Data", lines=15)
    ],
    title="Pose Estimation and Export",
    description="Generate full body pose including face and hands. Extracts and stores joint positions in CSV and JSON formats."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
