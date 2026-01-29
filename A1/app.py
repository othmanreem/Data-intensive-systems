import gradio as gr

similar_app = "https://huggingface.co/spaces/sems/pose-think"

def greet(name):
    return "Hello " + name + "!!"

description_string = """
Simple Demo app that should be developed into a Movement Assessment similar to the [posture analysis](https://huggingface.co/spaces/sems/pose-think).
"""
iface = gr.Interface(fn=greet, inputs="text", outputs="text", description=description_string)
iface.launch()
