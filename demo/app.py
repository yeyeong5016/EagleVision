import gradio as gr
from model_utils import init_detector, run_inference

# Model list (name → config, checkpoint)
model_options = {
    "EagleVision-1B (SHIPRS)": ("configs/EagleVision/EagleVision_1B-shiprsimagenet.py", "checkpoints/EagleVision-1B/SHIPRS"),
    "EagleVision-1B (MAR20)": ("configs/EagleVision/EagleVision_1B-mar20.py", "checkpoints/EagleVision-1B/MAR20"),
    "EagleVision-1B (FAIR1M)": ("configs/EagleVision/EagleVision_1B-fair1m.py", "checkpoints/EagleVision-1B/FAIR1M"),
    "EagleVision-2B (SHIPRS)": ("configs/EagleVision/EagleVision_2B-shiprsimagenet.py", "checkpoints/EagleVision-2B/SHIPRS"),
    "EagleVision-2B (MAR20)": ("configs/EagleVision/EagleVision_2B-mar20.py", "checkpoints/EagleVision-2B/MAR20"),
    "EagleVision-2B (FAIR1M)": ("configs/EagleVision/EagleVision_2B-fair1m.py", "checkpoints/EagleVision-2B/FAIR1M"),
    "EagleVision-4B (SHIPRS)": ("configs/EagleVision/EagleVision_4B-shiprsimagenet.py", "checkpoints/EagleVision-4B/SHIPRS"),
    "EagleVision-4B (MAR20)": ("configs/EagleVision/EagleVision_4B-mar20.py", "checkpoints/EagleVision-4B/MAR20"),
    "EagleVision-4B (FAIR1M)": ("configs/EagleVision/EagleVision_4B-fair1m.py", "checkpoints/EagleVision-4B/FAIR1M"),
    "EagleVision-7B (SHIPRS)": ("configs/EagleVision/EagleVision_7B-shiprsimagenet.py", "checkpoints/EagleVision-7B/SHIPRS"),
    "EagleVision-7B (MAR20)": ("configs/EagleVision/EagleVision_7B-mar20.py", "checkpoints/EagleVision-7B/MAR20"),
    "EagleVision-7B (FAIR1M)": ("configs/EagleVision/EagleVision_7B-fair1m.py", "checkpoints/EagleVision-7B/FAIR1M"),
}

DEVICE = 'cuda:0'

def inference_demo(image, with_attr, score_thr, model_name):
    # Load selected model
    config_path, checkpoint_path = model_options[model_name]
    model = init_detector(config_path, checkpoint_path, device=DEVICE)

    # Save uploaded image
    image_path = 'demo/temp_input.jpg'
    image.save(image_path)

    # Run inference
    result_path, attr_text = run_inference(model, image_path, with_attribute=with_attr, score_thr=score_thr)
    return result_path, attr_text

title = "🚀 EagleVision Demo (Multi-Model + Object Detection + Object Attributes Understanding)"
description = """
🎯 Upload an image to detect rotated objects with the model of your choice.<br>
🧠 Optionally enable objcet attribute understanding if supported.<br>
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(model_options.keys()),
                value=list(model_options.keys())[0],
                label="🧩 Select Model"
            )
            image_input = gr.Image(type="pil", label="📤 Upload Image")
            attr_checkbox = gr.Checkbox(label="🧠 Enable Object Attributes Understanding")
            score_slider = gr.Slider(0, 1, value=0.4, step=0.05, label="🎯 Score Threshold")
            submit_btn = gr.Button("🚀 Run Detection")
        with gr.Column(scale=2):
            image_output = gr.Image(type="filepath", label="📸 Detection Result")
            attr_output = gr.Textbox(label="🧾 Attribute Result", lines=5)

    submit_btn.click(
        fn=inference_demo,
        inputs=[image_input, attr_checkbox, score_slider, model_selector],
        outputs=[image_output, attr_output]
    )

demo.launch()
