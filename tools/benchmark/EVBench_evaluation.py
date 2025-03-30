from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np
import os
from openai import OpenAI
import argparse
import json
from vlmeval.utils import track_progress_rich
from vlmeval.smp import load
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Task2 using GPT')
    parser.add_argument('--root-dir', default='work_dirs/EagleVision_1B-shiprsimagenet/val',
                      help='Root directory containing evaluation data')
    parser.add_argument('--openai-key', required=True,
                      help='OpenAI API key')
    parser.add_argument('--predefined-attributes', default='SHIPRS',
                      choices=['SHIPRS', 'MAR20', 'FAIR1M'],
                      help='Type of predefined attributes to use')
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Initialize OpenAI client
client = OpenAI(api_key=args.openai_key)

def get_predefined_attributes(attr_type):
    # SHIPRS-related attributes
    SHIPRS_attributes = [
        "ship-visibility", "ship-purpose", "ship-motion", "ship-capacity", "ship-load-status", "ship-cargo-status", "ship-mooring-status",
        "hull-color", "hull-size", "hull-shadow", "hull-outline",
        "superstructure-color", "superstructure-size", "superstructure-height", "superstructure-position", "paint-condition", 
        "bow-design", "stern-design", "deck-utilization", "deck-condition", "deck-obstacles", "deck-color", "deck-structure", 
        "deck-accessories", "passenger-facilities", "container-presence", "container-count", "container-color", "container-layout", 
        "container-alignment", "container-densities", "container-type", "machinery-presence", "location", "weather-condition", 
        "water-color", "water-turbulence", "unique-attributes"
    ]
    
    # MAR20-related attributes
    MAR20_attributes = [
        'engine-color', 'engine-location', 'engine-size', 'engine-type', 'engines-number', 
        'engines-shape', 'engines-visible', 'fuselage-color', 'fuselage-length', 'fuselage-material', 
        'fuselage-shape', 'nose-cone-color', 'propeller-count', 'tail-color', 'tail-height', 
        'tail-material', 'tail-shape', 'tail-type', 'wings-angle', 'wings-color', 
        'wings-material', 'wings-shape', 'wings-span', 'wings-type'
    ]
    
    if attr_type == 'SHIPRS':
        return SHIPRS_attributes
    elif attr_type == 'MAR20':
        return MAR20_attributes
    else:  # 'FAI1R1M'
        return SHIPRS_attributes + MAR20_attributes

# Template for GPT evaluation
DATA_Template = """
[Instruction]
This image includes a remote sensing object in a bird's-eye view. Please help me to explain the visual content of this object in a fine-grained manner.\n
[Reference Answer]
{}\n
[Assistant's Final Answer]
{}\n
"""

# System message for GPT evaluation
sys_message = """
    Please act as an impartial judge and evaluate a multimodal AI assistant's performance in fine-grained attribute understanding. Each data sample includes:

    [Instruction]
    {Instruction}\n
    [Reference Answer]
    {Reference Answer}\n
    [Assistant's Final Answer]
    {Assistant's Final Answer}\n

    **Evaluation Criteria:**

    1. **Correctness:**  
    - **Description:** Ensure that the attribute value in the assistant’s answer is correct and aligns with the reference answer in meaning, allowing for reasonable variations in expression.
    - **Guidelines:**  
        - **Accurate:** The attribute value matches the reference answer with minimal errors or reasonable variation in phrasing.
        - **Moderate:** The attribute value is misaligned with the reference answer in meaning, or incomplete.
        - **Inaccurate:** The attribute value is largely incorrect or misleading, with significant deviation from the reference answer.
    - **Notes:**  
        - The assistant’s answer should match the reference answer in meaning, even if the wording is different. For example, phrases like "none", "not visible", "invisible", "minimal", and similar expressions can be considered equivalent if they convey the same underlying concept of absence or near-absence.
        - If the reference answer implies an absence or a near-absence (e.g., "none", "minimal", "slight"), the assistant’s interpretation should be flexible enough to accommodate slight differences in wording as long as the intended meaning remains clear.
        - The assistant can use different terminology to describe the same concept, but if the change in phrasing distorts the original meaning or causes ambiguity, it should be flagged as incorrect.
        - Avoid penalizing the assistant for using reasonable variants unless it leads to misunderstanding or over-complication of the reference meaning.
        - If an attribute is omitted or missing in the assistant's answer, assess whether the absence can be logically inferred or if it is critical to the answer.
    

    2. **Expressiveness:**  
    - **Description:** Assess whether each attribute's value sufficiently conveys the necessary information, matching the level of detail required by the reference answer.
    - **Guidelines:**  
        - **Adequate:** The value is clear and conveys the required information effectively, whether long, short or different variants.
        - **Insufficient:** The value is vague, too brief, or fails to clearly express the necessary information.


    **Scoring Guidelines:**

    - **Scale:** 1 to 5
    - **5:** Excellent  
        - The attribute value is highly accurate or with alternative phrasing, match the reference answer in meaning, and clearly convey the required information, even if the phrasing differs slightly.
    
    - **4:** Good  
        - The attribute value is mostly correct, with minor discrepancies in meaning, and still convey the necessary information effectively.
    
    - **3:** Satisfactory  
        - The attribute value is with noticeable errors or unclear expression that weakens the conveyed meaning.
    
    - **2:** Needs Improvement  
        - The attribute value is mostly incorrect, or incomplete, and fail to clearly convey the similar information in reference answer.
    
    - **1:** Poor  
        - The attribute value is completely incorrect or misleading, cannot be understood as a variant of the reference answer, and fails to provide meaningful or necessary information.

    **Instructions:**

    1. **Assess Correctness, and Expressiveness:** Assess Correctness and Expressiveness for each attribute based on the criteria above.
    2. **Attribute Score:** Provide a score for each attribute (1 to 5).
    3. **Explanation:** Briefly justify the score for each attribute, especially if there are reasonable variations in phrasing or missing attributes that can be inferred.

    **Output Format:**

    [Explanation] {Your evaluation}

    [attribute_name 1] {1-5}
    
    [attribute_name 2] {1-5}
    
    ...\n


    """

def gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                'role': 'system',
                'content': sys_message
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    return response

def main():
    
    # Get predefined attributes based on type
    predefined_attributes = get_predefined_attributes(args.predefined_attributes)
    
    # Set up file paths
    eval_json = os.path.join(args.root_dir, "Task2.json")
    tmp_file = os.path.join(args.root_dir, "Task2.pkl")
    
    # Load evaluation data
    with open(eval_json, 'r') as f:
        eval_datas = json.load(f)
        
    # Prepare prompts and indices for evaluation
    prompts = []
    indices = []
    
    remain_num = 0
    object_num = 0
    for eval_data in tqdm(eval_datas):
        objs = eval_data["objs"]
        object_num += len(objs)
        for obj in objs:
            gts = obj["gt_caption"]
            if "pred_caption" in obj:
                remain_num += 1
                preds = obj["pred_caption"]
                prompt = DATA_Template.format(gts, preds)
                prompts.append(prompt)
                indices.append("{}_{}".format(eval_data["img_id"], obj["obj_id"]))

    
    _ = track_progress_rich(
        gpt,
        prompts,
        keys=indices,
        save=tmp_file,
        nproc=8,
        chunksize=8
        )

    results = load(tmp_file)
    
    pattern = [r'\[(.*?)\]\s(\d)', 
        r'\[(.*?)\]\s\*\*(\d)\*\*',
        r'\[(.*?)\]\s\{(\d)\}',
        r'\*\*(.*?)\:\*\*\s(\d)',
        r"([a-zA-Z\-]+):\s*(\d+)",
        r"\[([^\]]+)\].*?Score: (\d+)",
               r"\[([^\]]+)\][\s\S]*?(\d+)\s*" ]   

    # Initialize score dictionary
    EV_Task2_score = defaultdict(list)
    scores = []
    for k, v in results.items():
        matched = False
        for i in pattern:
            matches = re.findall(i, v, re.DOTALL)
            if matches:
                matched = True
                score = {match[0]: int(match[1]) for match in matches}
                for match in matches:
                    true_attr_name = match[0].lower()
                    if true_attr_name in predefined_attributes:
                        EV_Task2_score[true_attr_name].append(int(match[1])) 
                scores.append(score)
            if matched:
                break
        if not matched:
            print(v)
            
    # Calculate final scores
    EV_Task2_score = {k: sum(v) * 20 / len(v) for k, v in EV_Task2_score.items()}
    
    # Print results
    max_key_length = max(len(key) for key in EV_Task2_score.keys())
    for key in predefined_attributes:
        print(f"{key:<{max_key_length}} | {EV_Task2_score[key]:.2f}")

    # Calculate and print mean score
    mean_value = np.mean(list(EV_Task2_score.values()))
    print(f"{'mean_score':<{max_key_length}} | {mean_value:.2f}")
    
    
    recall = remain_num / object_num * 100
    print(f"{'recall':<{max_key_length}} | {recall:.2f}")

if __name__ == '__main__':
    main()
