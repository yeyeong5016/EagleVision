from ...smp import *
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'

system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""

# time_system_prompt = """As an AI assistant, your task is to evaluate multiple candidate answers in comparison to a given correct answer. \
# The question itself, the correct 'groundtruth' answer, and the candidate answers will be provided to you.
# Your assessment should range from 0 to 3, based on the semantic similarity between the groundtruth and the candidate answers, \
# disregarding any grammatical differences.
# A rating of 0 suggests no similarity, implying all the candidate answers are entirely incorrect.
# A rating of 1 suggests low similarity, meaning all the candidate answers are largely incorrect.
# A rating of 2 suggests high similarity, meaning there is at least a candidate answer largely correct.
# Lastly, a rating of 3 indicates complete similarity, which means that there is at least a candidate answer entirely correct.
# Your response should be a single integer from 0, 1, 2, or 3.
# (Note: You should focus on the semantics of the content, and you cannot rate "3" just because the video time is the same)"""

time_system_prompt = '''
You are an assistant skilled at evaluating the quality of AI conversation. Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant. 

You'll need to assess the response based on the Answer Accuracy. We will provide you with a user question (or not) and the AI model's multiple responses and a reference answer for your evaluation. As you begin your assessment, follow this process:

1. Evaluate the AI model's answers on the given scoring rules and provide an score of 1 to 10 for each of the multiple responses from the AI model.
2. Select the highest score as the overall score.
3. Your scoring should be as stringent as possible and follow the rules below:

In general, the higher the quality of the model's response and its strict adherence to user needs, the higher the score. Responses that do not meet user needs will receive lower scores.

Scoring rules:
Scores 1-2 when the answer is significantly inconsistent with the groundtruth (or question) or contains obvious errors.
Scores 3-4 when the answer is partially correct but contains some errors or is incomplete.
Scores 5-6 when the answer is basically correct but lacks details or is not sufficiently detailed.
Scores 7-8 when the answer is accurate and detailed, fully corresponding to the question (or groundtruth).
Scores 9-10 when the answer is not only accurate and detailed but also provides additional useful information, exceeding expectations.

Please remember, you should focus on the semantics of the content, and you cannot rate a high score just because the video time is the same. 
Finally, your response must satisfy the following format:
{"Overall Score": , "Explanation": ""}
'''

MMV_DIMENSIONS = {
    'CP': ['Video Topic', 'Video Emotion', 'Video Scene', 'Video Style'],
    'FP-S': ['OCR', 'Object Recognition', 'Attribute Recognition', 'Event Recognition', 'Human Motion', 'Counting'],
    'FP-C': ['Spatial Relationship', 'Human-object Interaction', 'Human Interaction'],
    'HL': ['Hallucination'],
    'LR': ['Structuralized Image-Text Understanding', 'Mathematical Calculation'],
    'AR': ['Physical Property', 'Function Reasoning', 'Identity Reasoning'],
    'RR': ['Natural Relation', 'Physical Relation', 'Social Relation'],
    'CSR': ['Common Sense Reasoning'],
    'TR': ['Counterfactual Reasoning', 'Causal Reasoning', 'Future Prediction'],
}
L3_DIMS = []
for k, v in MMV_DIMENSIONS.items():
    L3_DIMS.extend(v)

MMV_DIMENSIONS['Perception'] = []
MMV_DIMENSIONS['Reasoning'] = []
MMV_DIMENSIONS['Overall'] = []
for k in ['CP', 'FP-C', 'FP-S', 'HL']:
    MMV_DIMENSIONS['Perception'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])
for k in ['LR', 'AR', 'RR', 'CSR', 'TR']:
    MMV_DIMENSIONS['Reasoning'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])


def get_dimension_rating(data_path):
    data = load(data_path)
    coarse_rating = {k: [] for k in MMV_DIMENSIONS}
    fine_rating = {k: [] for k in L3_DIMS}

    for i in range(len(data)):
        cate = data.iloc[i]['dimensions']
        cates = eval(cate)

        for c in cates:
            fine_rating[c].append(data.iloc[i]['score'])

        for d in MMV_DIMENSIONS:
            if np.any([x in MMV_DIMENSIONS[d] for x in cates]):
                coarse_rating[d].append(data.iloc[i]['score'])

    coarse_all = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in coarse_rating.items()}
    coarse_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in coarse_rating.items()}
    fine_all = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in fine_rating.items()}
    fine_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in fine_rating.items()}
    return dict(coarse_all=coarse_all, coarse_valid=coarse_valid, fine_all=fine_all, fine_valid=fine_valid)


def build_prompt(item):
    tmpl = '\nQuestion: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '
    return tmpl.format(item['question'], item['answer'], item['prediction'])
