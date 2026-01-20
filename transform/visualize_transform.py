import pandas as pd
import sys 

from transform.transform_functions import functions_dict
from config import *
from vit_explainability.vit_visualization_explainability import get_vit_explainability_model, visualize_attention_explainability
from vit_baseline.visualization_attention import get_vit_basic_model, visualize_attention_consecutive

def visualize_transform_basic(image_path, target_diameter, transform_fn=None, exist_origin=True):
    if exist_origin:
        transformed_path = transform_fn(image_path)
        model, transform = get_vit_basic_model()
        visualize_attention_consecutive(model, image_path, transform, target_diameter)
        visualize_attention_consecutive(model, transformed_path, transform, target_diameter)
    else:
        model, transform = get_vit_basic_model()
        visualize_attention_consecutive(model, image_path, transform, target_diameter)

def visualize_transform_explainability(image_path, target_diameter, transform_fn=None, exist_origin=True):
    if exist_origin:
        transformed_path = transform_fn(image_path)

        model, transform = get_vit_explainability_model()
        visualize_attention_explainability(model, image_path, transform, target_diameter)
        visualize_attention_explainability(model, transformed_path, transform, target_diameter)
    else: 
        model, transform = get_vit_explainability_model()
        visualize_attention_explainability(model, image_path, transform, target_diameter)

if __name__ == "__main__":
    # transform 시각화 
    # df = pd.read_csv(LABEL_FILE)
    # sample_row = df.iloc[:3]

    # # 인자로 함수 입력 받아서 사용
    # attention_type = sys.argv[1]
    # function_name = sys.argv[2]

    # for _, row in sample_row.iterrows():
    #     if attention_type == "1":
    #         visualize_transform_basic(DATA_DIR / row['filename'], row['diameter'], functions_dict[function_name])
    #     elif attention_type == "2":
    #         visualize_transform_explainability(DATA_DIR / row['filename'], row['diameter'], functions_dict[function_name])
    
    # circle 시각화 
    # for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     transformed_path, diameter = functions_dict["circle"](r)
    #     visualize_transform(transformed_path, diameter, exist_origin=False)

    # diverse_shapes 시각화 
    transformed_path = functions_dict["two_circles"]([30, 40, 50, 60, 70, 80, 90], 10)
    for path, diameter in transformed_path:
        visualize_transform_explainability(path, diameter, exist_origin=False)
