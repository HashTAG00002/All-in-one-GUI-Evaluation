import numpy as np
import json
from collections import defaultdict
import argparse
import re
import math
import os
import logging
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 模型输出的对应关系，按需添加你需要的内容
TYPE_MAPPING = {
    'click': 'CLICK',
    'long_press': 'LONG_PRESS',

    'input': 'TYPE',
    'type': 'TYPE',
    'text': 'TYPE',
    'write': 'TYPE',
    'input_text': 'TYPE',

    'scroll': 'SCROLL',

    'press_home': 'PRESS_HOME',
    'press_back': 'PRESS_BACK',
    'press_appselect': 'PRESS_APPSELECT',

    'open_app': 'OPEN_APP',
    
    'wait': 'WAIT',
    'idle': 'WAIT',

    'finished': 'COMPLETE',
    'completed': 'COMPLETE',

    'call_user': 'INCOMPLETE',
    'incompleted': 'INCOMPLETE',
}

CLICK_LP_KWARGS_MAPPING = {
    'start_box': 'point',
    'point': 'point',
}

# for type
def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# for click & long_press
def calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# for scroll
def calculate_scroll_angle(start_x, start_y, end_x, end_y):
    """
    计算滚动方向角度（相对于水平向右方向，逆时针为正）
    返回角度范围：[-180, 180]
    """
    dx = end_x - start_x
    dy = end_y - start_y
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def angle_difference(angle1, angle2):
    """
    计算两个角度之间的最小差值（考虑周期性）
    返回范围：[0, 180]
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def myload(predicted_path: str, gt_path: str, output_path: str) -> List[Tuple]:
    """
    加载预测文件和真实标签文件，进行数据匹配和整合
    Args:
        predicted_path: 预测结果文件路径（json或jsonl）
        gt_path: 真实标签文件路径（json或jsonl）
    
    Returns:
        List[Tuple]: 包含(predicted_response, gt_response)的元组列表
    """
    
    def load_file(file_path: str):
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    gt_data = load_file(gt_path)
    pred_data = load_file(predicted_path)
    gt_dict = {gt_item['images'][-1]: gt_item for gt_item in gt_data}
    result_tuples = []
    matched_count = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred_item in tqdm(pred_data, desc='Processing json pairs...'):
            # ["image_path"]/["images"][0]/["images"][0]['path']三选一即可
            pred_image_path = pred_item.get('image_path', None)
            if pred_image_path is None:
                pred_image_path = pred_item['images'][-1].get('path', pred_item['images'][-1])
            if pred_image_path not in gt_dict:
                print(f"Warning: No matching GT found for image path: {pred_image_path}")
                continue
            gt_item = gt_dict[pred_image_path]
            gt_response = gt_item['ground_truth']
            pred_response = pred_item.get('response', pred_item.get('pred'))
            merged_item = {
                "episode_id": gt_item.get("episode_id", ""),
                "step": gt_item.get("step", ""),
                "messages": gt_item['messages'],
                "gt_response": gt_response,
                "predicted_response": pred_response,
                "images": gt_item.get("images", [])
            }
            json.dump(merged_item, f, ensure_ascii=False, indent=2)
            result_tuples.append((pred_response, gt_response))
            matched_count += 1

    print(f"Total matched items: {matched_count}")
    print(f"Total result tuples: {len(result_tuples)}")
    return result_tuples

        

def action_parser(response: str) -> Tuple[str, Dict[str, Any]]:
    """
    从assistant response中解析动作信息
    
    Args:
        response: 包含动作信息的完整响应字符串
        
    Returns:
        Tuple[str, Dict]: (action_type, action_kwargs)
        - action_type: 动作类型字符串
        - action_kwargs: 动作参数字典
    """
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL)
    if action_match:
        action_content = action_match.group(1).strip()
    else:
        # this is for jinchao style response
        action_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if action_match:
            action_content = action_match.group(1).strip()
        # this is for ui-tars style response
        elif response.startswith('Action:'): 
            action_content = response.replace('Action:','',1).strip()
        else:
            raise ValueError("No action found in response")
    action_content = re.sub(r'\s+', ' ', action_content)
    action_type = action_content.split('(')[0].lower() # ALL LOWER!
    action_kwargs = {}
    if action_type not in TYPE_MAPPING:
        return '', action_kwargs
    else:
        action_type = TYPE_MAPPING[action_type]

    no_arg_actions = {'COMPLETE', 'INCOMPLETE', 'PRESS_HOME', 'PRESS_BACK', 'PRESS_APPSELECT', 'WAIT'}
    if action_type in no_arg_actions:
        return action_type, action_kwargs
    
    # 处理有参数的动作
    action_kwargs_str = action_content[len(action_type)+1:-1].replace('<|box_start|>','').replace('<|box_end|>','')
    # 匹配 click(start_box='(x,y)') CLICK(point='[x, y]')
    if action_type in {'CLICK', 'LONG_PRESS'}:
        action_kwargs_str = action_kwargs_str.replace('[','(',1).replace(']',')',1)
        for old_arg_name, new_arg_name in CLICK_LP_KWARGS_MAPPING.items():
            action_kwargs_str = action_kwargs_str.replace(old_arg_name, new_arg_name, 1)
        click_match = re.match(r"point\s*=\s*'\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*'\s*", action_kwargs_str)
        if not click_match:
            return '', action_kwargs
        action_kwargs["point"] = (int(click_match.group(1)), int(click_match.group(2)))
        return action_type, action_kwargs
    
    # 匹配 type(content='xxx')
    elif action_type in {'TYPE', 'OPEN_APP'}:
        type_match = re.match(r"content\s*=\s*'(.*)'\s*$", action_kwargs_str)
        if not type_match:
            return '', action_kwargs
        content = type_match.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
        action_kwargs["content"] = content
        return action_type, action_kwargs
    
    # 匹配 scroll - 处理两种可能的格式
    # 格式1: scroll(start_box='(x1,y1)', end_box='(x2,y2)')
    # 格式2: scroll(start_box='(x1,y1)', end_box='(x2,y2')  # 注意这里右括号在引号外
    elif action_type == 'SCROLL':
        # case1: start & end point
        if 'start_box' in action_kwargs_str and 'end_box' in action_kwargs_str:
            scroll_pattern = r"start_box\s*=\s*'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)'\s*,\s*end_box\s*=\s*'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?\s*'\s*"
            scroll_match = re.match(scroll_pattern, action_kwargs_str)
            if not scroll_match:
                return '', action_kwargs
            action_kwargs["start_point"] = (int(scroll_match.group(1)), int(scroll_match.group(2)))
            action_kwargs["end_point"] = (int(scroll_match.group(3)), int(scroll_match.group(4)))
            return action_type, action_kwargs
        # case2: direction only
        elif 'direction' in action_kwargs_str:
            scroll_pattern = r"direction\s*=\s*'(DOWN|UP|RIGHT|LEFT)'\s*"
            scroll_match = re.match(scroll_pattern, action_kwargs_str)
            if not scroll_match:
                return '', action_kwargs
            action_kwargs["direction"] = scroll_match.group(1)
            return action_type, action_kwargs
    else:
        raise ValueError(f"Unable to parse action: {action_content}")


def evaluate(args):
    """
    评估预测结果与真实标签的匹配程度
    
    Args:
        args: 包含以下属性的参数对象
            - prediction_file_path: 预测文件路径
            - ground_truth_file_path: 真实标签文件路径
            - split: 数据集划分名称（用于日志）
    """
    output_comparison = os.path.join(args.output_path, 'comparison.json')
    response_pairs = myload(args.prediction_file_path, args.ground_truth_file_path, output_comparison)
    print(f"Total response pairs loaded: {len(response_pairs)}")
    num = 0
    wrong_type = 0      # 错误识别动作类型的个数
    grounding_type = 0  # 正确识别为click和long_press的个数
    scroll_type = 0     # 正确识别为scroll的个数
    action_match_score_dict = defaultdict(int)  # 总完全正确动作个数
    action_number_dict = defaultdict(int)       # 总有效动作个数
    
    for idx, (pred_response, gt_response) in tqdm(enumerate(response_pairs), desc='Evaluating...'):
        try:
            gt_action_type, gt_action_kwargs = action_parser(gt_response)
            action_number_dict["ALL"] += 1
            action_number_dict[gt_action_type] += 1

            pred_action_type, pred_action_kwargs = action_parser(pred_response)
            if gt_action_type != pred_action_type:
                wrong_type += 1
                continue
            is_correct = False
            
            if gt_action_type in ["CLICK", "LONG_PRESS"]:
                grounding_type += 1
                pred_x, pred_y = pred_action_kwargs["point"]
                gt_x, gt_y = gt_action_kwargs["point"]
                distance = calculate_euclidean_distance(pred_x, pred_y, gt_x, gt_y)
                if calculate_euclidean_distance(pred_x, pred_y, gt_x, gt_y) <= args.threshold * 1000:
                    is_correct = True
            elif gt_action_type == "TYPE":
                if calculate_f1_score(pred_action_kwargs["content"], gt_action_kwargs["content"]) > 0.5:
                    is_correct = True
            elif gt_action_type == "OPEN_APP":
                if pred_action_kwargs["content"] == gt_action_kwargs["content"]:
                    is_correct = True
            elif gt_action_type == "SCROLL":
                scroll_type += 1
                if "start_point" in gt_action_kwargs and "end_point" in gt_action_kwargs:
                    pred_start_x, pred_start_y = pred_action_kwargs["start_point"]
                    pred_end_x, pred_end_y = pred_action_kwargs["end_point"]
                    gt_start_x, gt_start_y = gt_action_kwargs["start_point"]
                    gt_end_x, gt_end_y = gt_action_kwargs["end_point"]
                    start_distance = calculate_euclidean_distance(pred_start_x, pred_start_y, gt_start_x, gt_start_y)
                    if start_distance <= args.threshold * 1000:
                        pred_angle = calculate_scroll_angle(pred_start_x, pred_start_y, pred_end_x, pred_end_y)
                        gt_angle = calculate_scroll_angle(gt_start_x, gt_start_y, gt_end_x, gt_end_y)
                        if angle_difference(pred_angle, gt_angle) < 45:
                            is_correct = True
                elif "direction" in gt_action_kwargs:
                    if pred_action_kwargs["direction"] == gt_action_kwargs["direction"]:
                        is_correct = True
            else:
                # PRESS_HOME, PRESS_BACK, PRESS_APPSELECT, COMPLETE, INCOMPLETE, WAIT动作类型匹配就算正确
                is_correct = True
            
            if is_correct:
                action_match_score_dict["ALL"] += 1
                action_match_score_dict[gt_action_type] += 1
        
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {str(e)}, predicted response is \"{pred_response}\"")
            continue
    
    total_samples = action_number_dict["ALL"]
    # 所有动作中，有多少类型正确的
    type_acc = 1 - (wrong_type / total_samples) if total_samples > 0 else 0
    # 所有动作中，有多少动作完全正确
    step_acc = action_match_score_dict["ALL"] / total_samples

    # 正确识别为click和long_press的动作中，有多少grounding正确的
    grounding_acc = ((action_match_score_dict["CLICK"] + action_match_score_dict["LONG_PRESS"]) / grounding_type) if grounding_type > 0 else 0
    # 实际为click和long_press的动作中，有多少正确是识别为click和long_press的动作
    grounding_type_acc = (grounding_type / (action_number_dict.get("CLICK", 1) + action_number_dict.get("LONG_PRESS", 1))) if action_number_dict.get("CLICK", 0) > 0 else 0
    # 实际为scroll的动作中，有多少正确是识别为scroll的动作
    scroll_type_acc = (scroll_type / action_number_dict.get("SCROLL", 1)) if action_number_dict.get("SCROLL", 0) > 0 else 0
    
    # 输出日志
    logger.info("=" * 30 + f"Results" + "=" * 30)
    logger.info(f"Type Acc: {type_acc:.4f}")
    logger.info(f"Grounding Acc: {grounding_acc:.4f}")
    logger.info(f"Step Acc: {step_acc:.4f}")
    logger.info("-" * 20)
    logger.info(f"Click Type Acc: {grounding_type_acc:.4f}")
    logger.info(f"Scroll Type Acc: {scroll_type_acc:.4f}")
    
    # 输出各动作类型的匹配分数
    for action_type in ["CLICK", "LONG_PRESS", "TYPE", "SCROLL", "OPEN_APP", 
            "PRESS_BACK", "PRESS_HOME", "PRESS_APPSELECT", "COMPLETE", "INCOMPLETE", "WAIT"]:
        if action_number_dict.get(action_type, 0) > 0:
            score = action_match_score_dict[action_type] / action_number_dict[action_type]
            logger.info(f"Action Match Score - {action_type}: {score:.4f}")
    
    # 返回统计结果（可选）
    return {
        "type_acc": type_acc,
        "grounding_acc": grounding_acc,
        "step_acc": step_acc,
        "action_match_score_dict": dict(action_match_score_dict),
        "action_number_dict": dict(action_number_dict)
    }

###### 请注意，ground_truth_file_path参数使用和Swift SFT训练同格式的EVAL json/jsonl文件，可以包含其他字段但必须有完整的messages: system + user + assistant和images字段；
###### prediction_file_path使用Swift INFER同格式的输出，必要包含字段为：
###### ["response"]/["pred"]二选一即可，["image_path"]/["images"][0]/["images"][0]['path']三选一即可
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/look-ahead/eval/result/ByteDance-Seed/UI-TARS-7B-SFT/GUI-Odyssey-N5/GUI-Odyssey.jsonl')
    parser.add_argument('--ground_truth_file_path', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/look-ahead/data/GUI-Odyssey/uitars/eval_N5.json')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.14)
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.dirname(args.prediction_file_path)
    os.makedirs(args.output_path, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(args.output_path,'score.log'), mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    evaluate(args)