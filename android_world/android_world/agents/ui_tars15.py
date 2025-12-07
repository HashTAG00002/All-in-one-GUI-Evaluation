# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SeeAct agent for Android."""
from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import tools
from android_world.agents import new_json_action as json_action
from PIL import Image
import base64
import json
import pprint
import os
import re
import time
from qwen_vl_utils import smart_resize
from io import BytesIO

from android_world.agents.coordinate_resize import update_image_size_
from android_world.agents.coordinate_resize import convert_point_format
import traceback

# here, scroll -> swipe, press_home/press_back -> system_button
# from https://github.com/bytedance/UI-TARS/issues/83
UITARS_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='xxx')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x2,y2)<|box_end|>')
open_app(app_name='xxx')
press_home()
press_back()
answer(content='xxx') # Submit the answer to the user's question
wait()
finished(content='xxx') # Submit the task regardless of whether it succeeds or fails.
"""


UITARS_USER_PROMPT_FORMAT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
Thought: ...
Action: ...

## Action Space
{action_space}

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
Task Instruction: {instruction}"""


UITARS_USER_PROMPT_FORMAT_NO_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
Action: ...

## Action Space
{action_space}

## User Instruction
Task Instruction: {instruction}"""


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fetch_resized_image(screenshot_file):
    screenshot = Image.open(screenshot_file)
    width, height = screenshot.size
    current_image_ele = update_image_size_({'image': screenshot_file, 'width': width, 'height': height})
    resized_width = current_image_ele['resized_width']
    resized_height = current_image_ele['resized_height']
    screenshot = screenshot.resize((resized_width, resized_height))
    return screenshot, resized_width, resized_height, current_image_ele


# 请注意，如果调用android_world/android_world/agents/infer_ma3.py中的predict_mm，需要遵循的格式为：
# "messages": [
#     {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
#     {"role": "user",   "content": [{"text": "..."}, {"image": "./AudioRecorderRecordAudio/screenshot_0.png"}]}
# ]
def build_system_messages(instruction, screenshots, history, infer_mode_N = 5, add_thought=True):
    if add_thought:
        query_messages = [
            {'role': 'system', 'content': [{"text": "You are a helpful assistant."}]},
            {'role': 'user', 'content': [{"text": UITARS_USER_PROMPT_FORMAT_THOUGHT.format(action_space=UITARS_ACTION_SPACE, instruction=instruction)}]},
        ]
    else:
        query_messages = [
            {'role': 'system', 'content': [{"text": "You are a helpful assistant."}]},
            {'role': 'user', 'content': [{"text": UITARS_USER_PROMPT_FORMAT_NO_THOUGHT.format(action_space=UITARS_ACTION_SPACE, instruction=instruction)}]},
        ]
    for i,screenshot_path in enumerate(screenshots):
        assert len(history) + 1 == len(screenshots)
        if i >= len(screenshots) - infer_mode_N:
            query_messages.append({'role': 'user', 'content': [{"image": screenshot_path}]})
        if i < len(history):
            query_messages.append({'role': 'assistant', 'content': [{"text": history[i]}]})
    return query_messages



class UI_TARS15(base_agent.EnvironmentInteractingAgent):
    """mobile agent for Android."""

    def __init__(self, env: interface.AsyncEnv, vllm, src_format, name: str = "UI_TARS", output_path = ""):
        super().__init__(env, name)
        self._actions = []
        self._screenshots = []
        self._summarys = []
        self._thoughts = []
        self.output_result = {}
        self.output_path = output_path
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.vllm = vllm

        self.add_thought = True
        self._text_actions = []
        self.src_format = src_format

        self.output_list = []
        self._response = []
        self.task_name = {}

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._text_actions.clear()
        self._screenshots.clear() # TODO
        self._summarys.clear()
        self._thoughts.clear()
        self._response.clear()
  
    def initialize_chrome(self):
        print("Running additional chrome initialization...")
        # handle chrome initialization problem for browser tasks
        adb_utils.launch_app("chrome", self.env.controller)
        time.sleep(5)

        tool_controller = tools.AndroidToolController(env=self.env.controller)
        time.sleep(2)

        first_op = False
        try:
            print("try first variant...")
            tool_controller.click_element("Use without an account")
            time.sleep(5.0)
            first_op = True
        except:
            print("Failed to click 'Use without an account' button.")
        pass
    
        if not first_op:
            print("try second variant...")
            try:
                tool_controller.click_element("Accept & continue")
            except:
                pass
            time.sleep(3.0)
            try:
                tool_controller.click_element("No thanks")
            except:
                pass
            time.sleep(5.0)
      
        adb_utils.press_home_button(self.env.controller)
        time.sleep(2.0)
        print("Done additional chrome initialization")
    
    def get_task_name(self, suite):
        for name, instances in suite.items():
            self.task_name[instances[0].goal] = name
  
    def action_parser(
        self, action_string, img_ele, src_format='abs_resized', tgt_format='abs_origin'
    ) -> json_action.JSONAction:
        """Converts a UI-TARS action string to a JSONAction object.
        Args:
        action_string
        elements: UI elements.

        Returns:
        The corresponding JSONAction object.

        """
        action_type_mapping = {
        "click": json_action.CLICK,
        "long_press": json_action.LONG_PRESS,
        "type": json_action.INPUT_TEXT,
        "scroll": json_action.SWIPE, # scroll -> swipe
        "open_app": json_action.OPEN_APP,
        "drag": json_action.DRAG_AND_DROP,
        "press_home": json_action.NAVIGATE_HOME,
        "press_back": json_action.NAVIGATE_BACK,
        "finished": json_action.STATUS,
        "answer": json_action.ANSWER,
        "wait": json_action.WAIT,
        }

        x = None
        y = None
        text = None
        direction = None
        goal_status = None
        app_name = None

        action_string = re.sub(r'\s+', ' ', action_string)
        action_type = action_string.split('(')[0].lower() # ALL LOWER!
        action_kwargs_str = action_string[len(action_type)+1:-1]
        if action_type not in action_type_mapping:
            raise NotImplementedError
        else:
            action_type = action_type_mapping[action_type]

        # 匹配 click(start_box='(x,y)') CLICK(point='[x, y]')
        if action_type in {json_action.CLICK, json_action.LONG_PRESS}:
            action_kwargs_str = re.sub(r'\s+', '', action_kwargs_str).replace('[','(',1).replace(']',')',1).replace('<|box_start|>','').replace('<|box_end|>','')
            click_match = re.match(r"start_box='\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)'", action_kwargs_str)
            if not click_match:
                raise NotImplementedError
            x, y = convert_point_format([int(click_match.group(1)), int(click_match.group(2))], img_ele, src_format=src_format, tgt_format=tgt_format)
        # 匹配 type(content='xxx')
        elif action_type in {json_action.INPUT_TEXT, json_action.ANSWER}:
            type_match = re.match(r"content\s*=\s*'(.*)'\s*$", action_kwargs_str)
            if not type_match:
                raise NotImplementedError
            text = type_match.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
            if action_type == json_action.ANSWER:
                self.env.interaction_cache = text
        # 匹配 open_app(app_name='xxx')
        elif action_type == json_action.OPEN_APP:
            type_match = re.match(r"app_name\s*=\s*'(.*)'\s*$", action_kwargs_str)
            if not type_match:
                raise NotImplementedError
            app_name = type_match.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
        # 匹配 scroll & drag
        elif action_type in {json_action.SWIPE, json_action.DRAG_AND_DROP}:
            action_kwargs_str = re.sub(r'\s+', '', action_kwargs_str).replace('[','(',2).replace(']',')',2).replace('<|box_start|>','').replace('<|box_end|>','')
            scroll_pattern = r"start_box='\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)',end_box='\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)'"
            scroll_match = re.match(scroll_pattern, action_kwargs_str)
            if not scroll_match:
                raise NotImplementedError
            x1, y1 = convert_point_format([int(scroll_match.group(1)), int(scroll_match.group(2))], img_ele, src_format=src_format, tgt_format=tgt_format)
            x2, y2 = convert_point_format([int(scroll_match.group(3)), int(scroll_match.group(4))], img_ele, src_format=src_format, tgt_format=tgt_format)
            direction = [x1, y1, x2, y2]
        # 匹配 finished(content='xxx')
        elif action_type == json_action.STATUS:
            goal_status = "task_complete"
        elif action_type in {json_action.NAVIGATE_HOME, json_action.NAVIGATE_BACK, json_action.WAIT}:
            pass
        else:
            raise NotImplementedError
        
        action_json = json_action.JSONAction(
            action_type=action_type,
            x=x,
            y=y,
            text=text,
            direction=direction,
            goal_status=goal_status,
            app_name=app_name,
        )
        return action_json

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        result = {
            "ui_elements": None,
            "screenshot": None,
            "actionable_elements": None,
            "action_gen_payload": None,
            "action_gen_response": None,
            "action_ground_payload": None,
            "action_ground_response": None,
            "seeact_action": None,
            "action": None,
            "action_description": None,
        }
        step_idx = len(self._screenshots)
        state = self.get_post_transition_state()
        result["ui_elements"] = state.ui_elements

        result["screenshot"] = state.pixels
        screenshot = Image.fromarray(state.pixels)

        if self.output_path:
            if goal not in self.task_name:
                task_output_dir = os.path.join(self.output_path, goal.replace(" ", "_")[:50])
            else:
                task_output_dir = os.path.join(self.output_path, self.task_name[goal])
        
            os.makedirs(task_output_dir, exist_ok=True)
            screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx}.png")
            screenshot.save(screenshot_file)
            self._screenshots.append(screenshot_file)
        else:
            self._screenshots.append(screenshot)
        
        screenshot, resized_width, resized_height, current_image_ele = fetch_resized_image(screenshot_file)
        messages = build_system_messages(goal, screenshots=self._screenshots, history=self._response, add_thought=self.add_thought)

        action_response = None
        action = None
        action_response, _, _ = self.vllm.predict_mm(
            "",
            [],
            messages=messages
        )
        
        result["action_response"] = action_response
        print('========== action_response ==========')
        pprint.pprint(action_response)

        action_string = None
        thought = None
        try:
            if self.add_thought and 'Thought:' in action_response and 'Action:' in action_response:
                thought = action_response.split('Action:')[0].replace('Thought:','',1).strip()
            else:
                thought = None
            action_string = action_response.split('Action:')[1].strip()
            action = self.action_parser(action_string, current_image_ele, src_format=self.src_format, tgt_format='abs_origin')
            result["action"] = action
        except seeact_utils.ParseActionError as e:
            action = json_action.JSONAction(action_type=json_action.UNKNOWN)
            result["seeact_action"] = None
            result["action"] = action
        except:
            traceback.print_exc()
            print(action_response)
            raise
        else:
            actuation.execute_adb_action(
                action,
                [],
                self.env.logical_screen_size,
                self.env.controller
            )
      
            self._text_actions.append(action_string)
            self._actions.append(action)
            self._thoughts.append(thought)
            self._response.append(action_response)
    
        with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
            for item in self._response:
                f.write(item + '\n')

        return base_agent.AgentInteractionResult(
            done=action.action_type == json_action.STATUS,
            data=result,
        )
