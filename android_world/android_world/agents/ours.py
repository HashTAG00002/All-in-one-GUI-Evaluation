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

# here, press_appselect ignored
OUR_SYSTEM_PROMPT = """As a Reasoning GUI Agent, your responsibility is to provide the correct solution that specifies the action to be executed, based on the global task goal, the action history, and the screenshot.

The action space is as follows:
CLICK(point=[x, y]) ## Click at a specific point on the screen using the coordinates (x, y) in the 'point' field.
LONG_PRESS(point=[x, y]) ## Long press at a specific point on the screen using the coordinates (x, y) in the 'point' field.
TYPE(content='xxx') ## Type the text in the 'content' field.
SCROLL(direction='DOWN' or 'UP' or 'RIGHT' or 'LEFT') ## Scroll in a specific direction set in the 'direction' field.
OPEN_APP(content='xxx') ## Open an application specified by the text in the 'content' field.
PRESS_HOME() ## Go to the home screen.
PRESS_BACK() ## Go back to the previous screen.
WAIT() ## Wait 5 seconds for the screen to load.
ANSWER(content='xxx') ## Provide the specified answer in the 'content' field, which is needed for information retrieval task.
COMPLETED(content='xxx') ## Indicate that the task is completed, and the information in the 'content' field explains why the task is completed.
INCOMPLETE(content='xxx') ## Indicate that the task is incomplete, and the information in the 'content' field explains why the task is incomplete.

The solution includes the following four parts: thinking, answer, reflection, and conclusion. Each part is to be enclosed within specific tags
1. <thinking>thinking</thinking>: Present your complete logical chain of problem-solving. It follows a clear and concise three-step logical reasoning process, i.e., Step 1: Perception\nStep 2: Comprehension\nStep 3: Association.
    - Step 1: Perception: Describe in detail the layout, state, and key elements of the current-step screenshot.
    - Step 2: Comprehension: Infer what you should do in the current step based on the global task goal, the action history, and the screenshot.
    - Step 3: Association: Decide which action to execute and focus the corresponding region in the screenshot, based on the analysis from "Step 1: Perception" and "Step 2: Comprehension".
2. <answer>answer</answer>: Provide the action to be executed in the specified format of the Action Space defined above. If you conclude that the task cannot be completed, output exactly: "Task Failed".
3. <reflection>reflection</reflection>: Review the accuracy of the reasoning process within <thinking> and then verify the consistency between the reasoning process within <thinking> and the result within <answer>. If any error or inconsistency exists, end with: "Verification Failed"; otherwise, end with: "Verification Succeeded".
4. <conclusion>conclusion</conclusion>: Summarize the action taken in the current step.

Respond according to the user's input, supplying the requested sections of the problem-solving process, i.e., <thinking>thinking</thinking>\n<answer>answer</answer>\n<reflection>reflection</reflection>\n<conclusion>conclusion</conclusion>.
Solve the problem in accordance with these guidelines."""

OUR_USER_PROMPT = """You are given a global task goal, your action history, and a screenshot. You need to provide the correct solution that specifies the action to be executed.
Global Task Goal: {task}
Action History: {action_history}
Screenshot: 
"""

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
def build_system_messages(instruction, current_screenshot_path, history_list, infer_mode_N = 5, add_thought=True):
    if add_thought:
        query_messages = [
            {'role': 'system', 'content': [{"text": OUR_SYSTEM_PROMPT}]},
            {'role': 'user', 'content': [
                {"text": OUR_USER_PROMPT.format(task=instruction, action_history=json.dumps(history_list))},
                {"image": current_screenshot_path},
            ]},
        ]
    else:
        raise NotImplementedError
    return query_messages



class OUR_AGENT(base_agent.EnvironmentInteractingAgent):
    """mobile agent for Android."""

    def __init__(self, env: interface.AsyncEnv, vllm, src_format, name: str = "OUR_AGENT", output_path = ""):
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
            "scroll": json_action.SCROLL, # scroll -> swipe
            "open_app": json_action.OPEN_APP,
            #"press_appselect": json_action.DRAG_AND_DROP,
            "press_home": json_action.NAVIGATE_HOME,
            "press_back": json_action.NAVIGATE_BACK,
            "completed": json_action.STATUS,
            "incompleted": json_action.STATUS,
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
            action_kwargs_str = re.sub(r'\s+', '', action_kwargs_str).replace('[',"'(",1).replace(']',")'",1).replace('point','start_box',1)
            click_match = re.match(r"start_box='\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)'", action_kwargs_str)
            if not click_match:
                raise NotImplementedError
            x, y = convert_point_format([int(click_match.group(1)), int(click_match.group(2))], img_ele, src_format=src_format, tgt_format=tgt_format)
            print(x,y)
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
            type_match = re.match(r"content\s*=\s*'(.*)'\s*$", action_kwargs_str)
            if not type_match:
                raise NotImplementedError
            app_name = type_match.group(1).replace("\\'", "'").replace('\\"', '"').replace('\\n', '\n')
        # 匹配 scroll & drag
        elif action_type in {json_action.SCROLL}:
            action_kwargs_str = re.sub(r'\s+', '', action_kwargs_str)
            scroll_match = re.match(r"direction='(DOWN|UP|RIGHT|LEFT)'", action_kwargs_str)
            if not scroll_match:
                raise NotImplementedError
            direction = scroll_match.group(1).lower()
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
        messages = build_system_messages(goal, current_screenshot_path=screenshot_file, history_list=self._summarys, add_thought=self.add_thought)

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
        reflection = None
        conclusion = None
        env_exception = False
        try:
            if self.add_thought:
                thought = re.search(r'<thinking>(.*?)</thinking>', action_response, re.DOTALL).group(1)
            else:
                thought = None
            action_string = re.search(r'<answer>(.*?)</answer>', action_response, re.DOTALL).group(1)
            reflection = re.search(r'<reflection>(.*?)</reflection>', action_response, re.DOTALL).group(1)
            conclusion = re.search(r'<conclusion>(.*?)</conclusion>', action_response, re.DOTALL).group(1)
            summary = f"Step {step_idx+ 1}:\n\n Instruction: {conclusion}\n Action: {action_string}"
            action = self.action_parser(action_string, current_image_ele, src_format=self.src_format, tgt_format='abs_origin')
            result["action"] = action
        except NotImplementedError as e:
            action = json_action.JSONAction(action_type=json_action.UNKNOWN)
            result["action"] = action
        except:
            traceback.print_exc()
            print(action_response)
            raise
        finally:
            self._text_actions.append(action_string)
            self._actions.append(action)
            self._thoughts.append(thought)
            self._summarys.append(summary)
            self._response.append(action_response)
            try:
                actuation.execute_adb_action(
                    action,
                    [],
                    self.env.logical_screen_size,
                    self.env.controller
                )
            except Exception as e:
                env_exception = True
                exception = 'Error from the environment:\n' + str(e)
    
        with open(os.path.join(task_output_dir, "action.jsonl"), 'a', encoding='utf-8') as f:
            f.write(f'Step {step_idx}:\n{self._response[-1]}\n')
            if env_exception:
                f.write(exception + '\n')
            f.write('\n')
        return base_agent.AgentInteractionResult(
            done=action.action_type == json_action.STATUS,
            data=result,
        )