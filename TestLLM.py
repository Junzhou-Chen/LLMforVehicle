import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.llms.llamacpp import LlamaCpp
from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)

OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
model_path = r'G:\projects\llama.cpp\models\13B\ggml-model-q8_0.gguf'
os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
# llm = ChatOpenAI(
#     openai_api_key = "EMPTY",
#     openai_api_base = "http://localhost:8000/v1",
#     temperature=0,
#     model_name='gpt-3.5-turbo-16k-0613', # or any other model with 8k+ context
#     max_tokens=1024
# )
llm = LlamaCpp(model_path=model_path,
               verbose=False,
               # temperature=0.5,
               n_gpu_layers=15,
               n_ctx=4096,
               max_tokens=4096)


# base setting
vehicleCount = 15

# environment setting
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "vehicles_count": vehicleCount,
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": 40,
    "vehicles_density": 2,
    "show_trajectories": True,
    "render_agent": True,
}


env = gym.make('highway-v0', render_mode="rgb_array")
env.configure(config)
env = RecordVideo(
    env, './results-video',
    name_prefix=f"highwayv0"
)
env.unwrapped.set_record_video_wrapper(env)
obs, info = env.reset()
env.render()

# scenario and driver agent setting
if not os.path.exists('results-db/'):
    os.mkdir('results-db')
database = f"results-db/highwayv0.db"
sce = Scenario(vehicleCount, database)
toolModels = [
    getAvailableActions(env),               # Get Available Actions
    getAvailableLanes(sce),                 # Get Available Lanes
    getLaneInvolvedCar(sce),                # Get Lane Involved Car
    isChangeLaneConflictWithCar(sce),       # Whether change lone conflict with car
    isAccelerationConflictWithCar(sce),     #
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]
DA = DriverAgent(llm, toolModels, sce, verbose=True)
outputParser = OutputParser(sce, llm)
output = None
done = truncated = False
frame = 0
try:
    while not (done or truncated):
        sce.upateVehicles(obs, frame)
        DA.agentRun(output)
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)
        env.render()
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
        obs, reward, done, info, _ = env.step(output["action_id"])
        print(output)
        frame += 1
finally:
    env.close()
