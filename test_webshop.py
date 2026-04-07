import os, sys, gym, pdb
project_root = "/raid1/HOME/szhang/jywu/code/Planning/SkillRL/agent_system/environments/env_package/webshop/webshop"
sys.path.append(project_root)
from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
env_kwargs = {
            'observation_mode': 'text', 
            'num_products': None, 
            'human_goals': False,
            'file_path': '/raid1/HOME/szhang/jywu/code/Planning/SkillRL/agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json',
            'attr_path': '/raid1/HOME/szhang/jywu/code/Planning/SkillRL/agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json'
            }


env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
seed_for_reset = 10

# # pdb.set_trace()
obs, info = env.reset(seed=seed_for_reset)
actions = env.get_available_actions()
print(actions)
observation, reward, done, info = env.step('search[x]')    
print(observation)
# print(obs)
# print(actions)


## val
# import json
# idx = range(0,6910)
# all_obs = []
# for i in idx:
#     obs, info = env.reset(session=i)
#     all_obs.append(obs)

# with open('webshop_tasks.json', 'w') as f:
#     json.dump(all_obs, f)