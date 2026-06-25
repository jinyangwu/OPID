from scienceworld import ScienceWorldEnv
import json, random, pdb
variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'

env = ScienceWorldEnv("", None, envStepLimit=100)
taskNames = env.get_task_names()
print(taskNames)

for task_id, taskName in enumerate(taskNames):
    env.load(taskName, 0, "easy", True)
    print(f"Task {task_id}: {taskName}")
    print(f"Variations: {len(env.get_variations_train())} train, {len(env.get_variations_dev())} dev, {len(env.get_variations_test())} test")
    # with open(f'sciworld_variations/{task_id}_{taskName}_train.txt', 'w') as f:
    #     for var in env.get_variations_train():
    #         f.write(f"{var}\n")
    # with open(f'sciworld_variations/{task_id}_{taskName}_dev.txt', 'w') as f:
    #     for var in env.get_variations_dev():
    #         f.write(f"{var}\n")
    # with open(f'sciworld_variations/{task_id}_{taskName}_test.txt', 'w') as f:
    #     for var in env.get_variations_test():
    #         f.write(f"{var}\n")
# with open(variation_path, 'r') as f:
#     variations_idx = json.load(f)
# random.seed(0)

# print(len(variations_idx['train']))
# print(len(variations_idx['test']))


# task_id, task_variation = random.choice(variations_idx['train'])
# taskName = taskNames[task_id]
# env.load(taskName, task_variation, "easy", True)
# print(len(env.task_names)) #30

# print(f"Train: {len(env.get_variations_train())}")
# print(f"Dev: {len(env.get_variations_dev())}")
# print(f"Test: {len(env.get_variations_test())}")
# # while True:
# #     print(env.get_possible_actions())
# #     print(env.get_possible_objects())
# #     action = input("action:")
# #     observation, reward, done, info = env.step(action)
# #     print("obs:", observation)
# #     print("reward:", reward)
# #     print("done", done)


