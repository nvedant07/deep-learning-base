import sys, os
sys.path.append(f'{os.path.abspath(os.getcwd())}/deep-learning-base/')
sys.path.append(f'{os.path.abspath(os.getcwd())}/deep-learning-base/architectures')
sys.path.append(f'{os.path.abspath(os.getcwd())}/deep-learning-base/datasets')
sys.path.append(f'{os.path.abspath(os.getcwd())}/deep-learning-base/training')
sys.path.append(f'{os.path.abspath(os.getcwd())}/deep-learning-base/attack')
sys.path.append(f'{os.path.abspath(os.getcwd())}/deep-learning-base/self_supervised')

import pprint
pprint.pprint (sys.path)