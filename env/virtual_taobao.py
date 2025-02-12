import gym
import numpy as np

class VirtualTaobao(gym.Env):
    def __init__(self):
        super(VirtualTaobao, self).__init__()
        
        # 定义状态空间，明确指定dtype
        self.state_dim = 91  # 88维静态属性 + 3维动态属性
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32  # 明确指定dtype
        )
        
        # 定义动作空间，明确指定dtype
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(27,),
            dtype=np.float32  # 明确指定dtype
        )
        
        self.reset()
        
    def reset(self):
        # 初始化状态，确保返回float32类型
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        return self.state
        
    def step(self, action):
        # 确保动作是float32类型
        action = np.array(action, dtype=np.float32)
        
        # 更新状态
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        
        # 计算奖励
        reward = float(np.sum(action * self.state[:27]) / 27)
        
        # 判断是否结束
        done = False
        
        # 额外信息
        info = {}
        
        return self.state, reward, done, info 