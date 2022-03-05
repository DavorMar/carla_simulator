import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, name = "memory", chkpt_dir = "tmp\memory"):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.file_name = name
        self.chkpt_dir = chkpt_dir

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save_memory(self):
        print("saving states")
        np.save(fr"{self.chkpt_dir}\{self.file_name}-state", self.state_memory[-5000:])
        print("saving states_")
        np.save(fr"{self.chkpt_dir}\{self.file_name}-states_", self.new_state_memory[-5000:])
        print("saving actions")
        np.save(fr"{self.chkpt_dir}\{self.file_name}-actions", self.action_memory[-5000:])
        print("saving rewards")
        np.save(fr"{self.chkpt_dir}\{self.file_name}-rewards", self.reward_memory[-5000:])
        np.save(fr"{self.chkpt_dir}\{self.file_name}-dones", self.terminal_memory[-5000:])


    def load_memory(self):
        self.state_memory = np.load(fr"{self.chkpt_dir}\{self.file_name}-state.npy")
        self.new_state_memory = np.load(fr"{self.chkpt_dir}\{self.file_name}-states_.npy")
        self.action_memory = np.load(fr"{self.chkpt_dir}\{self.file_name}-actions.npy")
        self.reward_memory = np.load(fr"{self.chkpt_dir}\{self.file_name}-rewards.npy")
        self.terminal_memory = np.load(fr"{self.chkpt_dir}\{self.file_name}-dones.npy")