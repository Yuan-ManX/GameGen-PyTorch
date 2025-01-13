from collections import deque
from typing import Union, Tuple, List, Optional
import random
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from models.agent.blocks import Linear_QNet, QTrainer
from game.env import Environment, ActionResult


def plot(scores, mean_scores):
    """
    绘制训练过程中的得分变化图。

    该函数用于实时显示训练过程中每个游戏的得分以及平均得分的变化趋势。
    通过清除当前图形并绘制新的得分曲线，实现动态更新图表的效果。

    参数:
        scores (list): 每个游戏的得分列表。
        mean_scores (list): 每个游戏的平均得分列表。
    """
    # 清除当前图形，以避免图形重叠
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


class ValueForEndGame(Enum):
    """
    枚举类，用于表示游戏结束时价值的处理方式。

    Attributes:
        last_action (str): 表示游戏结束时使用最后一个动作的价值。
        not_exist (str): 表示游戏结束时不存在价值。
    """
    last_action = "last_action"
    not_exist = "not_exist"


@dataclass
class QAgentConfig:
    """
    Q 学习智能体（QAgent）的配置类。

    该数据类用于存储 Q 学习智能体的各种超参数配置，包括内存大小、批量大小、学习率、隐藏层大小等。

    Attributes:
        max_memory (int): 经验回放内存的最大容量。
        batch_size (int): 每次训练时的批量大小。
        lr (float): 学习率，用于控制模型参数更新的步长。
        hidden_state (int): 隐藏层的维度，用于控制神经网络的大小。
        value_for_end_game (ValueForEndGame): 游戏结束时价值的处理方式，可以是最后一个动作的价值或不存在价值。
        iterations (int): 训练的总迭代次数。
        min_deaths_to_record (int): 记录死亡事件的最小次数阈值。
        epsilon_start (float, optional): 初始的探索率。默认为 1.0。
        epsilon_min (float, optional): 探索率的最小值。默认为 0.01。
        epsilon_decay (float, optional): 探索率的衰减率。默认为 0.995。
        gamma (float, optional): 折扣因子，用于权衡即时奖励和未来奖励。默认为 0.9。
        train_every_iteration (int, optional): 每隔多少次迭代进行一次训练。默认为 10。
        save_every_iteration (Optional[int], optional): 每隔多少次迭代保存一次模型。默认为 None，表示不保存。
    """
    max_memory: int
    batch_size: int
    lr: float
    hidden_state: int
    value_for_end_game: ValueForEndGame
    iterations: int
    min_deaths_to_record: int
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.9
    train_every_iteration: int = 10
    save_every_iteration: Optional[int] = None


class ReplayMemory:
    """
    回放记忆（Replay Memory）类，用于存储智能体与环境交互的样本数据。

    该类使用双端队列（deque）作为存储结构，具有固定的最大容量。当新样本被添加时，如果队列已满，
    最旧的样本将被自动移除，以保持内存容量不变。

    参数:
        capacity (int): 回放记忆的最大容量。
    """
    def __init__(self, capacity: int):
        """
        初始化回放记忆。

        参数:
            capacity (int): 回放记忆的最大容量。
        """
        # 使用双端队列存储记忆，最大长度为 capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        将新的样本添加到回放记忆中。

        参数:
            state (np.ndarray): 当前状态。
            action (Union[np.ndarray, List[int]]): 执行的动作。
            reward (Union[np.ndarray, float]): 执行动作后获得的奖励。
            next_state (np.ndarray): 执行动作后的下一个状态。
            done (Union[np.ndarray, bool]): 标记当前状态是否为终止状态。
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        从回放记忆中随机抽取一批样本。

        参数:
            batch_size (int): 抽取的样本数量。

        返回:
            Tuple: 包含状态、动作、奖励、下一个状态和终止标记的元组。
        """
        # 随机抽取一批样本
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        # 将样本解压缩为状态、动作、奖励、下一个状态和终止标记的元组
        return zip(*batch)
    
    def __len__(self):
        """
        获取回放记忆中的样本数量。

        返回:
            int: 样本数量。
        """
        return len(self.memory)


class QAgent:
    """
    Q 学习智能体（QAgent）类，用于训练和执行 Q 学习算法。

    该智能体使用线性 Q 网络（Linear_QNet）进行决策，并通过经验回放和目标网络来稳定训练过程。

    参数:
        env (Environment): 环境对象，定义了智能体与环境的交互方式。
        config (QAgentConfig): QAgent 的配置，包含各种超参数。
        model_path (str): 模型参数的保存路径。
        dataset_path (str): 数据集的保存路径。
        last_checkpoint (Optional[str]): 上一个检查点的路径，用于恢复训练。
    """
    def __init__(
        self,
        env: Environment,
        config: QAgentConfig,
        model_path: str,
        dataset_path: str,
        last_checkpoint: Optional[str]
    ):
        """
        初始化 Q 学习智能体。

        参数:
            env (Environment): 环境对象。
            config (QAgentConfig): QAgent 的配置。
            model_path (str): 模型参数的保存路径。
            dataset_path (str): 数据集的保存路径。
            last_checkpoint (Optional[str]): 上一个检查点的路径。
        """
        self.config = config  # 保存配置
        self.model_path = model_path  # 保存模型路径
        self.memory = ReplayMemory(config.max_memory)  # 初始化回放记忆
        self.model = Linear_QNet(len(env.get_state()), config.hidden_state, env.actions_length())  # 初始化线性 Q 网络
        self.trainer = QTrainer(self.model, gamma=config.gamma)  # 初始化 Q 训练器
        self.env = env  # 保存环境对象
        self.steps = 0  # 初始化步数计数器
        self.dataset_path = dataset_path  # 保存数据集路径
        self.count_games = 0  # 初始化游戏计数
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)  # 初始化优化器
        self.recorded_actions = []  # 初始化记录的动作列表
        self.epsilon = config.epsilon_start  # 初始化探索率
        self.begin_iteration = 0  # 初始化开始迭代计数

        if last_checkpoint:
            # 如果提供了上一个检查点，则加载模型参数和优化器状态
            parameters = torch.load(last_checkpoint)
            self.model.load_state_dict(parameters["model"])
            self.optimizer.load_state_dict(parameters["optimizer"])
            self.count_games = parameters.get("count_games", 0)
            self.begin_iteration = parameters.get("begin_iteration", 0)
    
    def _remember(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        """
        将交互样本存储到回放记忆中。

        参数:
            state (np.ndarray): 当前状态。
            action (Union[np.ndarray, List[int]]): 执行的动作。
            reward (Union[np.ndarray, float]): 获得的奖励。
            next_state (np.ndarray): 下一个状态。
            done (Union[np.ndarray, bool]): 终止标记。
        """
        # 将样本添加到回放记忆中
        self.memory.append((state, action, reward, next_state, done))

    def _train_long_memory(self):
        """
        使用长记忆（经验回放）进行训练。

        如果回放记忆中的样本数量大于批量大小，则从中随机抽取一个批量进行训练。
        否则，使用整个回放记忆进行训练。
        """
        if len(self.memory) > self.config.batch_size:
            # 如果记忆中的样本数量大于批量大小，则随机抽取一个批量
            mini_sample = random.sample(self.memory, self.config.batch_size)
        else:
            # 否则，使用整个记忆进行训练
            mini_sample = self.memory

        # 将抽取的样本解压缩为状态、动作、奖励、下一个状态和终止标记
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # 调用训练步骤方法进行训练
        self._train_step(states, actions, rewards, next_states, dones)

    def _train_step(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[np.ndarray, float],
        next_state: np.ndarray,
        done: Union[np.ndarray, bool]
    ):
        """
        执行单步训练。

        参数:
            state (np.ndarray): 当前状态。
            action (Union[np.ndarray, List[int]]): 执行的动作。
            reward (Union[np.ndarray, float]): 获得的奖励。
            next_state (np.ndarray): 下一个状态。
            done (Union[np.ndarray, bool]): 终止标记。
        """
        self.optimizer.zero_grad()
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        loss.backward()
        self.optimizer.step()

    @property
    def snapshots_path(self):
        """
        获取快照保存路径。

        返回:
            str: 快照保存路径。
        """
        return os.path.join(self.dataset_path, "snapshots")

    @property
    def actions_path(self):
        """
        获取动作保存路径。

        返回:
            str: 动作保存路径。
        """
        return os.path.join(self.dataset_path, "actions")

    def _get_action(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        根据当前状态选择动作。

        该方法实现了 ε-贪婪策略，即以一定的概率选择随机动作，以探索环境；
        否则，选择 Q 值最高的动作。

        参数:
            state (np.ndarray): 当前状态。

        返回:
            Tuple[np.ndarray, int]: 动作向量和动作索引。
        """
        if random.random() < self.epsilon:
            # 以概率 ε 选择随机动作
            max_index = random.randint(0, self.env.actions_length() - 1)
        else:
            # 否则，根据 Q 网络选择动作
            with torch.no_grad():
                # 将状态转换为张量
                state_tensor = torch.tensor(state, dtype=torch.float)
                # 计算 Q 值
                q_values = self.model(state_tensor)
                # 选择 Q 值最高的动作索引
                max_index = torch.argmax(q_values).item()
        # 初始化动作向量
        final_move = [0] * self.env.actions_length()
        # 设置选择的动作为 1
        final_move[max_index] = 1
        # 返回动作向量和动作索引
        return np.array(final_move), max_index
    
    def _save_snapshot(self, step: int):
        """
        保存当前环境的快照。

        参数:
            step (int): 当前步骤。
        """
        plt.imsave(os.path.join(self.snapshots_path, f'{step}.jpg'), self.env.get_snapshot())
    
    def _save_actions(self):
        """
        保存记录的动作列表到文件。
        """
        with open(self.actions_path, mode="w") as file:
            file.write("\n".join([str(action) for action in self.recorded_actions]))
    
    def play_step(
        self,
        record: bool = False,
        step: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, ActionResult]:
        """
        执行一步游戏操作。

        参数:
            record (bool, optional): 是否记录当前步骤。默认为 False。
            step (Optional[int], optional): 当前步骤的编号。默认为 None。

        返回:
            Tuple[np.ndarray, np.ndarray, ActionResult]: 当前状态、选择的动作和执行动作后的结果。
        """
        # 获取当前状态
        old_state = self.env.get_state()
        # 根据当前状态选择动作
        action, max_index = self._get_action(old_state)
        # 增加步数计数器
        self.steps += 1
        if step is None:
            # 如果未提供步骤编号，则使用步数计数器
            step = self.steps
        # 执行动作并获取结果
        result = self.env.do_action(action)
        if record:
            # 保存当前环境的快照
            self._save_snapshot(step)
            # 记录选择的动作
            self.recorded_actions.append(max_index)
            # 保存记录的动作列表
            self._save_actions()
        # 返回当前状态、选择的动作和执行动作后的结果
        return old_state, action, result

    def train(self, show_plot: bool = False, record: bool = False, clear_old: bool = False):
        """
        训练智能体。

        参数:
            show_plot (bool, optional): 是否显示训练过程中的得分图。默认为 False。
            record (bool, optional): 是否记录训练过程中的动作和快照。默认为 False。
            clear_old (bool, optional): 是否清除之前的训练记录。默认为 False。
        """
        # 设置训练环境
        self._setup_training(clear_old)
        
        # 初始化每局得分的列表
        plot_scores = []
        # 初始化平均得分的列表
        plot_mean_scores = []
        # 初始化最高得分
        top_result = 0
        # 初始化总得分
        total_score = 0
        print(f"Begin iteration is {self.begin_iteration}")
        print(f"All iteration is {self.config.iterations}")
        if self.begin_iteration >= self.config.iterations:
            # 如果开始迭代次数大于或等于总迭代次数，则结束训练
            return
        for iteration in range(self.begin_iteration, self.config.iterations):
            # 执行一步游戏操作
            old_state, action, result = self.play_step(
                record=record and self.count_games >= self.config.min_deaths_to_record
            )
            # 获取奖励、下一个状态和终止标记
            reward, new_state, done = result.reward, result.new_state, result.terminated
            # 将样本添加到回放记忆中
            self.memory.push(old_state, action, result.reward, result.new_state, result.terminated)

            def do_training():
                # 从回放记忆中抽取一个批量
                batch = self.memory.sample(self.config.batch_size)
                # 执行单步训练
                self._train_step(*batch)

            # 如果回放记忆中的样本数量大于批量大小，并且当前迭代次数是训练间隔的倍数，则进行训练
            if len(self.memory) > self.config.batch_size and iteration % self.config.train_every_iteration == 0:
                do_training()
            # 更新探索率
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

            if done:
                # 增加游戏计数
                self.count_games += 1
                # 获取当前游戏的得分
                score = result.score
                # 重置环境
                self.env.reset()
                # 进行训练
                do_training()
                if record and self.count_games > self.config.min_deaths_to_record:
                    if self.config.value_for_end_game.value == ValueForEndGame.last_action.value:
                        # 增加步数计数器
                        self.steps += 1
                        # 记录最后一个动作
                        self.recorded_actions.append(self.env.actions_length())
                        # 保存快照
                        self._save_snapshot(self.steps)
                    elif self.config.value_for_end_game.value == ValueForEndGame.not_exist.value:
                        pass
                # 保存记录的动作列表
                self._save_actions()

                if score > top_result:
                    # 更新最高得分
                    top_result = score
                    # 保存智能体参数
                    self.save_agent(iteration)

                # 输出当前游戏的信息
                print('Game', self.count_games, 'Score', score, 'Record:', top_result, "Iteration:", iteration)
                if show_plot:
                    # 添加当前得分到列表中
                    plot_scores.append(score)
                    # 增加总得分
                    total_score += score
                    # 计算平均得分
                    mean_score = total_score / self.count_games
                    # 添加平均得分到列表中
                    plot_mean_scores.append(mean_score)
                    # 绘制得分图
                    plot(plot_scores, plot_mean_scores)
            if self.config.save_every_iteration is not None and iteration % self.config.save_every_iteration == 0:
                # 保存智能体参数
                self.save_agent(iteration)
        # 保存记录的动作列表
        self._save_actions()
        # 保存智能体参数
        self.save_agent(iteration+1)
        print(f"finish iteration is {iteration}")

    def _setup_training(self, clear_old: bool):
        """
        设置训练环境。

        参数:
            clear_old (bool): 是否清除之前的训练数据。
        """
        if clear_old:
            # 清除训练数据
            self._clear_training_data()
        else:
            # 加载训练数据
            self._load_training_data()
        # 创建快照保存目录
        os.makedirs(self.snapshots_path, exist_ok=True)
        if os.path.dirname(self.model_path) != "":
            # 创建模型保存目录
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def _clear_training_data(self):
        """
        清除训练数据。
        """
        # 重置步数计数器
        self.steps = 0
        # 清空记录的动作列表
        self.recorded_actions = []
        # 删除数据集目录
        shutil.rmtree(self.dataset_path)

    def _load_training_data(self):
        """
        加载训练数据。
        """
        try:
            # 计算快照数量作为步数
            self.steps = len([f for f in os.listdir(self.snapshots_path) if f.endswith('.jpg')])
            # 从动作文件中读取记录的动作列表
            with open(self.actions_path) as f:
                self.recorded_actions = [int(line) for line in f]
        except:
            # 如果加载失败，则重置步数和记录的动作列表
            self.steps = 0
            self.recorded_actions = []
        print(self.steps, len(self.recorded_actions))
        assert self.steps == len(self.recorded_actions)

    def save_agent(self, iteration: int):
        """
        保存智能体状态。

        参数:
            iteration (int): 当前迭代次数。
        """
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "count_games": self.count_games,
            "begin_iteration": iteration
        }, self.model_path)
