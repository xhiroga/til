import datetime
import io
import os
import turtle

import cv2
import gymnasium as gym
import numpy as np
import torch as th
from PIL import Image
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

import wandb
from wandb.integration.sb3 import WandbCallback


class DrawingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, target_image, render_mode=None, size=64):
        super(DrawingEnv, self).__init__()

        self.size = size
        self.agent_image = np.zeros((self.size, self.size), dtype=np.uint8)
        image = Image.open(target_image).convert("L")  # モノクロに変換

        self.target_image = np.array(image.resize((self.size, self.size)))
        self.target_image_transposed = self.target_image[
            None, :, :
        ]  # チャネル次元を追加
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8
                ),
                "target": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8
                ),
            }
        )

        self.action_space_degree = 4
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.action_space_degree,), dtype=np.float32
        )

        self.screen = turtle.Screen()
        # ウィンドウサイズはウィンドウのマージン込なので、多めに取らないと全範囲が見えない
        self.screen.setup(width=self.size + 64, height=self.size + 64)
        self.agent = turtle.Turtle()

        self.previous_mean_diff = None

        self.max_step = self.size / 2  # 勘

        self.reset()

    def _get_obs(self):
        return {
            "agent": self.agent_image[None, :, :],  # チャネル次元を追加
            "target": self.target_image_transposed,
        }

    def _calc_reward(self):
        diff = cv2.absdiff(self.agent_image, self.target_image)
        mean_diff = np.mean(diff)
        improvement = mean_diff - self.previous_mean_diff
        wow = improvement - 255 / self.max_step
        self.previous_mean_diff = mean_diff
        # 単にimprovementを用いると途中からほとんど線を引かなくなる（目標が高すぎ現象）
        # improvement if improvement > 0 else -mean_diff とすると、自分で黒い線を引く → 白い線で上書きする、でスコアを稼げてしまう...
        return wow

    def reset(self, seed=None, options=None):
        self.screen.clearscreen()
        self.agent.reset()
        self.agent.hideturtle()
        self.agent.speed(0)
        self._show_drawing_area()
        self.agent.teleport(0, 0)
        self.agent.pensize(self.size // 16 + 1)  # これも勘
        self.current_step = 0

        self.agent_image = np.ones((self.size, self.size), dtype=np.uint8)
        self.previous_mean_diff = self._calc_reward()

        return self._get_obs(), {}

    # マジで便利、絶対実装したほうがいい。学習が膠着したときにフリーズしてるのかどうか、枠の再描画でひと目で判断できる
    def _show_drawing_area(self):
        # 枠をキャプチャ領域より広く実装しないと、枠を白塗りで消すのが最適行動になってしまう
        self.agent.teleport(self.size / 2 + 1, self.size / 2 + 1)
        self.agent.pensize(1)
        for _ in range(4):
            self.agent.right(90)
            self.agent.forward(self.size + 2)

    def step(self, action):
        (x, y, distance, gray) = action
        x = x * self.size - self.size / 2
        # おまじない程度だが、アクションはCNNに合わせて右手系になっていると解釈し、step内で左手系に直す
        y = self.size / 2 - y * self.size
        distance = distance * self.size

        self.agent.teleport(x, y)
        self.agent.pencolor(gray, gray, gray)  # モノクロに設定
        self.agent.forward(distance)

        self.current_step += 1

        # setupは原点を中心にウインドウを展開したが、postscriptは原点から正方向に描画する上に、座標が左手系なので非常に分かりづらい...
        ps = self.screen.getcanvas().postscript(
            colormode="color",
            x=-(self.size / 2),
            y=-(self.size / 2),
            width=self.size,
            height=self.size,
        )
        self.agent_image = np.array(
            Image.open(io.BytesIO(ps.encode("utf-8")))
            .convert("L")
            .resize((self.size, self.size))
        )

        reward = self._calc_reward()
        observation = self._get_obs()
        terminated = self.current_step >= self.max_step
        return observation, reward, terminated, False, {}

    def close(self):
        self.screen.bye()


class CustomCallBack(BaseCallback):
    def __init__(self, task_name, verbose=0):
        super(CustomCallBack, self).__init__(verbose)
        self.log_dir = f"logs/{task_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.before_update_param = None
        self.update_counter = 0

    def _on_rollout_start(self) -> None:
        if self.before_update_param is not None:
            self.compare_weights()

        self.before_update_param = {
            name: param.clone() for name, param in self.model.policy.named_parameters()
        }

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        buffer_actions = self.locals["buffer_actions"]
        rewards = self.locals["rewards"]
        message = (
            f"n_calls={self.n_calls}, "
            f"rewards={rewards}, "
            f"actions={actions}, "
            f"buffer_actions={buffer_actions}"
        )
        self._write_log(
            "on_step",
            message,
        )
        if self.n_calls % 16 == 15:
            new_obs = self.locals["new_obs"]
            agent_image = new_obs["agent"]
            if isinstance(agent_image, np.ndarray):
                img_array = agent_image[0].astype(np.uint8)
                img_array = img_array.squeeze()  # 形状を (height, width) に変更
                img = Image.fromarray(img_array, mode="L")  # モノクロとして保存
                img.save(f"{self.log_dir}/observation_{self.n_calls}.png")

        return True

    # _on_rollout_endの終了後にポリシーが更新されるため、このタイミングで利用可能になる新しい情報はない...はず...
    def _on_rollout_end(self) -> None:
        self.update_counter += 1

    def compare_weights(self):
        for name, param in self.model.policy.named_parameters():
            before = self.before_update_param[name]
            after = param.clone()
            mean_diff = th.mean(after - before).item()
            max_diff = th.max(after - before).item()
            self._write_log(
                "compare_weights",
                f"Update #{self.update_counter} Layer {name} | {mean_diff=} | {max_diff=}",
            )

    def _write_log(self, file_name, message):
        with open(f"{self.log_dir}/{file_name}.log", "a") as f:
            f.write(message + "\n")


target_image_path = "data/monostar.bmp"
env = DrawingEnv(target_image_path)

policy_type = "MultiInputPolicy"
action_noise = NormalActionNoise(
    mean=np.zeros(env.action_space_degree),
    sigma=0.5 * np.ones(env.action_space_degree),
)
buffer_size = 1000
learning_rate = 0.01  # 膠着を打破するためにデフォルトの10倍にしてみる
total_timesteps = 512

config = {
    "policy_type": policy_type,
    "action_noise": action_noise,
    "buffer_size": buffer_size,
    "learning_rate": learning_rate,
    "total_timesteps": total_timesteps,
}
run = wandb.init(
    project="rl-turtle",
    sync_tensorboard=True,
    config=config,
    name="reward=wow, action=(x, y, distance, gray)",
    save_code=True,
)

# buffer_sizeを指定しないと、`buffers.py:241: UserWarning: This system does not have apparently enough memory to store the complete replay buffer 24.61GB > 6.86GB` のようにエラーになる
model = TD3(
    policy_type,
    env,
    action_noise=action_noise,
    buffer_size=buffer_size,
    learning_rate=learning_rate,
    verbose=1,
)
model.learn(
    total_timesteps=total_timesteps,
    log_interval=4,
    callback=[
        CustomCallBack(
            f"drawing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", verbose=1
        ),
        WandbCallback(verbose=1),
    ],
)
run.finish()
