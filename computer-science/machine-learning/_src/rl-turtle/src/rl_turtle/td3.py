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

        self.screen = turtle.Screen()
        # ウィンドウサイズはウィンドウのマージン込なので、多めに取らないと全範囲が見えない
        self.screen.setup(width=self.size + 64, height=self.size + 64)
        self.agent = turtle.Turtle()

        image = Image.open(target_image).convert("L")  # モノクロに変換
        self.target_image = np.array(image.resize((self.size, self.size)))
        self.target_image_transposed = self.target_image[None, :, :]
        self.target_image_inverted = 255 - self.target_image

        self.agent_image = np.ones((self.size, self.size), dtype=np.uint8)
        self.previous_agent_image = self.agent_image
        self.absdiff = self._get_absdiff()
        self.previous_mean_diff = np.mean(self.absdiff[None, :, :])

        self.observation_space = gym.spaces.Dict(
            {
                "absdiff": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8
                ),
                "agent": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8
                ),
                "target": gym.spaces.Box(
                    low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8
                ),
            }
        )

        self.action_space_degree = 3
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.action_space_degree,), dtype=np.float32
        )

        self.max_step = self.size / 2  # 勘
        self.expected_growth_rate = 128 ** (1 / self.max_step)

        self.reset()

    def _get_absdiff(self):
        return cv2.absdiff(self.agent_image, self.target_image)

    def _get_obs(self):
        return {
            "absdiff": self.absdiff[None, :, :],
            "agent": self.agent_image[None, :, :],
            "target": self.target_image_transposed,
        }

    # 前回からの差分が、絵をどれだけ完成に近づけた or 遠ざけたかを評価する
    def _calc_reward(self):
        # 白が255, 黒が0なので`前回 - 今回`で正しい
        書き足した部分 = self.previous_agent_image - self.agent_image
        書き足した部分_正規化 = 書き足した部分 / 255
        書き足した部分の中でお手本にある部分 = (
            書き足した部分_正規化 * self.target_image_inverted
        )
        reward = sum(sum(書き足した部分の中でお手本にある部分 / 255))
        return (reward, 書き足した部分, 書き足した部分の中でお手本にある部分)

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
        self.previous_agent_image = self.agent_image
        self.absdiff = self._get_absdiff()
        self.previous_mean_diff = np.mean(self.absdiff[None, :, :])

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
        (x, y, distance) = action
        x = x * self.size - self.size / 2
        # おまじない程度だが、アクションはCNNに合わせて左手系になっていると解釈し、step内で右手系に直す
        y = self.size / 2 - y * self.size
        distance = distance * self.size / 2

        self.agent.teleport(x, y)
        self.agent.pencolor(0, 0, 0)
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
        self.absdiff = cv2.absdiff(self.agent_image, self.target_image)

        reward, 書き足した部分, 書き足した部分の中でお手本にある部分 = (
            self._calc_reward()
        )

        observation = self._get_obs()
        terminated = self.current_step >= self.max_step

        self.previous_agent_image = self.agent_image
        return (
            observation,
            reward,
            terminated,
            False,
            {
                "書き足した部分": 書き足した部分,
                "書き足した部分の中でお手本にある部分": 書き足した部分の中でお手本にある部分,
            },
        )

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
            書き足した部分 = self.locals["infos"][0]["書き足した部分"]
            Image.fromarray(書き足した部分, mode="L").save(
                f"{self.log_dir}/書き足した部分_{self.n_calls}.png"
            )
            書き足した部分の中でお手本にある部分 = self.locals["infos"][0][
                "書き足した部分の中でお手本にある部分"
            ]
            Image.fromarray(書き足した部分の中でお手本にある部分, mode="L").save(
                f"{self.log_dir}/書き足した部分の中でお手本にある部分{self.n_calls}.png"
            )

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


target_image_path = "data/bw.bmp"
env = DrawingEnv(target_image_path)

policy_type = "MultiInputPolicy"
action_noise = NormalActionNoise(
    mean=np.zeros(env.action_space_degree),
    sigma=0.3 * np.ones(env.action_space_degree),
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

model.save("td3_drawing")

del model  # remove to demonstrate saving and loading

model = TD3.load(
    "td3_drawing",
    action_noise=NormalActionNoise(
        mean=np.zeros(env.action_space_degree),
        sigma=np.zeros(env.action_space_degree),
    ),
)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
