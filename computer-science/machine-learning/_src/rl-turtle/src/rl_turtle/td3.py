import datetime
import io
import os
import turtle
from colorsys import hsv_to_rgb

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
        # 公式には無かったが必要なのか？
        super(DrawingEnv, self).__init__()

        self.size = size
        self.agent_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        image = Image.open(target_image).convert("RGB")

        self.target_image = np.array(image.resize((self.size, self.size)))
        self.target_image_transposed = self.target_image.transpose((2, 0, 1))
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=0, high=255, shape=(3, self.size, self.size), dtype=np.uint8
                ),
                "target": gym.spaces.Box(
                    low=0, high=255, shape=(3, self.size, self.size), dtype=np.uint8
                ),
            }
        )

        self.action_space_degree = 6
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
            "agent": self.agent_image.transpose((2, 0, 1)),
            "target": self.target_image_transposed,
        }

    # 非常に難しい！
    def _calc_reward(self):
        img1_hsv = cv2.cvtColor(self.agent_image, cv2.COLOR_RGB2HSV)
        img2_hsv = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2HSV)
        # 色相は0~180なのに、彩度と明度は0~255なので、単に平均を取っていいのか？という話はある
        diff = cv2.absdiff(img1_hsv, img2_hsv)
        mean_diff = np.mean(diff)
        improvement = mean_diff - self.previous_mean_diff
        self.previous_mean_diff = mean_diff
        # TODO: 目標達成したときのrewardもどこかで加えたほうが良い
        return improvement if improvement > 0 else -mean_diff

    def reset(self, seed=None, options=None):
        self.screen.clearscreen()
        self.agent.reset()
        self.agent.hideturtle()
        self.agent.speed(0)
        self._show_drawing_area()
        self.agent.teleport(0, 0)
        self.agent.pensize(self.size // 16 + 1)  # これも勘
        self.current_step = 0
        self.previous_mean_diff = 255

        self.agent_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        return self._get_obs(), {}

    # マジで便利、絶対実装したほうがいい。学習が膠着したときにフリーズしてるのかどうか、枠の再描画でひと目で判断できる
    def _show_drawing_area(self):
        self.agent.teleport(-(self.size / 2), -(self.size / 2))
        self.agent.pensize(1)
        for _ in range(4):
            self.agent.forward(self.size)
            self.agent.left(90)

    def step(self, action):
        (x, y, distance, h, s, v) = action
        x = x * self.size - self.size / 2
        y = y * self.size - self.size / 2
        # action spaceをHSVで定義したほうが明度が似る
        r, g, b = hsv_to_rgb(h, s, v)
        distance = distance * self.size

        self.agent.teleport(x, y)
        self.agent.pencolor(r, g, b)
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
        # 念のためresizeしているがおそらく不要
        self.agent_image = np.array(
            Image.open(io.BytesIO(ps.encode("utf-8"))).resize((self.size, self.size))
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
        # `_on_step` は `reset` 呼び出し後に呼ばれるため、 `% 16 == 0` だと真っ黒な画面がログされる
        if self.n_calls % 16 == 15:
            new_obs = self.locals["new_obs"]
            agent_image = new_obs["agent"]
            if isinstance(agent_image, np.ndarray):
                img_array = agent_image[0].transpose((1, 2, 0)).astype(np.uint8)
                img = Image.fromarray(img_array)
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


target_image_path = "data/star.bmp"
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
    name="reward=(improvement if improvement > 0 else -mean_diff), action=(x, y, distance, h, s, v)",
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
# model.save("td3_drawing")

# del model  # remove to demonstrate saving and loading

# model = TD3.load("td3_drawing")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
