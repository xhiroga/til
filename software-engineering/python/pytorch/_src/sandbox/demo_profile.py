import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torchvision

parser = argparse.ArgumentParser(description="Profile ResNet50 model")
parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions")
parser.add_argument("--logdir", type=str, default="logs", help="Log directory")
args = parser.parse_args()

os.makedirs(args.logdir, exist_ok=True)

model = torchvision.models.resnet50()
model = model.to("cuda")
model.eval()

input_image = torch.ones((512, 3, 224, 224))
input_image = input_image.to("cuda")


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        # use_cuda=True に代わって推奨
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
    # 以下の設定で 14MB -> 11MB 程度まで削減できる。
    experimental_config=torch.profiler._ExperimentalConfig(
        profiler_measure_per_kernel=False,
        verbose=False,
        performance_events=[],
        enable_cuda_sync_events=False,
    ),
) as prof:
    # 出力されるJSONのトップレベルのプロパティに `"args": {"repeat": 10},` が追加される。
    prof.add_metadata_json("args", json.dumps(vars(args), default=str))
    with torch.no_grad():
        for i in range(args.repeat):
            # `torch.cuda.profiler.profile()` ではなく `torch.profiler.profile()` を使う場合は NVTXマーカーを手動で追加する。
            torch.cuda.nvtx.range_push(f"iteration_{i}")
            with torch.profiler.record_function("model_inference"):
                output = model(input_image)
            torch.cuda.nvtx.range_pop()

# prof オブジェクトに with ブロック外でアクセスしているが、これが正解。
# ブロック内でアクセスすると何も表示されない。
# なお、`schedule`を渡している場合は profile() が返すインスタンスがシグニチャ違いの別クラスになるので、呼び出す際に一手間必要そう。。
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# おそらく `on_trace_ready=torch.profiler.tensorboard_trace_handler(args.logdir)` で実装しても内容は同じ？ファイル名を指定するならこちら。
trace_path = Path(args.logdir) / f"{os.uname().nodename}-{Path(__file__).stem}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt.trace.json"
os.makedirs(trace_path.parent, exist_ok=True)
prof.export_chrome_trace(str(trace_path))
