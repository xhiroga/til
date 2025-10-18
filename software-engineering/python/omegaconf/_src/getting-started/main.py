import sys
from omegaconf import DictConfig, ListConfig, OmegaConf

from dataclasses import dataclass


@dataclass
class MyConfig:
    port: int = 80
    host: str = "localhost"


def main():
    # 初回作成は`create`だが、以降はソースに応じて`load`や`structured`など命名に統一感がない。
    conf: DictConfig | ListConfig = OmegaConf.create()
    print(OmegaConf.to_yaml(conf))

    conf = OmegaConf.load("source/example.yaml")
    print(OmegaConf.to_yaml(conf))

    dot_list = ["a.aa.aaa=1", "a.aa.bbb=2", "a.bb.aaa=3", "a.bb.bbb=4"]
    conf = OmegaConf.from_dotlist(dot_list)
    print(OmegaConf.to_yaml(conf))

    # 1つ目の引数はスクリプト名なので解釈されない。
    # 個人的な印象としては、勝手に sys.argv を読むのは行儀が悪い...
    sys.argv = ["your-program.py", "server.port=82", "log.file=log2.txt"]
    conf = OmegaConf.from_cli()
    print(OmegaConf.to_yaml(conf))

    # `structured()` は以下のような型ヒントを提供できるが、これは DuckTyping であり実際の方は DictConfig である (!??)
    # https://omegaconf.readthedocs.io/en/latest/structured_config.html#static-type-checker-support
    base_conf: MyConfig = OmegaConf.structured(MyConfig)
    print(OmegaConf.to_yaml(base_conf))
    # なお、omegaconf.structured() で作成した設定は structured config と呼ばれる状態になる。
    print(f"{OmegaConf.get_type(base_conf)=} vs {OmegaConf.get_type(conf)=}")

    # log_cfg = OmegaConf.create({"log": {"file": "log.txt"}})
    # merged_conf = OmegaConf.merge(base_conf, log_cfg)
    # print(OmegaConf.to_yaml(merged_conf))

if __name__ == "__main__":
    main()
