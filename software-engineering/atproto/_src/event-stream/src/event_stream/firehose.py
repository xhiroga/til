from atproto import CAR, models
from atproto.firehose import (FirehoseSubscribeReposClient,
                              parse_subscribe_repos_message)

client = FirehoseSubscribeReposClient()


def on_message_handler(message) -> None:
    commit = parse_subscribe_repos_message(message)
    # we need to be sure that it's commit message with .blocks inside
    if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
        return

    car = CAR.from_bytes(commit.blocks)
    print(car.root, car.blocks)


client.start(on_message_handler)
