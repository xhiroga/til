import os

from atproto import Client
from dotenv import load_dotenv

load_dotenv(verbose=True)

HANDLE = os.getenv('HANDLE')
PASSWORD = os.getenv('PASSWORD')

def generate_reply(text):
    


def main():
    client = Client()
    client.login(HANDLE, PASSWORD)

    # TODO: 最後に取得した日付を指定する
    timeline = client.bsky.feed.get_timeline({'algorithm': 'reverse-chronological'})
    for feed_view in timeline.feed:
        # feed_view.post.record.facets の中の faset.features の中に {"did": "did:plc:d7mnkzaznaop33oiowcbco7g","$type": "app.bsky.richtext.facet#mention"} があれば、aibot向けのメンションとして扱う
        
        # feed_view.post.record.text の文章をChatGPTに投げて、返答を生成する


        post = feed_view.post.record
        author = feed_view.post.author

        print(
            f'[{action} by {action_by}] Post author: {author.displayName}. Posted at {post.createdAt}. Post text: {post.text}'
        )


if __name__ == '__main__':
    main()
