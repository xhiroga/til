import json
import os

import google.generativeai as genai
from datasets import dataset_dict, load_dataset
from dotenv import load_dotenv
from typing_extensions import TypedDict


class Result(TypedDict):
    task_id: int
    input: str
    output: str


class TaskScore(TypedDict):
    task_id: int
    score: int


TaskScores = list[TaskScore]


class Evaluation(TypedDict):
    task_id: int
    input: str
    output: str
    eval_aspect: str
    score: int


load_dotenv()

# Get Gemini API Key from https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def eval(
    results: list[Result], eval_dataset: dataset_dict.DatasetDict, batch_size: int = 5
) -> list[Evaluation]:
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    evaluations = []
    for i in range(0, len(results), batch_size):
        batch_results = results[i : i + batch_size]
        batch_eval_aspects = eval_dataset["test"]["eval_aspect"][i : i + batch_size]

        # TODO: からあげさんのプロンプトのエッセンスを取り入れる。
        prompts = [
            f"Evaluate the following result and provide a score between 0 and 5:\nTask ID: {result['task_id']}\n, Question: {result['input']}\nAnswer: {result['output']}\nEvaluation Aspect: {eval_aspect}"
            for result, eval_aspect in zip(batch_results, batch_eval_aspects)
        ]

        response = model.generate_content(
            prompts,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=TaskScores
            ),
        )
        scores = json.loads(response.parts[0].text)

        for result, eval_aspect, score in zip(
            batch_results, batch_eval_aspects, scores
        ):
            if result["task_id"] == score["task_id"]:
                evaluations.append(
                    {
                        **result,
                        "eval_aspect": eval_aspect,
                        "score": score["score"],
                    }
                )
            else:
                print(f"task_id mismatch, {result['task_id']=} != {score['task_id']=}")

    return evaluations


if __name__ == "__main__":
    sample_results = [
        {
            "task_id": 0,
            "input": "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。",
            "output": "仕事への情熱が薄れてしまった場合、以下のような方法でやる気を取り戻せます。\n\n1.目標設定：自分に合った具体的な目標を立てることで、モチベーションアップにつながる。\n2.タスク管理：やるべきことを明確化することで、優先順位をつけやすくなる。\n3.コミュニケーション：同僚と積極的に会話することで信頼関係も深まり、より良い人間関係を作ることができる。\n4.リフレッシュ休暇：長期休みをとって気分転換し、新たな気持ちで仕事に戻ることで、",
        },
        {
            "task_id": 1,
            "input": "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。",
            "output": "クマは海辺でアザラシと一緒に遊んでいました。二人はとても仲良しで、一緒に泳いだり、魚釣りをしたりして遊びました。しかし、ある日、クマはアザラシのことを忘れてしまい、一人で帰ってしまいました。その後、クマはまたアザラシのところへ行きましたが、もうアザラシはいませんでした。クマは、自分も寂しくて悲しくなってしまいました。そして、クマは自分の家へと帰りました。\n  この物語では、クマがアザラシとの友情を失い、寂しさを感じた後に、再びアザラシのもとに戻ることを決意するところがポイントです。また、クマが",
        },
        {
            "task_id": 2,
            "input": "古代ギリシャを学ぶ上で知っておくべきポイントは？",
            "output": "古代ギリシャは紀元前8世紀から紀元後300年頃まで続いた文明です。そのため、様々な時代や地域の文化が混ざり合っています。また、当時の人々は、政治、宗教、芸術などあらゆる分野において独自性を持っていました。ここでは、その中でも特に重要な点について説明します。\n\n1.　アテネとスパルタ：アテネとスパルタは共にポリス（都市国家）であり、世界最古の民主主義国として有名ですが、この2つの国には",
        },
        {
            "task_id": 3,
            "input": "以下の説明を読んだユーザーが感じるであろう疑問とは？\n\n大量のゴミの処理にお困りではありませんか？\n\n「生産で出るゴミ処理費用が大幅にかかっている」\n「指定ゴミ袋が導入され、一気にコストが上がってしまった」\n「工場が狭いので、ゴミ保管スペースを減らしたい」\n「ISO14001でゴミ削減が大きな課題となっている」\n\n場所をとる嵩張るゴミをつぶしてまとめて運びやすい形にすることで、処理・保管・運搬にかかる様々なコストを削減したり、環境負荷の低減にも貢献できるのが、廃棄物圧縮減容機→『ゴミ圧縮機』です。\n\nゴミ圧縮機の主な用途は廃プラスチック・紙・段ボール・袋類・ペットボトル・プラスチック容器などつぶれるが元に戻ってしまうゴミを圧縮梱包します。",
            "output": "与えられた文章から考えられる、ユーザーが持つと思われる疑問は、以下のようなものがあります。\n\n\n１．大量のゴミの処理にお困りではないですか？\n\n　この質問に対しては、回答者が、どのような種類のゴミを持っているのかによって答え方が変わってきます。例えば、家庭からの生ごみや食品残渣などの場合には、自治体による分別収集制度があり、そのルールにしたがって処分すればよいので問題はありません。しかし、産業系のゴミの場合には、その量も多く、また多種多様",
        },
        {
            "task_id": 4,
            "input": "次に読むべきSFを10冊挙げて",
            "output": "以下は私が読んだ中で特に面白かったSF小説です。\n\n　1.「星新一ショートショート」（新潮文庫）：短編の名手である星新一が書いた短編集。短い話の中に、様々なテーマやアイデアがあり、とても面白い作品ばかりでした。\n\n2.「銀河英雄伝説」（創元SF文庫）：田中芳樹による壮大なスペースオペラで、宇宙戦争と政治・外交を描いた物語です。\n\n3.「一九八四年[新",
        },
        {
            "task_id": 5,
            "input": "次の文章を読んで、その人がどの程度怒っているか、1〜10の尺度で評価してください。(1＝怒っていない、10＝非常に怒っている)。\n\n1. またテストで赤点か。君は全く...\n2. テストで赤点？今回は難しかったね。",
            "output": "与えられた文章から、この人は「またテストで赤点か」と言っていますが、これは怒りというよりは、呆れているように感じます。また、この人はこの人の立場や状況が分かっていてもいるので、その人に向かって、「今回のは難しいね」とは言えないでしょう。つまり、相手のことを考えて言葉をかけているのです。\n\nよって、この人は１～３くらいの間だと思われます。\n\n以下に、それぞれの感情を表す表現を挙げますので、参考にして下さい。\n\n・怒らない：　I",
        },
    ]
    ds = load_dataset("elyza/ELYZA-tasks-100")

    evaluations = eval(sample_results, ds, 5)
    averagt_score = sum(evaluation["score"] for evaluation in evaluations) / len(
        evaluations
    )
    print(f"{evaluations=}, {averagt_score=}")
    assert 2 < averagt_score < 5
