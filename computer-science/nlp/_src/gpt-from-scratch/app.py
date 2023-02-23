import numpy as np

def gpt(inputs: list[int]) -> list[list[float]]:
    # inputs has shape [n_seq]
    # outputの形状は、単語jがi+1文字目に現れる確率。
    # （つまり、語彙応じて行方向にものすごく長い2次元配列）
    # これってoutput[-1]より前の行はすでにinputで与えられているので、使わないんじゃないかな...？
    # outpus has shape [n_seq, n_vocab]
    return output

vocab = ["all", "not", "heros", "the", "wear", ".", "capes"]

# ChatGPTには、回答を何文字生成するかは情報として与えていないはずだけど、どうしているんだろう？
def generate(inputs, n_tokens_to_generate):
    outputs = []
    for _ in range(n_tokens_to_generate):
        output = gpt(inputs + outputs)
        next_id = np.argmax(output[-1])
        outputs.append(int(next_id))
    return outputs

def lm_loss(inputs: list[int], params) -> float:
    x, y = inputs[:-1], inputs[1:]
    output = gpt(x, params) # あれ、paramsって何？
    loss = np.mean(-np.log(output[y])) # ちょっと書き方がよくわからないけど、要は既存の文章で訓練させている。
    return loss

def train(texts: list[list[str]], params) -> float:
    for text in texts:
        inputs = tokenizer.encode(text)
        loss = lm_loss(inputs, params)
        gradients = compute_gradients_via_backpropagation(loss, params)
        params = gradient_decent_update_step(gradients, params)
    return params

