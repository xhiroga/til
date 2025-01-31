import numpy as np


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    """
    Parameters
    ---
    inputs: 
    
    wte: 
        Word Token Embeddingsの略と思われる。
        これも学習対象だと思っていたら、引数で渡されるのでよくわからない。

    wpe:
        Word Position Embeddingsの略と思われる。
        これも外から渡されるの、よく分からないな...
    """

    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm   # アラビア語のtaqadum(進歩・前進)に由来する、進捗バーを簡単に表示するライブラリ

    for _ in tqdm(range(n_tokens_to_genarate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs) - n_tokens_to_generate :]

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # encoder: 高次元のデータを低次元のデータに処理する。テキストをベクトルにする。
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # アルファベット（およびそれ以外の文字）の配列として表せるテキストを、文字単位ではなく単語単位にする（BPE Tokenization）
    # 要するに頻繁に現れる文字のペアを新たな語彙として登録する。Asciiでは１文字バイトなので、Byte Pair Encodingという名前なのだろう。
    input_ids = encoder.encode(prompt)

    assert len(inputs_ids) + n_tokens_to_genarete < hparams["n_ctx"]

    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    output_text = encoder.decode(output_ids)

    return output_text

if __name__ == "__main__":
    import fire

    fire.Fire(main)
