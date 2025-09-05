import fairseq
import sentencepiece
from torch.serialization import add_safe_globals

add_safe_globals(
    [
        fairseq.data.dictionary.Dictionary,
        fairseq.data.encoders.sentencepiece_bpe.SentencepieceBPE,
        sentencepiece.SentencePieceProcessor,
    ]
)

ckpt_path = "models/hubert_large_ll60k_finetune_ls960.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
