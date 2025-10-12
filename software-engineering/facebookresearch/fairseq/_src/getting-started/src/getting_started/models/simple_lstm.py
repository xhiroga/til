import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder


class SimpleLSTMEncoder(FairseqEncoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_dim=128,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__(dictionary)
        self.args = args

        # Our encoder will embed the inputs before feeding them to the LSTM.
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, src_tokens, src_lengths):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We discuss Tasks in the next tutorial, but for now just
        # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
        # has shape `(batch)`.

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.
        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens, padding_idx=self.dictionary.pad(), left_to_right=True
            )

        # Embed the source.
        x = self.embed_tokens(src_tokens)

        # Apply dropout.
        x = self.dropout(x)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)

        # Get the output from the LSTM.
        _outputs, (final_hidden, _final_cell) = self.lstm(x)

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        return {
            # this will have shape `(bsz, hidden_dim)`
            "final_hidden": final_hidden.squeeze(0),
        }

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        final_hidden = encoder_out["final_hidden"]
        return {
            "final_hidden": final_hidden.index_select(0, new_order),
        }


import torch
from fairseq.models import FairseqDecoder


class SimpleLSTMDecoder(FairseqDecoder):
    def __init__(
        self,
        dictionary,
        encoder_hidden_dim=128,
        embed_dim=128,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__(dictionary)

        # Our decoder will embed the inputs before feeding them to the LSTM.
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # We'll use a single-layer, unidirectional LSTM for simplicity.
        self.lstm = nn.LSTM(
            # For the first layer we'll concatenate the Encoder's final hidden
            # state with the embedded target tokens.
            input_size=encoder_hidden_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )

        # Define the output projection.
        self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.
    def forward(self, prev_output_tokens, encoder_out):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        bsz, tgt_len = prev_output_tokens.size()

        # Extract the final hidden state from the Encoder.
        final_encoder_hidden = encoder_out["final_hidden"]

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(prev_output_tokens)

        # Apply dropout.
        x = self.dropout(x)

        # Concatenate the Encoder's final hidden state to *every* embedded
        # target token.
        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
            dim=2,
        )

        # Using PackedSequence objects in the Decoder is harder than in the
        # Encoder, since the targets are not sorted in descending length order,
        # which is a requirement of ``pack_padded_sequence()``. Instead we'll
        # feed nn.LSTM directly.
        initial_state = (
            final_encoder_hidden.unsqueeze(0),  # hidden
            torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
        )
        output, _ = self.lstm(
            x.transpose(0, 1),  # convert to shape `(tgt_len, bsz, dim)`
            initial_state,
        )
        x = output.transpose(0, 1)  # convert to shape `(bsz, tgt_len, hidden)`

        # Project the outputs to the size of the vocabulary.
        x = self.output_projection(x)

        # Return the logits and ``None`` for the attention weights
        return x, None
