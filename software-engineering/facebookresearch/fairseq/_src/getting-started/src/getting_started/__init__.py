from fairseq.models import FairseqEncoderDecoderModel, register_model
from getting_started.models.simple_lstm import SimpleLSTMEncoder, SimpleLSTMDecoder

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model("simple_lstm")
class SimpleLSTMModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="dimensionality of the encoder embeddings",
        )
        parser.add_argument(
            "--encoder-hidden-dim",
            type=int,
            metavar="N",
            help="dimensionality of the encoder hidden state",
        )
        parser.add_argument(
            "--encoder-dropout",
            type=float,
            default=0.1,
            help="encoder dropout probability",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="dimensionality of the decoder embeddings",
        )
        parser.add_argument(
            "--decoder-hidden-dim",
            type=int,
            metavar="N",
            help="dimensionality of the decoder hidden state",
        )
        parser.add_argument(
            "--decoder-dropout",
            type=float,
            default=0.1,
            help="decoder dropout probability",
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        encoder = SimpleLSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        decoder = SimpleLSTMDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout,
        )
        model = SimpleLSTMModel(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model

    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
    #     return decoder_out


from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.


@register_model_architecture("simple_lstm", "tutorial_simple_lstm")
def tutorial_simple_lstm(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_hidden_dim = getattr(args, "encoder_hidden_dim", 256)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 256)
