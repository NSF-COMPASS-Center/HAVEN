from models.virus_host_prediction_base import VirusHostPredictionBase
from models.nlp.embedding.embedding import EmbeddingLayer
from models.baseline.nlp.transformer import EncoderLayer, Encoder

class TransformerEncoderVirusHostPrediction(VirusHostPredictionBase):
    def __init__(self, vocab_size, max_seq_len, n_layers=6, input_dim=512, hidden_dim=1024, h=8, n_mlp_layers=2, n_classes=1):
        super(TransformerEncoderVirusHostPrediction, self).__init__(input_dim, hidden_dim,
                                                                    n_mlp_layers=n_mlp_layers,
                                                                    n_classes=n_classes,
                                                                    batch_norm=False)

        self.embedding = EmbeddingLayer(vocab_size=vocab_size, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), n_layers)

    def get_embedding(self, X):
        X = self.embedding(X)
        X = self.encoder(X, mask=None)

        # pool the model_params embeddings of all tokens in the input sequence using mean
        return X.mean(dim=1)

    # def forward() : use the template implementation in VirusHostPredictionBase

    def get_model(model_params) -> TransformerEncoderVirusHostPrediction:
        model = TransformerEncoderVirusHostPrediction(vocab_size=model_params["vocab_size"],
                                                      max_seq_len=model_params["max_seq_len"],
                                                      n_layers=model_params["n_layers"],
                                                      input_dim=model_params["input_dim"],
                                                      hidden_dim=model_params["hidden_dim"],
                                                      h=model_params["n_heads"],
                                                      n_mlp_layers = model_params["n_mlp_layers"],
                                                      n_classes=model_params["n_classes"])
        print(model)
        print("TransformerEncoderVirusHostPrediction: Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        return VirusHostPredictionBase.return_model(model, model_params["data_parallel"])