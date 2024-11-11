from models.baseline.nlp.fnn import FNN_VirusHostPrediction
from models.baseline.nlp.cnn1d import CNN_1D_VirusHostPrediction
from models.baseline.nlp.rnn import RNN_VirusHostPrediction
from models.baseline.nlp.lstm import LSTM_VirusHostPrediction
from models.baseline.nlp.transformer_encoder import TransformerEncoderVirusHostPrediction

from models.virprobert import VirProBERT
from models.virprobert_wo_hierattn import VirProBERT_wo_HierAttn

from transfer_learning.fine_tuning.bert_virus_host_prediction import BERT_VirusHostPrediction
from models.external.prost5_host_prediction import ProstT5_VirusHostPrediction
from datasets.protein_sequence_custom_dataset import ProteinSequenceProstT5Dataset
# mappings of all classes
model_map = {

    "FNN": FNN_VirusHostPrediction,
    "CNN": CNN_1D_VirusHostPrediction,
    "RNN": RNN_VirusHostPrediction,
    "LSTM": LSTM_VirusHostPrediction,
    "Transformer_Encoder": TransformerEncoderVirusHostPrediction,
    "BERT": BERT_VirusHostPrediction,
    "VirProBERT_wo_Hierarchical_Attention": VirProBERT_wo_HierAttn,
    "VirProBERT": VirProBERT,

    "ProstT5": ProstT5_VirusHostPrediction

}

dataset_map = {
    "ProstT5": ProteinSequenceProstT5Dataset
}