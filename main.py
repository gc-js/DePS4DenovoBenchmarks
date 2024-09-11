import deepnovo_config
from train_func import build_model
from data_reader import DeepNovoDenovoDataset
from model import InferenceModelWrapper
from denovo import IonCNNDenovo
from writer import DenovoWriter

data_reader = DeepNovoDenovoDataset(feature_filename=deepnovo_config.denovo_input_feature_file,
                                    spectrum_filename=deepnovo_config.denovo_input_spectrum_file)
denovo_worker = IonCNNDenovo(deepnovo_config.MZ_MAX,
                                deepnovo_config.knapsack_file,
                                beam_size=deepnovo_config.args.beam_size)
forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
writer = DenovoWriter(deepnovo_config.denovo_output_file)
denovo_worker.search_denovo(model_wrapper, data_reader, writer)
