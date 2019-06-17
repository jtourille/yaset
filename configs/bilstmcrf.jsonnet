local data_dir = "/path/to/data";

{
    // Model type. See Lample et al. (2016) or Ma and Hovy (2016) for further information
    "model_name": "bilstmcrf",

    // Activate testing mode, reduce dataset size (10x factor)
    "testing": true,

    "data": {
        // Path to your train and test files (in CoNLL format)
        "train_file": data_dir + "/train.conll",
        "dev_file": data_dir + "/dev.conll",

        // Label format: you can choose between BIO, IOB1, BIOUL, BMES
        // Please ensure the consistency of your dataset as no check will be performed (for now)
        "format": "BIOUL"
    },
    "network_structure": {
        "nb_layers": 2, // Number of LSTM layers
        "hidden_size": 512, // LSTM hidden size
        "cell_size": 1024, // LSTM cell size
        "skip_connections": true, // Do you want to use skip connections?
        "ffnn": {
            "use": true, // Do you want to use a feed forward neural network before projection and classification?
            "hidden_layer_size": "auto", // FFNN hidden size
            "activation_function": "relu" // You can choose between relu and tanh
        }
    },
    "training": {
        "input_dropout_rate": 0.5,
        "lstm_layer_dropout_rate": 0.2,
        "ffnn_input_dropout_rate": 0.2,
        "optimizer": "adam",
        "lr_rate": 0.001,
        "max_iterations": 1,
        "patience": 1,
        "cuda": false,
        "train_batch_size": 32,
        "clip_grad_norm": 5.0,
        "test_batch_size": 32,
        "num_global_workers": 12,
        "num_dataloader_workers": 4,
        "lr_scheduler": {
            "use": true,
            "mode": "max",
            "factor": 0.5,
            "patience": 5,
            "verbose": true,
            "threshold": 0.0001,
            "threshold_mode": "rel"
        }
    },
  "embeddings": {
    "pretrained": {
      "use": true,
      "format": "glove", // gensim or glove
      "model_path": data_dir + "/glove/glove.6B.300d.txt"
    },
    "characters": {
      "use": true,
      "type": "cnn",
      "char_embedding_size": 25,
      "cnn_filters": [
        [3, 32],
        [4, 32],
        [5, 32],
        [6, 32]
      ]
    },
    "elmo": {
      "use": false,
      "weight_path": data_dir + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
      "options_path": data_dir + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    },
    "bert":{
        "use": false,
        "fine_tune": true,
        "type": "tensorflow",
        "do_lower_case": false,
        "model_root_dir": data_dir + "/bert/biobert_v1.1_pubmed",
        "model_file": "model.ckpt-1000000.index",
        "vocab_file": "vocab.txt",
        "config_file": "bert_config.json",
        "only_final_layer": false
    },
//    "bert":{
//        "use": true,
//        "fine_tune": true,
//        "type": "pytorch",
//        "do_lower_case": false,
//        "model_root_dir": data_dir + "/bert/pytorch/bert-base-cased",
//        "model_file": "pytorch_model.bin",
//        "vocab_file": "bert-base-cased-vocab.txt",
//        "config_file": "bert_config.json",
//        "only_final_layer": false
//    },
  }
}