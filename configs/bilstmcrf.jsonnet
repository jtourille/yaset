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

        // Label format: you can choose between IOB1, IOB2, IOBES
        // Please ensure the consistency of your dataset as no check will be performed (for now)
        "format": "IOBES"
    },
    "network_structure": {
        "input_dropout_rate": 0.2,

        "lstm": {
            "nb_layers": 2, // Number of LSTM layers
            "hidden_size": 512, // LSTM hidden size
            "layer_dropout_rate": 0.5,
            "highway": true, // Do you want to use highway connections?
        },

        "ffnn": {
            "use": true, // Do you want to use a feed forward neural network before projection and classification?
            "hidden_layer_size": "auto", // FFNN hidden size
            "activation_function": "relu", // You can choose between relu and tanh
            "input_dropout_rate": 0.2
        }
    },
    "training": {
        "optimizer": "adam", # ["adam", "adamw"]
        "weight_decay": 0.0,
        "lr_rate": 0.001,
        "fp16": false,
        "fp16_level": "O1",

        "max_iterations": 100,
        "patience": 10,
        "cuda": true,
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
      "format": "w2v", // gensim or glove
      "model_path": data_dir + "/glove/glove.6B/glove.6B.300d.txt",
      "singleton_replacement_ratio": 0.2
    },
    "chr_cnn": {
      "use": true,
      "type": "type2",
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
      "fine_tune": false,
      "weight_path": data_dir + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
      "options_path": data_dir + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    },
    "bert":{
        "use": false,
        "fine_tune": false,
        "type": "pytorch",
        "do_lower_case": true,
        "model_file": data_dir + "/bert/pytorch/bert-base-uncased-pytorch_model.bin",
        "vocab_file": data_dir + "/bert/pytorch/bert-base-uncased-vocab.txt",
        "config_file": data_dir + "/bert/pytorch/bert-base-uncased-config.json",
        "only_final_layer": false
    }
  }
}