local data_dir = "/path/to/data";
{
    // Activate debug mode: enforce some paramater values to finish quickly model training
    "debug": false,

    "data": {
        // Path to your train and test files (in CoNLL format)
        "train_file": data_dir + "/train.conll",
        "dev_file": data_dir + "/dev.conll",

        // Label format: limited to IOBES (for now)
        // Please ensure the consistency of your dataset as no check will be performed (for now)
        "format": "IOBES"
    },
    "network_structure": {

        "lstm": {
            "nb_layers": 2, // Number of LSTM layers
            "hidden_size": 512, // LSTM hidden size
            "layer_dropout_rate": 0.5, // Hidden layer dropout rate
            "highway": true, // Do you want to use highway connections?
            "input_dropout_rate": 0.2, // Input dropout rate (!)
        },

        "ffnn": {
            "use": true, // Do you want to use a feed forward neural network before projection and classification?
            "hidden_layer_size": "auto", // FFNN hidden size (auto: half size of input layer)
            "activation_function": "relu", // You can choose between relu and tanh
            "input_dropout_rate": 0.2 // Input dropout rate (!)
        }
    },
    "training": {
        "optimizer": "adamw", # ["adam", "adamw"]
        "weight_decay": 0.01,
        "lr_rate": 2e-5,
        "clip_grad_norm": null, # null or float

        "cuda": true, # Use GPU for computation
        "fp16": true, # Activate mixed precision
        "fp16_level": "O1", # Mixed precision level (O1 supported)

        "train_batch_size": 8,
        "accumulation_steps": 1, # Use this parameter to get larger batch size
        "test_batch_size": 128,

        "num_global_workers": 12, # Number of processes used for computation
        "num_dataloader_workers": 4, # Number of processes used for each dataloader

        # Warmup scheduler: increase the learning rate for a given number of steps, then decrease it
        "warmup_scheduler": {
            "use": true,
            "%_warmup_steps": 0.10,
        },

        # Learning rate scheduler (plateau)
        "lr_scheduler": {
            "use": false,
            "mode": "max",
            "factor": 0.5,
            "patience": 5,
            "verbose": true,
            "threshold": 0.0001,
            "threshold_mode": "rel"
        },

        "eval_every_%": 0.20, # Dev corpus evaluation frequency (fraction of training corpus)
        "num_epochs": 10, # Maximum number of epochs to perform

    },
  "embeddings": {
    "pretrained": {
      "use": false,
      "format": "w2v", // gensim or glove
      "model_path": data_dir + "/glove/glove.6B/glove.6B.300d.txt",
      "singleton_replacement_ratio": 0.2
    },
    "chr_cnn": {
      "use": false,
      "type": "literal", // ["literal", "utf8"]
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
        "use": true,
        "fine_tune": true,
        "type": "pytorch",
        "do_lower_case": true,
        "model_file": data_dir + "/bert/pytorch/bert-base-uncased-pytorch_model.bin",
        "vocab_file": data_dir + "/bert/pytorch/bert-base-uncased-vocab.txt",
        "config_file": data_dir + "/bert/pytorch/bert-base-uncased-config.json",
        "only_final_layer": true
    }
  }
}