local transformer_dim = 512;
local transformer_emb_dim = 768;

local cuda_device = 0;

local env_or_default(env_name, default_value) =
    local env_value = std.extVar(env_name);
    if env_value == "" then default_value else env_value;

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local transformer_model = std.extVar("transformer_model");
local batch_size = std.parseInt(env_or_default("BATCH_SIZE", "4"));
local epochs = std.parseInt(env_or_default("EPOCHS", "10"));

local scheduler = "slanted_triangular";

local slanted_triangular_scheduler = {
    "type": "slanted_triangular",
    "cut_frac": 0.06
};

local learning_rate_scheduler = slanted_triangular_scheduler;

local encoder_type = "pt_encoder";

local pt_encoder = {
    "type": "pass_through",
    "input_dim": transformer_emb_dim
};

local encoder = pt_encoder;

local dropout = std.parseJson(std.extVar("DROPOUT"));
local learning_rate = std.parseJson(std.extVar("lr"));
local learning_rate_ner = std.parseJson(std.extVar("lr_ner"));
local gradient_accumulation_steps = std.parseInt(env_or_default("GRADIENT_ACCUMULATION_STEPS", "4"));

local wandb_name = std.extVar("WANDB_NAME");

local tuning_callbacks = [
    {
        "type": "tensorboard",
        "should_log_learning_rate": true
    }
];

local callbacks = tuning_callbacks;

local tokenizer_kwargs = {
        "max_len": transformer_dim,
        "use_auth_token": false
    };

local tokenizer_kwargs = {
        "use_auth_token": false
    };

local token_indexer = {
        "type": "pretrained_transformer_mismatched",
        "max_length": transformer_dim,
        "model_name": transformer_model,
        "tokenizer_kwargs": tokenizer_kwargs
    };

local conll_reader = {
    "type": "conll2003",
    "convert_to_coding_scheme": "BIOUL",
    "tag_label": "ner",
    "token_indexers": {
        "tokens": token_indexer
    }
};

local dataset_folder = std.extVar("DATASET_PATH");
local seed = std.parseJson(std.extVar('seed'));
local fold = std.parseJson(std.extVar('fold'));
local dataset_reader = conll_reader;
local version = std.extVar("VERSION");
local base_path = dataset_folder + "/fold-" + fold;
local train_path = base_path + "/train.conll";
local dev_path = base_path + "/dev.conll";
local test_path = base_path + "/test.conll";
local evaluate_on_test = true;

{
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "random_seed": seed,
    "dataset_reader": dataset_reader,
    "train_data_path": train_path,
    "validation_data_path": dev_path,
    "test_data_path": test_path,
    "evaluate_on_test": evaluate_on_test,
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "dropout": dropout,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "max_length": transformer_dim,
                    "model_name": transformer_model,
                    "tokenizer_kwargs": tokenizer_kwargs,
                    "transformer_kwargs": tokenizer_kwargs
                }
            }
        },
        "encoder": encoder,
        "regularizer": {
          "regexes": [
            [
                "scalar_parameters",
                {
                    "type": "l2",
                    "alpha": 0.1
                }
            ]
          ]
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : batch_size,
            "sorting_keys": [
                "tokens"
            ]
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0,
            "lr": learning_rate,
            "parameter_groups": [
                [
                    ["^text_field_embedder(?:\\.(?!(LayerNorm|bias))[^.]+)+$"],
                    {"weight_decay": 0, "lr": learning_rate}
                ],
                [
                    ["^text_field_embedder\\.[\\S]+(LayerNorm[\\S]+|bias)$"],
                    {"weight_decay": 0, "lr": learning_rate}
                ],
                [
                    ["encoder._module", "tag_projection_layer", "crf"],
                    {"weight_decay": 0, "lr": learning_rate_ner}
                ]
            ]
        },
        "callbacks": callbacks,
        "learning_rate_scheduler": learning_rate_scheduler,
        "num_gradient_accumulation_steps": gradient_accumulation_steps,
        "cuda_device": cuda_device,
        "num_epochs": epochs,
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "patience": 5,
        "validation_metric": "+f1-measure-overall"
    }
}
