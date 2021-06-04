{
    "dataset_reader": {
        "type": "transformer_mc_qa",
        "context_syntax": "q#_a!",
        "max_pieces": 256,
        "num_choices": 5,
        "pretrained_model": "roberta-large",
        "sample": -1
    },
    "iterator": {
        "type": "basic",
        "batch_size": 1
    },
    "model": {
        "type": "roberta_mc_qa",
        "pretrained_model": "roberta-large",
        "transformer_weights_model":"",
        "reset_classifier": true
    },
    "train_data_path": "../data_mcqa/science_data/train.jsonl",
    "validation_data_path": "../data_mcqa/science_data/dev.jsonl",
    "test_data_path": "../data_mcqa/science_data/larger_dev.jsonl",
    "trainer": {
        "cuda_device": 0,
        "gradient_accumulation_batch_size": 16,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac": 0.06,
            "num_epochs": 4,
            "num_steps_per_epoch": 2163
        },
        "num_epochs": 4,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "adam_w",
            "lr": 1e-05,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm\\.weight",
                        "layer_norm\\.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "weight_decay": 0.1
        },
        "should_log_learning_rate": true,
        "validation_metric": "+accuracy"
    },
    "datasets_for_vocab_creation": [],
    "evaluate_custom": {
        "metadata_fields": "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs"
    },
    "evaluate_on_test": true
}
