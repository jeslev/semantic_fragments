#!/bin/sh
#SBATCH --job-name=ext_robraw

#SBATCH --output=./runs/robraw.output
#SBATCH --error=./runs/robraw.error

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=RTX6000Node
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jesus.lovon@irit.fr

SCRIPT_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_LOC="$( dirname $SCRIPT_LOC )"

MODEL_DIR="${LIB_LOC}/_experiments/roberta_raw" # to store models
GPUNUM=0
OUTPUT_DIR="${MODEL_DIR}" # to store outputs

train_config="${LIB_LOC}/_experiments/config.jsonnet" # params
TRAIN_PATH=
VALID_PATH=
PREMODEL=

export PYTHONPATH=.
_container="/logiciels/containerCollections/CUDA10/pytorch.sif"
_python="${HOME}/miniconda3/envs/roberta/bin/python"

cd ./etc/allennlp-transf-exp1

srun singularity exec ${_container} ${_python} -m allennlp.run train ${train_config} -s ${MODEL_DIR} -o "{'model': {'transformer_weights_model': '${PREMODEL}', 'reset_classifier':true }, 'train_data_path': '${TRAIN_PATH}', 'validation_data_path': '${VALID_PATH}'}" --file-friendly-logging
wait


for ds in "definitions" "hypernymy" "hyponymy" "synonymy" "dictionary_qa"; do
    dataset=${LIB_LOC}/data_mcqa/$ds

    echo "now running ${dataset}"
    srun singularity exec ${_container} ${_python} -m allennlp.run evaluate_custom \
           --evaluation-data-file ${dataset}/test.jsonl \
           --metadata-fields "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs,choice_context_list" \
           --output-file ${OUTPUT_DIR}/result_${ds}.jsonl \
           --cuda-device 0 \
           --overrides '{"iterator":{"batch_size":32}}' \
           ${MODEL_DIR}
    wait
done
