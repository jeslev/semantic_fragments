# Save path: runs/model_name/config_name/train_set/{model, tests}
# srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=3 --gres=gpu:1 --hint=nomultithread --account=evd@gpu bash
import os

slurm_template_train = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -A evd@gpu

#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err

#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=3
##SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_p2 
#SBATCH --gres=gpu:1
##SBATCH --time=02:00:00
#SBATCH --time=20:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jesus.lovon@irit.fr

module purge
module load cuda/10.2

SCRIPT_LOC={path_script}
LIB_LOC="$( dirname $SCRIPT_LOC )"

MODEL_DIR="${{LIB_LOC}}/_experiments/{job_name}" # to store models
GPUNUM=0
OUTPUT_DIR="${{MODEL_DIR}}" # to store outputs

train_config="${{LIB_LOC}}/_experiments/config.jsonnet" # params
TRAIN_PATH=${{LIB_LOC}}/data_mcqa/science_data/train.jsonl
VALID_PATH=${{LIB_LOC}}/data_mcqa/science_data/dev.jsonl
TEST_PATH=${{LIB_LOC}}/data_mcqa/science_data/larger_dev.jsonl
PREMODEL={premodel}


"""

slurm_static = """

_python="${WORK}/miniconda3/envs/roberta/bin/python"
export PYTHONPATH=.

if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR};
fi


cd ./etc/allennlp-transf-exp1

${_python} -u -m allennlp.run train  ${train_config} -s ${MODEL_DIR} -o "{'model': {'transformer_weights_model': '${PREMODEL}', 'reset_classifier':true }, 'train_data_path': '${TRAIN_PATH}', 'validation_data_path': '${VALID_PATH}', 'test_data_path': '${TEST_PATH}'}" --file-friendly-logging
wait

for ds in "definitions" "hypernymy" "hyponymy" "synonymy" "dictionary_qa"; do
    dataset=${LIB_LOC}/data_mcqa/$ds

    echo "now running ${dataset}"
    ${_python} -u -m allennlp.run evaluate_custom \
           --evaluation-data-file ${dataset}/dev.jsonl \
           --metadata-fields "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs,choice_context_list" \
           --output-file ${OUTPUT_DIR}/result_${ds}.jsonl \
           --cuda-device 0 \
           --overrides '{"iterator":{"batch_size":32}}' \
           ${MODEL_DIR}/model.tar.gz
    wait
done

"""


base_train = "$WORK/data/_generated/"
_path_script = "/gpfswork/rech/evd/uwe77wt/semantic_fragments/scripts_mcqa"

configs = {
    "rob_raw": "", #rob_raw
    #"35k.s1.e1.o1.t1": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/lot_exp/model/model.tar.gz", #rob neurips
    #sh2_rnd
    "35k.s3.e3.o1.t1.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s3.e3.o1.t1/sh2_rnd/model/model.tar.gz",
    "35k.s3.e3.o1.t3.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s3.e3.o1.t3/sh2_rnd/model/model.tar.gz",
    "35k.s1.e1.o1.t1.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/sh2_rnd/model/model.tar.gz",
    #sh3_rnd_t 
    "35k.s1.e4.o2.t1.sh3t.p7": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e4.o2.t1/sh3_rnd_t/p7/model/model.tar.gz", # p7
    "35k.s3.e1.o1.t1.sh3t.p7": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s3.e1.o1.t1/sh3_rnd_t/p7/model/model.tar.gz",# -> p7
    "35k.s1.e1.o1.t1.sh3t.p3": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/sh3_rnd_t/p3/model/model.tar.gz",#
    "35k.s1.e1.o1.t1.sh3t.p6": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/35k.s1.e1.o1.t1/sh3_rnd_t/p6/model/model.tar.gz",#-> p3, p6
}

configs = {
    "s1.e1.o1.t1.sh2": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/all.s1.e1.o1.t1/sh2_rnd/model/model.tar.gz",
    "s1.e1.o1.t1.sh3t.p3": "/gpfsscratch/rech/evd/uwe77wt/neurips/roberta-large/all.s1.e1.o1.t1/sh3_rnd_t/p3/model/model.tar.gz",
}

def create_script_training(config_name, model):
    global train_path, valid_path, model_name, slurm_template_train, slurm_static

    params = {
        "job_name": "",
        "path_script": "",
        "premodel": "",
    }
    job_name = config_name


    params["path_script"] = _path_script
    params["premodel"] = model
    params["job_name"] = job_name
    output_script_path = job_name + ".slurm"

    _template = slurm_template_train.format(**params) + slurm_static

    return _template, output_script_path


all_slurms = []
for config, model in configs.items():

    _template, output_script_path = create_script_training(config, model)

    with open(output_script_path, "w") as file:
        file.write(_template)

    all_slurms.append(output_script_path)


with open("all.bash", "w") as file:
    file.write("#!/bin/bash\n")
    for line in all_slurms:
        file.write("sbatch " + line+"\n")
