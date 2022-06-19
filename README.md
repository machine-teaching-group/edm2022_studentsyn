## From {Solution} Synthesis to {Student Attempt} Synthesis for Block-Based Visual Programming Tasks
This repository contains the implementation for the paper ["From {Solution} Synthesis to {Student Attempt} Synthesis for Block-Based Visual Programming Tasks"](https://arxiv.org/abs/2205.01265).

----------------------------------------
### Overview

The repository has the following structure:
* `code/`: this folder contains files required for the evaluation of our techniques (RandD, EditD, EditEmbD, NeurSS, SymSS, TutorSS) on the benchmark StudentSyn.
* `data/`: this folder contains data necessary for the evaluation.
* `outputs/`: this is the output folder where the evaluation results will be stored.

The implementation requires python >= 3.7.9 to run. Required packages can be installed by running `pip install -r requirements.txt`. Next, we present the procedure for obtaining discriminative and generative evaluation results.

----------------------------------------

### Obtaining results

The results for the discriminative evaluation for all techniques (RandD, EditD, EditEmbD, NeurSS, SymSS, TutorSS) can be obtained by running the following commands. The results will be stored in `outputs/discriminative/`. 

* `python -m code.get_discriminative_results --source_task hoc4`

* `python -m code.get_discriminative_results --source_task hoc18`


The generated student codes for techniques with a generative capability (NeurSS, SymSS, TutorSS) can be obtained by running the following commands. The results will be stored in `outputs/generative/`.

* `python -m code.generate_codes --source_task hoc4`

* `python -m code.generate_codes --source_task hoc18`


NOTE: To run the above commands, you need to first create embeddings of the student data and train solution synthesizers for hoc4/hoc18 as described below.


----------------------------------------
### Creating embeddings

For the techniques (editEmbD, neurSS), an embedding-network is required that can generate embeddings of the student data.  We provide pre-trained embedding-networks for the two reference tasks in  `data/embD/hoc4/model_ID/` and `data/embD/hoc18/model_ID/`. Alternatively, new embedding-networks can be trained by running the commands below.
* `python -m code.embeddings.train_embeddings --task hoc4 --cmd train --model_dir model_new` 
* `python -m code.embeddings.train_embeddings --task hoc18 --cmd train --model_dir model_new` 
 

The trained embedding-networks will be stored in `data/embD/hoc4/model_new/` and `data/embD/hoc18/model_new/`.

Next, you need to create embeddings of the student data by running the commands below. Before executing these commands, the student data should be present in `data/student/` and the embedding-networks should be present in `data/embD/hoc4/model_1/` and `data/embD/hoc18/model_1/`.  The `--model_dir` argument can be used to specify a different embedding-network.

* `python -m code.embeddings.train_embeddings --task hoc4 --cmd create_emb`
* `python -m code.embeddings.train_embeddings --task hoc18 --cmd create_emb`

The embeddings will be stored in `data/embD/hoc4/model_1/` and `data/embD/hoc18/model_1/`.



----------------------------------------
### Training solution synthesizers for NeurSS

In order to train solution synthesizer models, synthetic training data needs to be generated. The training data for hoc4/hoc18 respectively can be obtained by running the commands below. 


* `python -m code.neurSS.generate_data.generate_training_data --num_examples 50000,2500,2500 --num_io 2 --dataset_name data  --patience 10  --num_cores 1 --task hoc4`

* `python  -m code.neurSS.generate_data.generate_training_data --num_examples 200000,2500,2500 --num_io 2 --dataset_name data  --patience 0  --num_cores 1 --task hoc18 --num_iterations 1000`

The synthetic training data will be stored in `data/neurSS/hoc4/training_data/` and `data/neurSS/hoc18/training_data/`. The `--num_cores` argument can be used for parallel data generation. We note that the data generation process for hoc18 will take a significant amount of time. 

You can train solution synthesizers for hoc4 and hoc18 by running the commands below.

* `python -m code.neurSS.synthesizer.train --data data --rotate True --run_id trained_synth_1 --val_freq 1 --epochs 100  --task hoc4  --override`

* `python -m code.neurSS.synthesizer.train --data data --rotate True --run_id trained_synth_1 --epochs 100  --batch_size 128 --val_freq 1 --task hoc18  --override`

The synthesizer models will be stored in `data/neurSS/hoc4/runs/` and `data/neurSS/hoc18/runs/`.

----------------------------------------
### Standalone evaluation scripts
We provide scripts to evaluate individual techniques in different experimental conditions and evaluation scales. 

To evaluate the discriminative performance of different techniques, we provide the script in `code/exp_discriminative/single_run.py`. The `--model_name` argument specifies the technique used for the evaluation and can be set to {randD, editD, editembD, neurSS, symSS, tutorSS}.  The results will be stored in `outputs/discriminative/`. See the script for additional input arguments. As an example, you can run the command below.

* `python -m code.exp_discriminative.single_run --model_name editD`

To generate student codes for a technique with a generative capability, we provide the script in `code/exp_generative/generate.py`. The `--model_name` argument specifies the technique used for the generation and can be set to {neurSS, symSS, tutorSS}.. The results will be stored in `outputs/generative/`. See the script for additional input arguments. As an example, you can run the command below.

* `python -m code.exp_generative.generate --model_name tutorSS`

----------------------------------------
### Data directory 


* `benchmark/`: this folder contains the benchmark StudentSyn.
* `embD/`: this folder contains pre-trained embedding-networks. 
* `neurSS/`: this folder is a placeholder where the solution synthesizer models for NeurSS will be stored. 
* `student/`: this folder is a placeholder where the code.org student data should be present.
* `symSS/hoc18/`: this folder contains extra information required by the SymSS model for the hoc18 reference task. 
* `tutorSS/`: this folder contains the discriminative and generative results for TutorSS. 

----------------------------------------
### Outputs directory 

* `discriminative/`: this folder is a placeholder where the results for the discriminative evaluation will be stored. 
* `generative/`: this folder is a placeholder where the results for the generative evaluation will be stored. 
* `saved_benchmarks/`: this folder is a placeholder where benchmark instances generated during evaluation will be stored.
