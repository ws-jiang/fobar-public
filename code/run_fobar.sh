!/bin/bash

#################################
## OpenAI key
#################################
export APIKEY="xxx"

#################################
## LLM
#################################
#model_name=text-davinci-003
model_name=gpt-3.5-turbo
#model_name=gpt-4

#################################
## base prompt
#################################
method_name="SCStandardCoT"      # i.e., FOBAR + CoT
#method_name="SCComplexCoT"        # i.e., FOBAR + ComplexCoT
#################################

#################################
## dataset: MultiArith SingleEq SVAMP GSM8K AQuA AddSub
#################################
ds_name="MultiArith"

#################################
## run the program
#################################
# 1. forward reasoning
python main_forward_reasoning.py --eng $model_name --ds $ds_name --method_name $method_name --num_repeat 10 --batch_size 40 --time_out 30 --num_proc 40

# 2. backward reasoning
python main_backward_reasoning.py --eng $model_name --ds $ds_name --method_name $method_name --num_repeat 8 --batch_size 40 --time_out 30 --num_proc 40

# 3. combine forward and backward reasoning
python main_fobar_eval.py --eng $model_name --ds $ds_name --method_name $method_name