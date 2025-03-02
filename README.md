
# Evaluating Personalized Tool-Augmented LLMs from the Perspectives of Personalization and Proactivity





### Abstract

Personalized tool utilization is essential for aligning large language models (LLMs) with user preference in interaction scenarios with various tools. However, most of the current benchmarks primarily focus on either personalization of text generation or direct tool-utilizing, without considering both. In this work, we introduce a novel benchmark **ETAPP** for evaluating personalized tool invocation, establishing a sandbox environment, and a comprehensive dataset of 800 testing cases covering diverse user profiles. To improve the accuracy of our evaluation, we propose a key-point-based LLM evaluation method, mitigating biases in the LLM-as-a-judge system by manually annotating key points for each test case and providing them to LLM as the reference. Additionally, we evaluate the excellent LLMs and provide an in-depth analysis. Furthermore, we investigate the impact of different tool-invoking strategies on LLMs' personalization performance and the effects of fine-tuning in our task. The effectiveness of our preference-setting and key-point-based evaluation method is also validated. Our findings offer insights into improving personalized LLM agents.



### Installation

```bash
cd ETAPP
pip install -r requirements.txt
```

Some of our tools require the use of Pyserini(https://github.com/castorini/pyserini), please install JDK according to the corresponding requirements. We recommend JDK-21.0.5.

### Data
The instructions used for testing is in ./data/instruction/instruction.json, and the training data is in folder ./data/training_data.
The tools we construct is in folder ./toolkit and their corresponding data is in ./database.
The profile we construct is in folder ./profile. The high level User Profile is in ./profile/profile.json, and the low level Tool-Utilizing Preferences are in folder ./profile/concrete_profile.

### Inference
To inference with different model, please utiliz the following command:

You need to set the environment config first:
```bash
cd Inference
export API_KEY='The OpenAI key'
export DEEPSEEK_API_KEY='The DeepSeek key'
export JAVA_HOME='The path to JDK environment'
export PATH=$PATH:$JAVA_HOME/bin
export max_turn=20
export qwen_model_name="The name of Qwen model in vllm"
export llama_model_name="The name of LLaMA model in vllm"
export watt_model_name="The name of watt model in vllm"
export url="The url of the vllm"
```

For ReAct Method in Tool-Given Scenario:
```bash
python evaluate_prompted_agent.py \
    --model_type CHATGPT \
    --model_name gpt-4o \
    --model_name_or_path gpt-4o \
    --mode chat \
    --method react \
    --add_example \
    --max_turn ${max_turn} 
```

for FC Method in Tool-Given Scenario:
```bash
python evaluate_prompted_agent.py \
    --model_type CHATGPT \
    --model_name gpt-4o \
    --model_name_or_path gpt-4o \
    --mode function_call \
    --method fine-tuned \
    --max_turn ${max_turn} 
```

For E-ReAct Method in Tool-Given Scenario:
```bash
python evaluate_prompted_agent.py \
    --model_type CHATGPT \
    --model_name gpt-4o \
    --model_name_or_path gpt-4o \
    --mode chat \
    --method e-react \
    --add_example \
    --max_turn ${max_turn} 
```

For ReAct Method in Tool-Retrieval Scenario:
```bash
python evaluate_prompted_agent.py \
    --model_type CHATGPT \
    --model_name gpt-4o \
    --model_name_or_path gpt-4o \
    --mode chat \
    --use_retrieval \
    --method react \
    --add_example \
    --max_turn ${max_turn} 
```

for FC Method in Tool-Retrieval Scenario:
```bash
python evaluate_prompted_agent.py \
    --model_type CHATGPT \
    --model_name gpt-4o \
    --model_name_or_path gpt-4o \
    --mode function_call \
    --use_retrieval \
    --method fine-tuned \
    --max_turn ${max_turn} 
```

For E-ReAct Method in Tool-Retrieval Scenario:
```bash
python evaluate_prompted_agent.py \
    --model_type CHATGPT \
    --model_name gpt-4o \
    --model_name_or_path gpt-4o \
    --mode chat \
    --use_retrieval \
    --method e-react \
    --add_example \
    --max_turn ${max_turn} 
```
For open source model using vllm:
```bash
python evaluate_prompted_agent.py \
    --model_type OpenModel \
    --model_name qwen_2.5_72B \
    --model_name_or_path /netcache/huggingface/Qwen2.5-72B-Instruct \
    --method fine-tuned \
    --use_vllm \
    --max_turn ${max_turn} 
```




If you want to test the reasoning model on part of data, run the following command:
```bash
python evaluate_prompted_agent_reason_inference.py \
    --model_type CHATGPT \
    --model_name o1-preview \
    --model_name_or_path o1-preview \
    --mode chat \
    --method react \
    --add_example \
    --max_turn ${max_turn} 



python evaluate_prompted_agent_reason_inference.py \
    --model_type CHATGPT \
    --model_name o1-preview \
    --model_name_or_path o1-preview \
    --mode chat \
    --method react \
    --use_retrieval \
    --add_example \
    --max_turn ${max_turn}
```





### Evaluation

If you want to evaluate the result in the inference process, run the following command:
```bash
export API_KEY='The OpenAI key'

python evaluate.py --result_file ../output/prompted_CHATGPT_gpt-4o_react
python evaluate.py --result_file ../output/prompted_CHATGPT_gpt-4o_react_retrieve

# for model testing on a subset of query
python evaluate_reason_inference.py --result_file ../output/prompted_CHATGPT_o1-preview_react_retrieve
```

### Fine-tuning
The fine-tuning process is using LLaMA-Factory(https://github.com/hiyouga/LLaMA-Factory).

