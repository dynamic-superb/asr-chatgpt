# ASR-ChatGPT

Implementation of ASR-ChatGPT in Dynamic-SUPERB

## Introduction
We adopted a concatenative framework using pre-trained models. Specifically, we used Whisper to transcribe the speech and then merge it with the instruction to guide ChatGPT in responding. However, we found responses from ChatGPT vary. For example, an answer “A” might be elaborated as “The answer is A.” To ensure ChatGPT adheres to our requirements, we added a prompt: “Please select one label from the provided options and respond with that label only.” Empirically, this constrains ChatGPT effectively.

## Utility

### Parameter Descriptions

This project allows users to work with the Whisper and ChatGPT models by specifying command-line parameters. Below is a detailed description of each parameter and its default value:

* ``` token ```: Your OpenAI API key. This parameter is mandatory.

* ``` whisper ```: Specifies which version of the Whisper model you'd like to use. Default is set to "openai/whisper-large-v2".

* ``` chatgpt ```: Specifies which version of the ChatGPT model you'd like to use. Default is "gpt-3.5-turbo".

* ``` dataset ```: The datasets you'd like to use. Default options include AccentClassification_AccentdbExtended and BirdSoundDetection_Warblrb10k.

* ``` dataset_path ```: The path to where your dataset is stored. Default is set to ./dynamic-superb-dataset.

* ``` transcript_path ```: The path where the Whisper transcripts will be saved. Default is ./whisper_transcript.

* ``` response_path ```: The path where the ChatGPT responses will be saved. Default is ./chatgpt_response.

* ``` log_path ```: The path where logs will be saved. Default is ./log.

* ``` device ```: The device on which you'd like to run the computations. Default is set to cuda for GPU computations.

### Running the Script
After setting up your desired parameters, run the script using:

``` python
python asr-chatgpt-pipeline.py --token YOUR_API_KEY --whisper "openai/whisper-large-v2" --chatgpt "gpt-3.5-turbo" --dataset YOUR_DATASET --dataset_path YOUR_DATASET_PATH --transcript_path YOUR_TRANSCRIPT_PATH --response_path YOUR_RESPONSE_PATH --log_path YOUR_LOG_PATH --device YOUR_DEVICE

```


or simply run the script with the default setting:

``` python
python asr-chatgpt-pipeline.py --token YOUR_API_KEY

```