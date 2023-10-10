import os
import csv
import json
import time
import openai
from numpy import array
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset, Dataset, load_from_disk, disable_caching


disable_caching()


def query_whisper(audio, processor, model):

    input_features = processor(audio["array"], sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features.to("cuda"))
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription


def query_chatgpt(text, prompt, TOKEN, model_name="gpt-3.5-turbo"):

    openai.api_key = TOKEN

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            break

        except:
            print("\nChatGPT API failed. Try again.\n")
            time.sleep(5)
            continue

    response = completion.choices[0].message["content"].encode('unicode-escape').decode('unicode-escape')
    return response


def metric(dataset_name, error, total):

    error_rate = error / total
    accuracy = 1.0 - error_rate

    results = {}

    results[dataset_name] = {
        "Accuracy": accuracy,
        "Error-rate": error_rate,
        "Correct num": (total - error),
        "Error num": error,
        "Total num": total
    }

    return results


def log(dataset_name, results, output_log_path):

    with open(f"{output_log_path}/{dataset_name.split('/')[1]}.json", "w") as fp:
            json.dump(results, fp, indent=4)
    
    print(f"Write log and save in {output_log_path}/{dataset_name.split('/')[1]}.json")


def main(args):

    ''' Handle data path. '''
    dataset_path = Path(args.data_path)
    transcript_path = Path(args.transcript_path)
    response_path = Path(args.response_path)

    ''' Load Whisper model. '''
    processor = WhisperProcessor.from_pretrained(args.whisper)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper).to(args.device)
    model.config.forced_decoder_ids = None

    for dataset_name in tqdm(args.dataset):
        print(f"Download {dataset_name}")

        # Download dataset to disk
        if (dataset_path/dataset_name).exists():
            print("Dataset Exists")
            dataset = load_from_disk(dataset_path/dataset_name)
            print("Dataset loaded")
        else:

            dataset = load_dataset(f"DynamicSuperb/{dataset_name}", cache_dir=dataset_path)
            (dataset_path/dataset_name).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(dataset_path/dataset_name)    
        
        ''' Transcript the speech input via Whisper. '''
        transcripts = []
        with open(f"{transcript_path}/{dataset_name.split('/')[1]}_transcript.txt", 'w') as f:
            for i, audio_data in tqdm(enumerate(dataset["test"]["audio"]), total=len(dataset['test'])):
                transcript = query_whisper(audio_data, processor, model)
                transcripts.append(transcript)
                f.writelines(t + '\n' for t in transcript)

        ''' Load ChatGPT and OpenAI token. '''
        model_name = args.chatgpt
        api_token = args.token
        
        answers = []
        err = 0
        total = 0

        ''' Prompting ChatGPT. '''
        with open(f"{response_path}/{dataset_name.split('/')[1]}_chatgpt_response.txt", 'r+') as ans_file:
            with open(f"{transcript_path}/{dataset_name.split('/')[1]}_transcript.txt", 'r') as trans_file:
                answered = ans_file.readlines()
                lines = trans_file.readlines()
                for i, transcript in tqdm(enumerate(lines), total=len(lines)):

                    # Prompting ChatGPT given an instruction.
                    prompt = dataset["test"]["instruction"][i] + " Choose one answer from above options. Only one word is needed."
                    ans = query_chatgpt(transcript, prompt, api_token, model_name)
                    answers.append(ans)
                    ans_file.writelines(ans + '\n')

                    label = dataset["test"]["label"][i]
                   
                    # Check answer.
                    if ans.lower() != label.lower():
                        err += 1

                    total += 1
        
        print("Testing Finish!")
        print("Dataset Name: " + dataset_name)

        ''' Calculate accuracy. '''
        results = metric(dataset_name=dataset_name, error=err, total=total)

        output_log_path = args.log

        ''' Write log. '''
        log(dataset_name=dataset_name, results=results, output_log_path=output_log_path)


def parse_args() -> Namespace:

    parser = ArgumentParser()

    parser.add_argument("--token", type=str, help="Your OpenAI API key.")
    parser.add_argument("--whisper", type=str, default="openai/whisper-large-v2", help="Which Whisper do you want to utilize.")
    parser.add_argument("--chatgpt", type=str, default="gpt-3.5-turbo", help="Which ChatGPT do you want to utilize.")
    parser.add_argument("--dataset", type=str, default=["AccentClassification_AccentdbExtended", "BirdSoundDetection_Warblrb10k"])
    parser.add_argument("--dataset_path", type=str, default="./dynamic-superb-dataset")
    parser.add_argument("--transcript_path", type=str, default="./whisper_transcript")
    parser.add_argument("--response_path", type=str, default="./chatgpt_response")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()    

    return args


if __name__ == "__main__":

    args = parse_args()
    main(args)