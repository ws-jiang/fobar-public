import openai
import time
import os

openai.api_key = os.environ["APIKEY"]
if "APIBASE" in os.environ:
    openai.api_base = os.environ["APIBASE"]
    print(openai.api_base)

def update_args(args):
    if args.eng in ("text-davinci-003"):
        args.batch_size = 20
    return args

def create_response(prompt_input, eng='text-davinci-003', max_tokens=1024, temperature=0.0, stop="Q", timeout=20):
    assert eng in ('text-davinci-003')
    response = openai.Completion.create(
        engine=eng,
        prompt=prompt_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["{}:".format(stop)],
        request_timeout=timeout
    )
    return response


def create_response_chat(prompt_input, eng='gpt-3.5-turbo',  temperature=0.0, timeout=20):
    assert eng in ["gpt-3.5-turbo", "gpt-4"]
    response = openai.ChatCompletion.create(
        model=eng,
        messages=prompt_input,
        temperature=temperature,
        request_timeout=timeout,
    )
    return response