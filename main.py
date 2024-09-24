# Read config from config.yaml
import yaml
import os
import openai


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

openai.api_key = config['openai']['api_key']

if __name__ == '__main__':
    print("Hello World!")
