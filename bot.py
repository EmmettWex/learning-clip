from dotenv import load_dotenv
import os
import torch
import glob
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from clip_retrieval.clip_client import ClipClient
from annoy import AnnoyIndex
from PIL import Image

import discord
load_dotenv()
intents = discord.Intents.default()

client = discord.Client(intents=intents)

# declare the various clip models we will use
MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
TOKENIZER = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
DESCRIBER = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-H-14", num_images=1)

def build_annoy_tree():
    # check if a tree exists, if it does delete the old one and make a new one
    if os.path.exists('images.ann'):
        os.remove('images.ann')
    
    # the below code grabs all images in the dataset directory and turns them into vectors
    images = []
    image_paths = glob.glob('dataset/*_rgb.png')
    
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
        
    inputs = PROCESSOR(images=images, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        embeddings = MODEL.get_image_features(**inputs)
    
    # build the annoy tree
    annoy_tree = AnnoyIndex(512, 'euclidean')
    for idx, image in enumerate(embeddings):
        annoy_tree.add_item(idx, image)
    
    annoy_tree.build(72)
    annoy_tree.save('images.ann')
    
    return "Your tree has been built"

# search the db by text
def search_vector_database(query):
    # generate list containing the dataset
    image_paths = glob.glob('dataset/*_rgb.png')
    
    # pass query string into CLIP model tokenizer to generate a vector
    input = TOKENIZER([query], padding=True, return_tensors="pt")
    with torch.no_grad():
        vector = MODEL.get_text_features(**input)
    
    # query the Annoy Index with the vector to do a NNS returning
    # the closest image in the tree
    annoy_index = AnnoyIndex(512, 'euclidean')
    annoy_index.load('images.ann')
    indexes = annoy_index.get_nns_by_vector(vector[0], 1, search_k=-1)
    output = [image_paths[i] for i in indexes]
    
    return output

# ask the Clip model to describe an image
def describe_image(image_url):
    query_results = DESCRIBER.query(image=image_url)
    return query_results['caption']

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Hello command
    if message.content.startswith('!hello'):
        await message.channel.send('Hello!')

    # Search command
    if message.content.startswith('!search'):
        query = message.content.split("!search ",1)[1]
        # image_url = search_vector_database(query)
        await message.channel.send(image_url)

    # Ask command
    if message.content.startswith('!ask'):
        query = message.content.split("!ask ",1)[1]
        # answer = ask_function(query)
        await message.channel.send(answer)
        
    # Build annoy tree command
    if message.content.startswith('!build tree'):
        # helper function needs to check for previous tree
        # delete the previous tree if it exist, and build a new tree
        # if no tree exists then it just builds a new tree
        
        # answer = built_annoy_tree()
        await message.channel.send(answer)

    # Blend command
    if message.content.startswith('!blend'):
        images = message.content.split("!blend ",1)[1]
        # result = blend_images(images)
        await message.channel.send(result)

    # Daily theme command
    if message.content.startswith('!daily_theme'):
        # daily_theme = get_daily_theme()
        await message.channel.send(daily_theme)

    # Docs command
    if message.content.startswith('!docs'):
        topic = message.content.split("!docs ",1)[1]
        # link = generate_docs_link(topic)
        await message.channel.send(link)

    # Describe command
    if message.content.startswith('!describe'):
        image_url = message.content.split("!describe ",1)[1]
        # description = describe_image(image_url)
        await message.channel.send(description)

    # FAQ command
    if message.content.startswith('!faq'):
        # faq = get_faq()
        await message.channel.send(faq)

    # Fast command
    if message.content.startswith('!fast'):
        # response = switch_to_fast_mode()
        await message.channel.send(response)

    # Help command
    if message.content.startswith('!help'):
        # help_text = get_help()
        await message.channel.send(help_text)

    # Imagine command
    if message.content.startswith('!imagine'):
        prompt = message.content.split("!imagine ",1)[1]
        # image = generate_image(prompt)
        await message.channel.send(image)

    # Info command
    if message.content.startswith('!info'):
        # info = get_account_info()
        await message.channel.send(info)

    # Stealth command
    if message.content.startswith('!stealth'):
        # response = switch_to_stealth_mode()
        await message.channel.send(response)

    # Public command
    if message.content.startswith('!public'):
        # response = switch_to_public_mode()
        await message.channel.send(response)

    # Subscribe command
    if message.content.startswith('!subscribe'):
        # link = generate_subscription_link()
        await message.channel.send(link)

    # Settings command
    if message.content.startswith('!settings'):
        # settings = get_bot_settings()
        await message.channel.send(settings)

    # Prefer option command
    if message.content.startswith('!prefer option'):
        option = message.content.split("!prefer option ",1)[1]
        # response = set_preferred_option(option)
        await message.channel.send(response)

    # Prefer option list command
    if message.content.startswith('!prefer option list'):
        # options = list_preferred_options()
        await message.channel.send(options)

    # Prefer suffix command
    if message.content.startswith('!prefer suffix'):
        suffix = message.content.split("!prefer suffix ",1)[1]
        # response =```python
        # set_preferred_suffix(suffix)
        await message.channel.send(response)

    # Prefer suffix list command
    if message.content.startswith('!prefer suffix list'):
        # suffixes = list_preferred_suffixes()
        await message.channel.send(suffixes)

    # Suggest command
    if message.content.startswith('!suggest'):
        suggestion = message.content.split("!suggest ",1)[1]
        # response = send_suggestion(suggestion)
        await message.channel.send(response)

    # Theme command
    if message.content.startswith('!theme'):
        theme = message.content.split("!theme ",1)[1]
        # response = set_theme(theme)
        await message.channel.send(response)

    # Theme list command
    if message.content.startswith('!theme list'):
        # themes = list_themes()
        await message.channel.send(themes)

    # Top command
    if message.content.startswith('!top'):
        # top_results = get_top_results()
        await message.channel.send(top_results)

    # Warn command
    if message.content.startswith('!warn'):
        # response = warn_user()
        await message.channel.send(response)

client.run(os.getenv("TOKEN"))