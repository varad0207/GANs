import os
from tqdm.auto import tqdm

from PIL import Image

from datasets import load_dataset

# dataset from: https://huggingface.co/datasets/keremberke/chest-xray-classification
# dataset labels: ['NORMAL', 'PNEUMONIA']
# number of images: {'train': 4077, 'test': 582, 'valid': 1165}

def resize_img(i, dir):
    img = Image.open(f'{dir}/{i}.jpg')
    resized_img = img.resize((64, 64))
    resized_img.save(f'{dir}/{i}.jpg')

if __name__ == "__main__":

    print('Loading Dataset')
    df = load_dataset("keremberke/chest-xray-classification", name="full")
    print('Performing Split')
    train_df = df['train']
    test_df = df['test']
    validation_df = df['validation']

    normal_data_len = 0
    pneumonia_data_len = 0

    if not os.path.exists('data/train_data_normal'):
        os.mkdir('data/train_data_normal')
    if not os.path.exists('data/train_data_pneumonia'):
        os.mkdir('data/train_data_pneumonia')
    if not os.path.exists('data/test_data'):
        os.mkdir('data/test_data')
    if not os.path.exists('data/validation_data'):
        os.mkdir('data/validation_data')

    print('Resizing and Saving Images')
    for i in tqdm(range(len(train_df))):
        if train_df[i]['labels'] == 0:
            normal_data_len = normal_data_len + 1
            train_df[i]['image'].save(f'data/train_data_normal/{i}.jpg')
            resize_img(i, 'data/train_data_normal')
        else:
            pneumonia_data_len = pneumonia_data_len + 1
            train_df[i]['image'].save(f'data/train_data_pneumonia/{i}.jpg')
            resize_img(i, 'data/train_data_pneumonia')

    for i in tqdm(range(len(test_df))):
        test_df[i]['image'].save(f'data/test_data/{i}.jpg')
        resize_img(i, 'data/test_data')

    for i in tqdm(range(len(validation_df))):
        validation_df[i]['image'].save(f'data/validation_data/{i}.jpg')
        resize_img(i, 'data/validation_data')
