import argparse
import text
from utils import load_filepaths_and_text
from tqdm import tqdm
if __name__ == '__main__':
    text_index = 1
    text_cleaners = ['english_cleaners2']
    filelist = './filelists/processed_result.txt'
    out_extension = 'cleaned'
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in tqdm(range(len(filepaths_and_text))):
        original_text = filepaths_and_text[i][text_index]
        cleaned_text = text._clean_text(original_text, text_cleaners)
        filepaths_and_text[i][text_index] = cleaned_text
    new_filelist = filelist + "." + out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])