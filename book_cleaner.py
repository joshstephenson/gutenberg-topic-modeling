#!/usr/bin/env python
###############################################################################
# This script is meant to process a folder of bulk books from the Gutenberg
# Project. It will find any text files in a folder named 'PG'. To change that
# folder name, change the `path` variable below.
#
# For any book that it is able to successfully parse the metadata, it will
# prompt the user to save it. Type [ENTER] to save, type anything else to skip.
# Saved books will be saved to `./Books`.
###############################################################################

import subprocess
import readchar
import os.path
from book_modeler import Book
from pathlib import Path
import re
import pickle

path = 'PG'
files = [str(child.resolve()) for child in sorted(Path.iterdir(Path(path)))]
files = [file for file in files if file.endswith('.txt')]

###############################################################################
# Saves a log of skipped books so they won't be imported again
###############################################################################
log = '.book-cleaner-log'
pickle_path = Path(log).resolve()
ignored = []
if os.path.exists(pickle_path):
    file = open(pickle_path, 'rb')
    ignored = pickle.load(file)
    file.close()

def path_for(book):
    def valid_char(char):
        return char.isalnum() or '_' == char
    title = re.sub(r'\s', '_', book.title)
    title = ''.join(e for e in title if valid_char(e))
    author = re.sub(r'\s', '_', book.author)
    author = ''.join(e for e in author if valid_char(e))
    filename = f'{title}-by-{author}.txt'
    path = Path(f'Books/{filename}').resolve()
    return path

def file_exists(path):
    return os.path.exists(path)

def save_clean_file(book, path):
    with open(path, 'w') as f:
        f.write('\n'.join(book.body_lines))
    f.close()
    subprocess.run(['vim', path])

def ignore(key):
    ignored.append(key)
    file = open(pickle_path, 'wb')
    pickle.dump(ignored, file)
    file.close()

for file in files:
    book = Book(file)
    if book.has_body():
        path = path_for(book)
        if not file_exists(path) and path not in ignored:
            print(f'Would you like to save "{book.title}" by "{book.author}"? ')
            inp = readchar.readchar()
            if inp == '\n':
                save_clean_file(book, path)
            else:
                ignore(path)
#        os.remove(file)
