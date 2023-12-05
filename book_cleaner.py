#!/usr/bin/env python

import subprocess
import readchar
import os.path
from gts import Book
from pathlib import Path
import re
import pickle

path = 'PG'
files = [str(child.resolve()) for child in sorted(Path.iterdir(Path(path)))]
files = [file for file in files if file.endswith('.txt')]

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
        os.remove(file)
