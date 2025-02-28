"""
# Here are all the installs and imports you will need for your word cloud script and uploader widget

!pip install wordcloud
!pip install fileupload
!pip install ipywidgets
!jupyter nbextension install --py --user fileupload
!jupyter nbextension enable --py fileupload
"""

import wordcloud
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
import fileupload
import io
import sys

# This is the uploader widget

def _upload():

    _upload_widget = fileupload.FileUploadWidget()

    def _cb(change):
        global file_contents
        decoded = io.StringIO(change['owner'].data.decode('utf-8'))
        filename = change['owner'].filename
        print('Uploaded `{}` ({:.2f} kB)'.format(
            filename, len(decoded.read()) / 2 **10))
        file_contents = decoded.getvalue()

    _upload_widget.observe(_cb, names='data')
    display(_upload_widget)

_upload()


def removing_words(list1,list2):
    for word in list1:
        word=word.lower()
        if word in list2:
            for i in range(list1.count(word)):
                list1.remove(word)
    for w in list2:
        if w in list1:
            removing_words(list1,list2)
    return list1


def removing_punctuations(list1,punc):
    for word in list1:
        if not word.isalpha():
            new_word=""
            ind=list1.index(word)
            for letter in word:
                if letter not in punc:
                    new_word+=letter
            if new_word.isalpha():
                list1[ind]=new_word
            else:
                list1.remove(word)
                removing_punctuations(list1,punc)
    return list1

def calculate_frequencies(file_contents):
    # Here is a list of punctuations and uninteresting words you can use to process your text
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    uninteresting_words = ["the", "a", "to", "if", "is", "it", "of", "and", "or", "an", "as", "i", "me", "my", \
    "we", "our", "ours", "you", "your", "yours", "he", "she", "him", "his", "her", "hers", "its", "they", "them", \
    "their", "what", "which", "who", "whom", "this", "that", "am", "are", "was", "were", "be", "been", "being", \
    "have", "has", "had", "do", "does", "did", "but", "at", "by", "with", "from", "here", "when", "where", "how", \
    "all", "any", "both", "each", "few", "more", "some", "such", "no", "nor", "too", "very", "can", "will", "just"]

    # LEARNER CODE START HERE
    file_contents_list=file_contents.split()
    file_contents_list=removing_punctuations(file_contents_list,punctuations)
    file_contents_list=removing_words(file_contents_list,uninteresting_words)
    print(type(file_contents_list),len(file_contents_list))
    frequencies={}
    for word in file_contents_list:
        if word not in frequencies:
            frequencies[word]=0
        frequencies[word]+=1
    #wordcloud
    cloud = wordcloud.WordCloud()
    cloud.generate_from_frequencies(frequencies)
    return cloud.to_array()


# Display your wordcloud image

myimage = calculate_frequencies(file_contents)
plt.imshow(myimage, interpolation = 'nearest')
plt.axis('off')
plt.show()
