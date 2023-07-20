import string
import nltk
import requests

from googlesearch import search
from lxml import html
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
# helper function to generate text corpus from html elements
def generate_corpus(all_p_elements):
    corpus = ""
    for p_element in all_p_elements:
        corpus += '\n' + ''.join(p_element.findAll(text = True))
    return corpus



import os
import transformers


def initialize_model():

  model = transformers.pipeline("conversational", model="facebook/blenderbot_small-90M")
  os.environ["TOKENIZERS_PARALLELISM"] = "true"

  return model


def get_bot_response(model, user_input):

  chat = model(transformers.Conversation(user_input))
  bot_response = str(chat)
  bot_response = bot_response[bot_response.find("bot >> ")+6:].strip()

  return bot_response
