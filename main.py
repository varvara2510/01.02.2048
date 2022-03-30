import telebot
import conf
import random
import requests
from telebot import types

import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model


bot = telebot.TeleBot(conf.TOKEN)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):           # функция для токенизации сообщения пользователя
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# функция для преобразования токенизированного сообщения пользователя в читаемый для модели циферный вид
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# функция для предсказания модели класса вопроса
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.15
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# функция для выдачи ответа по классу вопроса
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
    return result


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("A dog. Now.")
    item2 = types.KeyboardButton("Let's talk")

    markup.add(item1, item2)

    bot.send_message(message.chat.id, 'Welcome to this dump', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def lalala(message):
    if message.chat.type == "private":
        if message.text == "A dog. Now.":
            contents = requests.get('https://random.dog/woof.json').json()
            url = contents['url']
            bot.send_photo(chat_id=message.chat.id, photo=url)
        elif message.text == "Let's talk":
            bot.send_message(message.chat.id, "Okay, ask/tell me anything you want")
        else:
            ints = predict_class(message.text)
            res = get_response(ints, intents)
            bot.send_message(message.chat.id, res)



if __name__ == '__main__':
    bot.polling(none_stop=True)