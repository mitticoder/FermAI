"""
Powered by @x4cc3r
"""
import telebot
from deep_translator import GoogleTranslator

bot = telebot.TeleBot('6905726268:AAF3VMCy1xdx4RQbZI14MPQsfrxWIfnmmkQ')

print("started")
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Assalomu alaykum, {}! Botimizga xush kelibsiz. menga rasm yuboring!!! \n Привет! Добро пожаловать в наш бот. Отправь мне картинку!!!".format(message.from_user.first_name))

@bot.message_handler(content_types=['photo'])
def save_photo(message):

    photo_file = bot.get_file(message.photo[-1].file_id)
    downloaded_photo = bot.download_file(photo_file.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_photo)
        print("rasm saqlandi")

    from keras.models import load_model
    from PIL import Image, ImageOps
    import numpy as np

    np.set_printoptions(suppress=True)

    model = load_model("keras_Model.h5", compile=False)

    class_names = open("labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open("image.jpg").convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Bu o'simlik:", class_name[2:], end="")
    print("Aniqlik:", confidence_score * 100, '%')

    bot.reply_to(message, "Это растение: "+GoogleTranslator(source='auto', target="ru").translate(class_name[2:]))
    bot.reply_to(message, "Точность: "+str(confidence_score * 100) + ' %')

bot.polling()