# -*- coding:utf-8 -*-
# @Project : AI_flask
# @File : server.py
# @Time : 2023/2/12 9:41


#::: 导入模块和包 :::
# Flask 常用工具
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# 导入Keras依赖
from keras.models import model_from_json
from tensorflow.python.framework import ops

ops.reset_default_graph()
from keras.utils import image_utils

# 导入其他依赖
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask应用引擎 :::
# 定义一个Flask应用
# 给Flask一个实例化对象,其中__name__入参是你的模块名或者包名，Flask应用会根据这个来确定你的应用路径以及静态文件和模板文件夹的路径
app = Flask(__name__)

# ::: 准备Keras模型 :::
# 模型文件
MODEL_ARCHITECTURE = './model/model_adam.json'
MODEL_WEIGHTS = './model/model_100_eopchs_adam_20190807.h5'

# 从外部文件加载模型

json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# 将权重输入到模型中
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: 模型函数 :::
def model_predict(img_path, model):
    '''
		Args:
			-- img_path : an URL path where a given image is stored.存储给定图像的URL路径。
			-- model : a given Keras CNN model.给定Keras CNN模型
	'''

    IMG = image_utils.load_img(img_path).convert('L')
    print(type(IMG))

    # 图像预处理
    IMG_ = IMG.resize((257, 342))
    print(type(IMG_))
    IMG_ = np.asarray(IMG_)
    print(IMG_.shape)
    IMG_ = np.true_divide(IMG_, 255)
    IMG_ = IMG_.reshape(1, 342, 257, 1)
    print(type(IMG_), IMG_.shape)

    print(model)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    prediction = model.predict(IMG_)
    prediction = np.argmax(prediction, axis=1)

    return prediction


# ::: FLASK ROUTES路线
@app.route('/', methods=['GET'])
def index():
    # 主页
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    # Constants:
    classes = {'TRAIN': ['Pneumonia', 'NORMAL', 'COVID'],
               'VALIDATION': ['Pneumonia', 'NORMAL'],
               'TEST': ['Pneumonia', 'NORMAL', 'COVID']}

    if request.method == 'POST':
        # 通过post请求获取文件
        f = request.files['file']

        # 将文件保存到 ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # 作出预测
        prediction = model_predict(file_path, model)

        predicted_class = classes['TRAIN'][prediction[0]]
        print('We think that is {}.'.format(predicted_class.lower()))

        return str(predicted_class).lower()


if __name__ == '__main__':
    app.run(debug=True)
