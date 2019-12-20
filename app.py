# -*- coding: utf-8 -*-

from flask import Flask, redirect, make_response, request, jsonify, render_template, url_for, send_from_directory, session
from keras import models
from PIL import Image
from keras.models import load_model
from flask_cors import CORS
from PIL import ImageFile
from keras.backend import tensorflow_backend as backend
import keras
import numpy as np
import sys, os, io
import glob
import tensorflow as tf
from keras.models import model_from_json
from werkzeug import secure_filename
import base64

# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

CORS(app)

sys.path.append("./keras_yolo3/")


from my_yolo import MyYOLO


def rotateImage(img, orientation):
    """
    iphoneでアップした画像が回転する現象対策
    画像ファイルをOrientationの値に応じて回転させる
    """
    #orientationの値に応じて画像を回転させる
    if orientation == 1:
        pass
    elif orientation == 2:
        #左右反転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        #180度回転
        img_rotate = img.transpose(Image.ROTATE_180)
    elif orientation == 4:
        #上下反転
        img_rotate = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        #左右反転して90度回転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    elif orientation == 6:
        #270度回転
        img_rotate = img.transpose(Image.ROTATE_270)
    elif orientation == 7:
        #左右反転して270度回転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
    elif orientation == 8:
        #90度回転
        img_rotate = img.transpose(Image.ROTATE_90)
    else:
        pass

    return img_rotate

yolo = MyYOLO(
    classes_path="voc_classes.txt",
    model_path="trained_weights_final.h5",
    anchors_path="yolo_anchors.txt")

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            print("ファイルがありません")
        else:
            img = request.files["file"]
            filename = secure_filename(img.filename)

            root, ext = os.path.splitext(filename)
            ext = ext.lower()

            gazouketori = set([".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".bmp", ".pbm", ".pgm", ".ppm",
                      ".pxm", ".pnm",  ".sr",  ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic", ".dib"])
            if ext not in gazouketori:
                return render_template('index.html',massege = "対応してない拡張子です",color = "red")
            print("success")

            image = Image.open(img)

            #exif対応
            try:
                #exif情報取得
                exifinfo = image._getexif()
                #exif情報からOrientationの取得
                orientation = exifinfo.get(0x112, 1)
                #画像を回転
                image = rotateImage(image, orientation)
            except:
                pass

            image_size_yolo = 320
            rgb_im = image.convert('RGB')
            rgb_im.thumbnail([image_size_yolo,image_size_yolo])

            back_ground = Image.new("RGB", (image_size_yolo,image_size_yolo), color=(255,255,255))
            back_ground.paste(rgb_im)

            result, result_img = yolo.detect_image(back_ground)
            # result, result_img = yolo.detect_image(image)
            # yolo.close_session()

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            [ print(f"{label} ({score})") for label, score in result.items()]
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            details = [
                   '水温95℃を限度に、洗濯機で洗えます。',
                   '水温70℃を限度に、洗濯機で洗えます。',
                   '水温60℃を限度に、洗濯機で洗えます。',
                   '水温60℃を限度に、洗濯機で弱い洗濯ができます。',
                   '水温50℃を限度に、洗濯機で洗えます。',
                   '水温50℃を限度に、洗濯機で弱い洗濯ができます。',
                   '水温40℃を限度に、洗濯機で洗えます',
                   '水温40℃を限度に、洗濯機で弱い洗濯ができます。',
                   '水温40℃を限度に、洗濯機で非常に弱い洗濯ができます。',
                   '水温30℃を限度に、洗濯機で洗えます。',
                   '水温30℃を限度に、洗濯機で弱い洗濯ができます。',
                   '水温30℃を限度に、洗濯機で非常に弱い洗濯ができます。',
                   '水温40℃を限度に、手洗いできます。',
                   'ご家庭では洗えません。',
                   '塩素系・酸素系漂白剤で、漂白できます。',
                   '酸素系漂白剤で、漂白できます。塩素系漂白剤ではできません。',
                   '漂白できません。',
                   '排気温度80℃を上限に、タンブル乾燥できます。',
                   '排気温度60℃を上限に、タンブル乾燥できます。',
                   'タンブル乾燥禁止です。',
                   '平干しします。',
                   '脱水せずぬれたまま、平干しします。',
                   'ハンガー等を使って、つり干しします。',
                   '脱水せずぬれたまま、つり干しします。',
                   '日陰で、平干しします。',
                   '日陰で、脱水せずぬれたまま平干しします。',
                   '日陰で、つり干しします。',
                   '日陰で、脱水せずぬれたままつり干しします。',
                   '200℃を限度に、アイロンが使えます。',
                   '150℃を限度に、アイロンが使えます。',
                   '110℃を限度に、アイロンが使えます。',
                   'アイロンは使えません。',
                   'パークロロエチレン及び石油系溶剤による、ドライクリーニングができます。',
                   'パークロロエチレン及び石油系溶剤による、弱いドライクリーニングができます。',
                   '石油系溶剤による、ドライクリーニングができます。',
                   '石油系溶剤による、弱いドライクリーニングができます。',
                   'ドライクリーニングはできません。',
                   'ウエットクリーニングができます。',
                   '弱い操作による、ウエットクリーニングができます。',
                   '非常に弱い操作による、ウエットクリーニングができます。',
                   'ウエットクリーニングはできません。']


            results = []
            for label, score in result.items():
                results.append([str(label), details[int(label)-1], str(score)])

            buf = io.BytesIO()
            result_img.save(buf, 'png')
            qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
            qr_b64data = "data:image/png;base64,{}".format(qr_b64str)

            return render_template('index.html', img=qr_b64data, results=reversed(results))
    else:
        print("get request")

    return render_template('index.html')

@app.route('/individual',methods=['GET','POST'])
def individual():
    return render_template('individual.html')

@app.route('/predict_individual', methods=['GET','POST'])
def predict_individual():
    if request.method == "POST":
        if 'file' not in request.files:
            print("ファイルがありません")
        else:
            img = request.files["file"]
            filename = secure_filename(img.filename)

            root, ext = os.path.splitext(filename)
            ext = ext.lower()

            gazouketori = set([".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".bmp", ".pbm", ".pgm", ".ppm",
                      ".pxm", ".pnm",  ".sr",  ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic", ".dib"])
            if ext not in gazouketori:
                return render_template('individual.html',massege = "対応してない拡張子です",color = "red")
            print("success")

            graph = tf.get_default_graph()
            backend.clear_session() # 2回以上連続してpredictするために必要な処理
            # モデルの読み込み
            model = model_from_json(open('and_1.json', 'r').read())

            # 重みの読み込み
            model.load_weights('and_1_weight.hdf5')


            image_size = 50

            image = Image.open(img)

            #exif対応
            try:
                #exif情報取得
                exifinfo = image._getexif()
                #exif情報からOrientationの取得
                orientation = exifinfo.get(0x112, 1)
                #画像を回転
                image = rotateImage(image, orientation)
            except:
                pass

            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = np.array(data)
            X = X.astype('float32')
            X = X / 255.0
            X = X[None, ...]

            prd = model.predict(X)
            other_labels = np.argsort(prd)[0][::-1][:3]
            other_pros = [prd[0][other_labels[0]], prd[0][other_labels[1]], prd[0][other_labels[2]]]

            details = [
                   '水温95℃を限度に、洗濯機で洗えます。',
                   '水温50℃を限度に、洗濯機で洗えます。',
                   'ハンガー等を使って、つり干しします。',
                   '漂白できません。',
                   '弱い操作による、ウエットクリーニングができます。',
                   '水温60℃を限度に、洗濯機で洗えます。',
                   'ウエットクリーニングができます。',
                   'ドライクリーニングはできません。',
                   '非常に弱い操作による、ウエットクリーニングができます。',
                   '水温30℃を限度に、洗濯機で非常に弱い洗濯ができます。',
                   '水温40℃を限度に、洗濯機で非常に弱い洗濯ができます。',
                   '200℃を限度に、アイロンが使えます。',
                   'ご家庭では洗えません。',
                   'ウエットクリーニングはできません。',
                   '石油系溶剤による、ドライクリーニングができます。',
                   '塩素系・酸素系漂白剤で、漂白できます。',
                   '日陰で、平干しします。',
                   '石油系溶剤による、弱いドライクリーニングができます。',
                   '脱水せずぬれたまま、平干しします。',
                   'パークロロエチレン及び石油系溶剤による、弱いドライクリーニングができます。',
                   '排気温度60℃を上限に、タンブル乾燥できます。',
                   '150℃を限度に、アイロンが使えます。',
                   'タンブル乾燥禁止です。',
                   '水温30℃を限度に、洗濯機で洗えます。',
                   '脱水せずぬれたまま、つり干しします。',
                   '日陰で、脱水せずぬれたまま平干しします。',
                   '水温40℃を限度に、洗濯機で弱い洗濯ができます。',
                   'アイロンは使えません。',
                   '平干しします。',
                   '水温60℃を限度に、洗濯機で弱い洗濯ができます。',
                   '日陰で、つり干しします。',
                   '酸素系漂白剤で、漂白できます。塩素系ではできません。',
                   '水温30℃を限度に、洗濯機で弱い洗濯ができます。',
                   'パークロロエチレン及び石油系溶剤による、ドライクリーニングができます。',
                   '水温40℃を限度に、手洗いできます。',
                   '水温70℃を限度に、洗濯機で洗えます。',
                   '日陰で、脱水せずぬれたままつり干しします。',
                   '排気温度80℃を上限に、タンブル乾燥できます。',
                   '水温40℃を限度に、洗濯機で洗えます',
                   '水温50℃を限度に、洗濯機で弱い洗濯ができます。',
                   '110℃を限度に、スチームなしでアイロンが使えます。']

            icons = [1,
                    5,
                    23,
                    17,
                    39,
                    3,
                    38,
                    37,
                    40,
                    12,
                    9,
                    29,
                    14,
                    41,
                    35,
                    15,
                    25,
                    36,
                    22,
                    34,
                    19,
                    30,
                    20,
                    10,
                    24,
                    26,
                    8,
                    32,
                    21,
                    4,
                    27,
                    16,
                    11,
                    33,
                    13,
                    2,
                    28,
                    18,
                    7,
                    6,
                    31]

            pre1_icon = str(icons[other_labels[0]])
            pre1_detail = details[other_labels[0]]
            pre1_pro = str(round(other_pros[0] * 100)) + '%'

            pre2_icon = str(icons[other_labels[1]])
            pre2_detail = details[other_labels[1]]
            pre2_pro = str(round(other_pros[1] * 100)) + '%'

            pre3_icon = str(icons[other_labels[2]])
            pre3_detail = details[other_labels[2]]
            pre3_pro = str(round(other_pros[2] * 100)) + '%'

            kwargs = {
                "pre1_icon" : pre1_icon,
                "pre1_detail"  : pre1_detail,
                "pre1_pro"     : pre1_pro,
                "pre2_icon" : pre2_icon,
                "pre2_detail"  : pre2_detail,
                "pre2_pro"     : pre2_pro,
                "pre3_icon" : pre3_icon,
                "pre3_detail"  : pre3_detail,
                "pre3_pro"     : pre3_pro
            }

            buf = io.BytesIO()
            image = Image.open(img)
            image.save(buf, 'png')

            qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
            qr_b64data = "data:image/png;base64,{}".format(qr_b64str)



            return render_template('individual.html', img=qr_b64data, **kwargs)
    else:
        print("get request")

    return render_template('individual.html')

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
@app.errorhandler(413)
def oversize(error):
    return render_template('index.html',massege = "画像サイズが大きすぎます",color = "red")
@app.errorhandler(400)
def nosubmit(error):
    return render_template('index.html',massege = "画像を送信してください",color = "red")
@app.errorhandler(503)
def all_error_handler(error):
     return 'InternalServerError\n', 503

if __name__ == '__main__':
    app.run()
