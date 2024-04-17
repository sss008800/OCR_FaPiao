import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import draw_ocr
from tools.infer.utility import text_visual

fapiao_labels = {'HUOHJDXV': '合计大写', 'XADDRESSV': '销售方地址电话', 'DATE': '开票日期', 'XCODEV': '销售方纳税人识别号',
                 'GADDRESSV': '购买方地址电话', 'CODEPM': '发票代码（票面）', 'HUODWV': '单位', 'HUOSEV': '税额', 'HUOSLUV': '税率',
                 'HUODJV': '单价', 'HUOHJXX': '合计小写', 'HUOXHV': '型号', 'NUMPRINT': '发票号码（打印）', 'XBANKV': '销售方开户行账号',
                 'GNAMEV': '购买方名称', 'GBANKV': '购买方开户行账号', 'NUMPM': '发票号码（票面）', 'XNAMEV': '销售方名称',
                 'HUOSLV': '税率', 'CODEPRINT': '发票代码（打印）', 'HUONAMEV': '货物或应税劳务服务名称', 'HUOJEV': '金额',
                 'GCODEV': '购买方纳税人识别号', 'HUOHJJE': '合计金额', 'HUOHJSE': '合计税额'}


def draw_ser_results(image,
                     ocr_results,
                     font_path="doc/fonts/simfang.ttf",
                     font_size=14):
    np.random.seed(2021)

    color = (np.random.permutation(range(255)),
             np.random.permutation(range(255)),
             np.random.permutation(range(255)))
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx])
        for idx in range(1, 255)
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')

    ## add by dxl
    if image.width < image.height:
        image = image.transpose(Image.ROTATE_90)

    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # boxes = [line["bbox"] for line in ocr_results]
    # txts = [line['pred']+ "--" +line['transcription'] for line in ocr_results]
    boxes = []
    txts = []

    # scores = [line[1][1] for line in ocr_results]

    #################################################################################
    print('--------------------------------------------------------------{}', image)
    print(ocr_results)
    # 合计的纵坐标
    hejih = [info['bbox'] for info in ocr_results if info['pred'] == 'HUOHJ']
    # print('===========================合计{}',hejih)
    # 名称的横坐标
    gnamew = [info['bbox'] for info in ocr_results if info['pred'] == 'GNAME']
    gnamex = [info['bbox'] for info in ocr_results if info['pred'] == 'XNAME']
    if len(gnamew) > 0:
        gnamew = gnamew[0][0]
    elif len(gnamex) > 0:
        gnamew = gnamex[0][0]
    else:
        gnamew = 0

    # 税额的横坐标
    shuie = [info['bbox'] for info in ocr_results if info['pred'] == 'HUOSEV']
    if len(shuie) > 0:
        shuie = shuie[0][2]
    else:
        shuie = image.width

    if len(hejih) > 0:
        hejih = hejih[0][1]
    else:
        hejih = 0
    #################################################################################

    for ocr_info in ocr_results:
        # 后处理
        ocr_info = after_process(ocr_info, color_map, hejih, gnamew, shuie)
        if ocr_info is None:
            continue

        color = color_map[ocr_info["pred_id"]]
        text = "{}: {}".format(fapiao_labels[ocr_info["pred"]], ocr_info["transcription"])

        if "bbox" in ocr_info:
            # draw with ocr engine
            bbox = ocr_info["bbox"]
        else:
            # draw with ocr groundtruth
            bbox = trans_poly_to_bbox(ocr_info["points"])

        boxes.append(bbox)
        txts.append(text)

        draw_box_txt(bbox, '', draw, font, font_size, color)

    im_show = Image.blend(image, img_new, 0.7)
    # 画txt by dxl
    # im_show = draw_ocr(im_show, boxes, txts, None, font_path='doc/fonts/simfang.ttf')
    if txts is not None:
        scores = [1] * len(boxes)
        img = np.array(resize_img(im_show, input_size=1000))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=1000,
            threshold=0.5,
            font_path=font_path)
        im_show = np.concatenate([np.array(img), np.array(txt_img)], axis=1)

    # return np.array(im_show)
    return im_show


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, outline=color, width=5)

    if text is not None and text != '':
        # draw ocr results
        left, top, right, bottom = font.getbbox(text)
        tw, th = right - left, bottom - top
        start_y = max(0, bbox[0][1] - th)
        draw.rectangle(
            [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + th)],
            fill=(0, 0, 255))
        draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)


def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


def draw_re_results(image,
                    result,
                    font_path="doc/fonts/simfang.ttf",
                    font_size=18):
    np.random.seed(0)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    color_head = (0, 0, 255)
    color_tail = (255, 0, 0)
    color_line = (0, 255, 0)

    for ocr_info_head, ocr_info_tail in result:
        draw_box_txt(ocr_info_head["bbox"], ocr_info_head["transcription"],
                     draw, font, font_size, color_head)
        draw_box_txt(ocr_info_tail["bbox"], ocr_info_tail["transcription"],
                     draw, font, font_size, color_tail)

        center_head = (
            (ocr_info_head['bbox'][0] + ocr_info_head['bbox'][2]) // 2,
            (ocr_info_head['bbox'][1] + ocr_info_head['bbox'][3]) // 2)
        center_tail = (
            (ocr_info_tail['bbox'][0] + ocr_info_tail['bbox'][2]) // 2,
            (ocr_info_tail['bbox'][1] + ocr_info_tail['bbox'][3]) // 2)

        draw.line([center_head, center_tail], fill=color_line, width=5)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def draw_rectangle(img_path, boxes):
    boxes = np.array(boxes)
    img = cv2.imread(img_path)
    img_show = img.copy()
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_show


## 处理一些推理不准的逻辑
def after_process(ocr_info, color_map, hejih, gnamew, shuie):
    if ocr_info["pred"] not in fapiao_labels:
        return None
    if ocr_info["pred_id"] not in color_map:
        return None

    transcription = ocr_info["transcription"]
    label_infer = ocr_info["pred"]

    if (label_infer == 'GNAMEV' or label_infer == 'XBANKV' or label_infer == 'GBANKV') and '：' in transcription:
        ocr_info["transcription"] = ocr_info["transcription"].split('：')[1]

    if label_infer == 'GADDRESSV' and (int(hejih) < int(ocr_info['bbox'][1])):
        ocr_info["pred"] = 'XADDRESSV'
    elif label_infer == 'GCODEV' and (int(hejih) < int(ocr_info['bbox'][1])):
        ocr_info["pred"] = 'XCODEV'
    elif label_infer == 'GBANKV' and (int(hejih) < int(ocr_info['bbox'][1])):
        ocr_info["pred"] = 'XBANKV'
    elif label_infer == 'XBANKV' and (int(hejih) > int(ocr_info['bbox'][1])):
        ocr_info["pred"] = 'GBANKV'
    elif int(gnamew) > int(ocr_info['bbox'][2]) or int(shuie) < int(ocr_info['bbox'][0]):  # 排除名称左侧的框和税额右侧的框
        return None
    elif label_infer == 'DATE' and '20' not in transcription:
        return None

    return ocr_info
