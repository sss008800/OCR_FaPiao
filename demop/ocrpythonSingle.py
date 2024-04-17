from paddleocr import PaddleOCR, draw_ocr
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
print('--***-'*3 + '\r\n')

img_path = 'D:\\03newland\\01ocr\\test\\773.jpeg'
result = ocr.ocr(img_path, cls=True)

print('-------------'*3)
for idx in range(len(result)):
    res = result[idx]
    print('\r\n')
    print('****'*10)
    for line in res:
        print(line)

# 显示结果
# 如果本地没有simfang.ttf，可以在doc/fonts目录下下载
from PIL import Image

result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('D:\\03newland\\01ocr\\result.jpg')
im_show.show()

# imm = Image.open('D:\\03newland\\01ocr\\773.jpeg')
# imm.show()