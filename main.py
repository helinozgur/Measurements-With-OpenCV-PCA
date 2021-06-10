import cv2 as cv
import numpy as np
from medium_functions import Functions
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

detector = Functions()
src = cv.imread('./images/img3.jpg')
result, res_list = detector.calculateWidhtHeight(src)
result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
im_pil = Image.fromarray(result)
text = ''
for i in range(len(res_list)):
    x, y = res_list[i][1]
    index = res_list[i][0]
    text = text + '\n' + (str(index) + '. Cisim: \nUzunluk: ' + str(round(res_list[i][3], 1)) + ' piksel' +
                          '\nGenislik: ' + str(round(res_list[i][2], 1)) + ' piksel')

draw = ImageDraw.Draw(im_pil)
font = ImageFont.truetype("C:/WINDOWS/Fonts/ARIBLK.TTF", result.shape[1] // 50)
draw.text((20, 20), text, (0, 102, 0), font=font)
last_img = cv.cvtColor(np.array(im_pil), cv.COLOR_RGB2BGR)
cv.imshow("Sonuclar", cv.resize(last_img, None, None, .25, .25))
cv.waitKey(-1)
