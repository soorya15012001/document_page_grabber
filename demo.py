import math
import cv2
import numpy as np
import operator


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


def draw(image, c):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = {}
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if c == 'v':
            lines[h] = [x, y, w, h]
            # cv2.line(img1, (x, y), (x + w, y + h), (0, 255, 0), 1)

        elif c == 'h':
            lines[w] = [x, y, w, h]
            # cv2.line(img1, (x, y), (x + w, y + h), (0, 255, 0), 1)


    l = dict(sorted(lines.items(), key=operator.itemgetter(0), reverse=True))
    print(len(l))
    if c == 'v':
        x, y, w, h = list(l.values())[0]
    else:
        # x, y, w, h = list(l.values())[3]
        x, y, w, h = list(l.values())[0]


    return [x, y, w, h]






# read a image using imread
img1 = cv2.imread("test3.PNG")
img1 = cv2.resize(img1, (400,400))
img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

# creating a Histograms Equalization
# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(img)
# equ = cv2.equalizeHist(equ)
thresh = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 93, 13)

kernel_h = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], np.uint8)

kernel_v = kernel_h.transpose()
image_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h, iterations=3)
image_h = cv2.erode(image_h,kernel_h,iterations = 1)

image_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v, iterations=5)
image_v = cv2.erode(image_v,kernel_v,iterations = 5)
# edges = cv2.Canny(image_v, 46, 109,3)
[x,y,w,h] = draw(image_h, 'h')
[x1,y1,w1,h1] = draw(image_v, 'v')

xi, yi = findIntersection(x,y,w,h,x1,y1,w1,h1)
xi = math.floor(xi)
yi = math.floor(yi)
print(xi, yi)




# cv2.circle(img1, (x,y), 20, (0,0,255),-1)
# cv2.circle(img1, (x+w,y+h), 20, (0,0,255),-1)
# cv2.circle(img1, (x1,y1), 20, (0,0,255),-1)
# cv2.circle(img1, (x1+w1,y1+h1), 20, (0,0,255),-1)
cv2.circle(img1, (xi, yi), 20, (0,200,255),-1)

if(math.sqrt((x-xi)**2 + (y-yi)**2) < math.sqrt(((x+w)-xi)**2 + ((y+h)-yi)**2)):
    cv2.line(img1, (xi, yi), (x + w, y + h), (0, 255, 0), 5) ########horizontal
    cv2.line(img1, (x+w, y+h), (x+w, 400 - (y+h)), (0, 255, 0), 10)
    cv2.circle(img1, (x+w, 400 - (y+h)), 20, (0, 0, 255), -1)
else:
    cv2.line(img1, (x, y), (xi, yi), (0, 255, 0), 5)  ########horizontal
    cv2.line(img1, (x, y), (x, 400 - y), (0, 255, 0), 10)
    cv2.circle(img1, (x, 400 - y), 20, (0, 0, 255), -1)

if (math.sqrt((x1 - xi) ** 2 + (y1 - yi) ** 2) < math.sqrt(((x1 + w1) - xi) ** 2 + ((y1 + h1) - yi) ** 2)):
    cv2.line(img1, (x1+w1, y1+h1), (x+w, 400 - (y+h)), (0, 255, 0), 10)
    cv2.line(img1, (xi, yi), (x1 + w1, y1 + h1), (0, 255, 0), 5) ######vertical
else:
    cv2.line(img1, (x1, y1), (x+w, 400 - (y+h)), (0, 255, 0), 10)
    cv2.line(img1, (x1, y1), (xi, yi), (0, 255, 0), 5) ######vertical

    # cv2.line(img1, (x+w, y+h), (x+w,400-(y+h)), (0, 255, 0), 10) ######vertical
    # cv2.line(img1, (x1 + w1, y1 + h1), (x+w,400-(y+h)), (0, 255, 0), 5) ########horizontal

cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()








