import cv2
import numpy as np
from glob import glob
import random
from config import image_shape 

global drawing 
drawing = False # true if mouse is pressed
global pt1_x , pt1_y
pt1_x , pt1_y = None , None
global img
img = np.array(cv2.imread(random.sample(glob('random_samples/*'),k=1)[0]))
# img = np.zeros(image_shape, np.uint8)
# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=5)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=5)        

def line_removing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,0,0),thickness=10)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,0,0),thickness=10)        


def my_drawing(img_size):
    
    draw_rmv = True
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw',line_drawing)
    while True:
        cv2.imshow('test draw',img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('test_img.png', img)
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            draw_rmv = not(draw_rmv)
            if draw_rmv==True:
                cv2.setMouseCallback('test draw',line_drawing)
            elif draw_rmv==False:
                cv2.setMouseCallback('test draw',line_removing)
#         elif cv2.waitKey(1) & 0xFF == ord('c'):
#             img = np.array(np.zeros(image_shape, np.uint8))
        
    cv2.destroyAllWindows()

def main():
    my_drawing(image_shape)
    
if __name__ == "__main__":
    main()
