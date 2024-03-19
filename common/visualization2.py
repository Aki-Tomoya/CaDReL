import numpy as np
import cv2
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop

def visulize_attention_ratio(img_path, attention_mask, save_path, ratio=1, cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    # img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    n_px = 288
    transform = Compose([ Resize(n_px, interpolation=Image.BICUBIC),  CenterCrop(n_px)])
    img = transform(img)
    plt.imshow(img, alpha=1)
    plt.axis('off')
    
    # normalize the attention mask
    mask = cv2.resize(attention_mask, (n_px, n_px))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.7, interpolation='nearest', cmap=cmap)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

#使用cv2在图像上绘制bounding boxes
def draw_bounding_boxes(img_path, bboxes, save_path, thickness=3):
    """
    img_path: 读取图片的位置
    bounding_boxes: bounding boxes的坐标，格式为[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    color: bounding boxes的颜色，可选
    thickness: bounding boxes的粗细，可选
    """
    print("load image from: ", img_path)
    # load the image557686
    img = cv2.imread(img_path)
    # draw bounding boxes
    # cv2.rectangle(img, (bboxes[0][0], bboxes[0][1]), (bboxes[0][2], bboxes[0][3]), (0, 0, 255), thickness)
    cv2.rectangle(img, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), (0, 255, 0), thickness)
    # cv2.rectangle(img, (bboxes[2][0], bboxes[2][1]), (bboxes[2][2], bboxes[2][3]), (255, 0, 0), thickness)
    # show the image
    cv2.imshow("img", img)
    #保存图片
    # cv2.imwrite(save_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = 'data/108982.jpg'
dir_path = '/Users/weimingboya/Desktop/results/108982'

bounding_boxes = np.array([[2.57175842e+02, 1.61188446e+02, 3.46592133e+02, 3.61927399e+02],
       [0.00000000e+00, 0.00000000e+00, 4.44710724e+02, 4.25145416e+02],
       [2.82684357e+02, 1.73391403e+02, 3.39373260e+02, 2.03646973e+02],
       [2.82925781e+02, 2.16270340e+02, 3.36422119e+02, 2.73790161e+02],
       [5.86275757e+02, 2.62174286e+02, 6.14115906e+02, 3.51804688e+02],
       [3.00071487e+01, 2.77527405e+02, 4.68565948e+02, 4.11812653e+02],
       [1.81141769e+02, 1.74975372e+02, 6.08747742e+02, 3.11253113e+02],
       [2.85293915e+02, 3.06045410e+02, 3.11439148e+02, 3.49479095e+02],
       [1.36330917e+02, 1.97765549e+02, 1.73507339e+02, 2.35721481e+02],
       [4.49973511e+02, 2.43019043e+02, 5.00869385e+02, 3.64724762e+02],
       [4.89817780e+02, 2.07459076e+02, 5.87829224e+02, 4.20915985e+02],
       [1.29921326e+02, 3.63495850e+02, 1.79225433e+02, 4.17945648e+02],
       [4.47822083e+02, 1.70189514e+01, 4.95926666e+02, 6.53237228e+01],
       [3.13763947e+02, 0.00000000e+00, 6.39288147e+02, 2.42106140e+02],
       [5.16365852e+01, 2.28962692e+02, 9.99287872e+01, 2.55390701e+02],
       [5.66120178e+02, 1.81628323e+00, 6.27518433e+02, 3.04670124e+01],
       [3.17553650e+02, 1.82142700e+02, 3.30937653e+02, 1.96600693e+02],
       [2.65120758e+02, 2.13831284e+02, 2.88616699e+02, 2.82119385e+02],
       [2.71192841e+02, 2.78980042e+02, 2.85003174e+02, 3.00872772e+02],
       [4.61262909e+02, 2.73529633e+02, 4.86292389e+02, 3.06200897e+02],
       [2.82197662e+02, 2.12487839e+02, 3.17178101e+02, 2.70349396e+02],
       [2.92886261e+02, 1.39801910e+02, 3.78188812e+02, 1.69931656e+02],
       [7.57838516e+01, 3.03960171e+01, 1.45679260e+02, 7.42448273e+01],
       [3.03154778e+00, 3.37670593e+02, 2.08796204e+02, 3.88364044e+02],
       [0.00000000e+00, 1.96201523e+02, 1.02065689e+02, 2.20938583e+02],
       [2.51424694e+01, 1.83962067e+02, 1.01444588e+02, 2.02906113e+02],
       [8.79922943e+01, 7.30064621e+01, 1.74604477e+02, 1.02738640e+02],
       [4.45846008e+02, 1.42386780e+02, 4.78089661e+02, 1.83669235e+02],
       [1.37644043e+01, 2.17329437e+02, 3.71467133e+01, 2.44177994e+02],
       [4.81648407e+01, 2.56171912e-01, 1.05968353e+02, 2.64843025e+01],
       [1.95521423e+02, 0.00000000e+00, 4.63426147e+02, 4.05218414e+02],
       [1.02791275e+02, 2.09226990e+02, 1.30550690e+02, 2.31270721e+02],
       [3.05284515e+02, 0.00000000e+00, 5.80644897e+02, 4.26288330e+02],
       [4.31599030e+02, 4.16534996e+00, 6.16698547e+02, 1.68705109e+02]]).astype(int)

attention_mask = np.array([[0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.002, 0.005, 0.004],
       [0.001, 0.004, 0.004, 0.01 , 0.004, 0.004, 0.002, 0.004, 0.012],
       [0.004, 0.006, 0.003, 0.004, 0.004, 0.001, 0.002, 0.091, 0.004],
       [0.002, 0.004, 0.008, 0.001, 0.012, 0.001, 0.121, 0.178, 0.007],
       [0.002, 0.001, 0.003, 0.022, 0.039, 0.003, 0.028, 0.061, 0.004],
       [0.004, 0.003, 0.001, 0.007, 0.018, 0.015, 0.003, 0.004, 0.001],
       [0.019, 0.004, 0.036, 0.009, 0.018, 0.001, 0.002, 0.042, 0.004],
       [0.003, 0.004, 0.019, 0.025, 0.003, 0.007, 0.002, 0.004, 0.003],
       [0.004, 0.003, 0.004, 0.004, 0.008, 0.004, 0.004, 0.004, 0.001]])

save_path = os.path.join(dir_path, '7-3.jpg')

# visulize_attention_ratio(img_path, attention_mask, save_path, cmap=plt.cm.gray)
draw_bounding_boxes(img_path, bounding_boxes[[6, 5, 30]], save_path)