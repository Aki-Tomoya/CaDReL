import math
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

data = [
{'word': 'a', 'v_boxes': [[285.4287109375, 118.23052978515625, 407.8519592285156, 246.18463134765625], [383.2654113769531, 151.52398681640625, 405.1028137207031, 176.86322021484375], [344.7373962402344, 185.7603759765625, 360.4090576171875, 200.19662475585938]], 'v_labels': ['man', 'head', 'hand'], 'v_values': [0.3745099902153015, 0.2729544937610626, 0.010958410799503326], 't_boxes': [[256, 0, 640, 319], [256, 0, 640, 319], [256, 0, 640, 319]], 't_texts': ['man surfing in suit', 'person surfing in a dark suit', 'surfer wearing a suit'], 't_values': [0.019367771223187447, 0.017514709383249283, 0.01691298931837082]},
{'word': 'man', 'v_boxes': [[285.4287109375, 118.23052978515625, 407.8519592285156, 246.18463134765625], [383.2654113769531, 151.52398681640625, 405.1028137207031, 176.86322021484375], [377.1704406738281, 189.9475555419922, 393.56805419921875, 247.18222045898438]], 'v_labels': ['man', 'head', 'arm'], 'v_values': [0.40431249141693115, 0.3740907609462738, 0.00919727049767971], 't_boxes': [[256, 0, 640, 319], [256, 0, 640, 319], [256, 0, 640, 319]], 't_texts': ['man surfing in suit', 'person surfing in a dark suit', 'surfer wearing a suit'], 't_values': [0.016206426545977592, 0.014726798981428146, 0.013730183243751526]},
{'word': 'riding', 'v_boxes': [[344.7373962402344, 185.7603759765625, 360.4090576171875, 200.19662475585938], [380.8319091796875, 235.21926879882812, 394.1870422363281, 251.14512634277344], [377.1704406738281, 189.9475555419922, 393.56805419921875, 247.18222045898438]], 'v_labels': ['hand', 'hand', 'arm'], 'v_values': [0.14928531646728516, 0.12371888756752014, 0.08945716172456741], 't_boxes': [[128, 107, 512, 426], [192, 159, 448, 373], [192, 159, 448, 373]], 't_texts': ['person surfing in a dark suit', 'man using h surfboard', 'male riding a surfboard'], 't_values': [0.017740972340106964, 0.016714297235012054, 0.01637962833046913]},
{'word': 'a', 'v_boxes': [[0.0, 157.11666870117188, 634.0059814453125, 306.001953125], [380.8319091796875, 235.21926879882812, 394.1870422363281, 251.14512634277344], [438.8356018066406, 125.74446868896484, 639.1111450195312, 294.0233154296875]], 'v_labels': ['wave', 'hand', 'wave'], 'v_values': [0.23961694538593292, 0.08043818175792694, 0.08037707209587097], 't_boxes': [[384, 319, 640, 533], [192, 319, 448, 533], [384, 319, 640, 533]], 't_texts': ['water in front of shore', 'water in front of where bird', 'cap visible on water'], 't_values': [0.0037391248624771833, 0.0036370849702507257, 0.0035592024214565754]},
{'word': 'wave', 'v_boxes': [[0.0, 157.11666870117188, 634.0059814453125, 306.001953125], [250.70619201660156, 242.65171813964844, 392.4864807128906, 303.01629638671875], [438.8356018066406, 125.74446868896484, 639.1111450195312, 294.0233154296875]], 'v_labels': ['wave', 'surfboard', 'wave'], 'v_values': [0.5933595299720764, 0.2031438946723938, 0.138631671667099], 't_boxes': [[192, 319, 448, 533], [192, 319, 448, 533], [192, 319, 448, 533]], 't_texts': ['water in front of where bird', 'water in front of bird', 'bird on water water'], 't_values': [0.0087924525141716, 0.008565760217607021, 0.007982484996318817]},
{'word': 'on', 'v_boxes': [[0.0, 157.11666870117188, 634.0059814453125, 306.001953125], [250.70619201660156, 242.65171813964844, 392.4864807128906, 303.01629638671875], [438.8356018066406, 125.74446868896484, 639.1111450195312, 294.0233154296875]], 'v_labels': ['wave', 'surfboard', 'wave'], 'v_values': [0.14310230314731598, 0.12153464555740356, 0.08680381625890732], 't_boxes': [[0, 0, 384, 319], [256, 0, 640, 319], [256, 0, 640, 319]], 't_texts': ['man paddling on a surfboard', 'surfer wearing a suit', 'man surfing in suit'], 't_values': [0.004442834760993719, 0.004250239115208387, 0.004191512707620859]},
{'word': 'a', 'v_boxes': [[250.70619201660156, 242.65171813964844, 392.4864807128906, 303.01629638671875], [383.2654113769531, 151.52398681640625, 405.1028137207031, 176.86322021484375], [344.7373962402344, 185.7603759765625, 360.4090576171875, 200.19662475585938]], 'v_labels': ['surfboard', 'head', 'hand'], 'v_values': [0.3888210952281952, 0.11190517991781235, 0.024943934753537178], 't_boxes': [[128, 107, 512, 426], [128, 107, 512, 426], [128, 107, 512, 426]], 't_texts': ['female on surfboard', 'male riding surfboard', 'male riding a surfboard'], 't_values': [0.011450551450252533, 0.010144750587642193, 0.01014423556625843]},
{'word': 'surfboard', 'v_boxes': [[250.70619201660156, 242.65171813964844, 392.4864807128906, 303.01629638671875], [10.22214126586914, 0.0, 639.1111450195312, 114.54202270507812], [344.6131286621094, 167.72752380371094, 378.4246520996094, 185.3402099609375]], 'v_labels': ['surfboard', 'sky', 'arm'], 'v_values': [0.920962929725647, 0.019951369613409042, 0.007984144613146782], 't_boxes': [[0, 214, 384, 533], [0, 214, 384, 533], [0, 214, 384, 533]], 't_texts': ['surfboard to left of person', 'male on surfboard', 'male on a surfboard'], 't_values': [0.021284522488713264, 0.021159347146749496, 0.019980894401669502]},
{'word': 'in', 'v_boxes': [[5.501492023468018, 246.04669189453125, 639.1111450195312, 532.1116333007812], [0.0, 94.94699096679688, 639.1111450195312, 427.98779296875], [0.0, 0.0, 639.1111450195312, 381.5587463378906]], 'v_labels': ['water', 'ocean', 'water'], 'v_values': [0.16452544927597046, 0.06955409795045853, 0.056376196444034576], 't_boxes': [[0, 319, 256, 533], [0, 0, 256, 213], [384, 319, 640, 533]], 't_texts': ['ripple nearing shore', 'splash in ocean', 'boundary_line on water'], 't_values': [0.02015048824250698, 0.015539647080004215, 0.01041356474161148]},
{'word': 'the', 'v_boxes': [[5.501492023468018, 246.04669189453125, 639.1111450195312, 532.1116333007812], [0.0, 94.94699096679688, 639.1111450195312, 427.98779296875], [10.22214126586914, 0.0, 639.1111450195312, 114.54202270507812]], 'v_labels': ['water', 'ocean', 'sky'], 'v_values': [0.3097544312477112, 0.06991825997829437, 0.06755413860082626], 't_boxes': [[0, 0, 256, 213], [0, 0, 256, 213], [0, 319, 256, 533]], 't_texts': ['splash in ocean', 'splash on sea', 'ripple nearing shore'], 't_values': [0.049145955592393875, 0.025092003867030144, 0.018156593665480614]},
{'word': 'ocean', 'v_boxes': [[5.501492023468018, 246.04669189453125, 639.1111450195312, 532.1116333007812], [10.22214126586914, 0.0, 639.1111450195312, 114.54202270507812], [0.0, 94.94699096679688, 639.1111450195312, 427.98779296875]], 'v_labels': ['water', 'sky', 'ocean'], 'v_values': [0.46057993173599243, 0.1435345560312271, 0.11512649059295654], 't_boxes': [[0, 0, 256, 213], [0, 0, 256, 213], [0, 319, 256, 533]], 't_texts': ['splash in ocean', 'splash on sea', 'ripple nearing shore'], 't_values': [0.1310810148715973, 0.07198073714971542, 0.06591401249170303]},
]

#使用matplotlib绘制图像，分四列显示，上面是v_boxes，下面是t_boxes，每一列的标签为word
def draw_bounding_boxes(img):
    print("load image from: ", img_path)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    
    #创建一个子图
    fig, axs = plt.subplots(math.ceil(len(data)/4)*2, 4, figsize=(7, 10), dpi=1000)
    
    for idx in range(len(data)):
        blk1 = np.zeros(img.shape, np.uint8)
        blk2 = np.zeros(img.shape, np.uint8)
        
        word = data[idx]['word']
        v_boxes = data[idx]['v_boxes']
        v_labels = data[idx]['v_labels']
        v_values = data[idx]['v_values']
        t_boxes = data[idx]['t_boxes']
        t_texts = data[idx]['t_texts']
        t_values = data[idx]['t_values']
        
        # 画框
        img1 = img
        for i in range(len(v_boxes)):
            cv2.rectangle(blk1, (int(v_boxes[i][0]), int(v_boxes[i][1])), (int(v_boxes[i][2]), int(v_boxes[i][3])), (255, 0, 0), -1)
            img1 = cv2.addWeighted(img1, 1, blk1, v_values[i]*10, 0)
            cv2.rectangle(img1, (int(v_boxes[i][0]), int(v_boxes[i][1])), (int(v_boxes[i][2]), int(v_boxes[i][3])), colors[i], 2)
            cv2.putText(img1, v_labels[i], (int(v_boxes[i][0]+5), int(v_boxes[i][1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
        
        img2 = img
        for i in range(len(t_boxes)):
            cv2.rectangle(blk2, (int(t_boxes[i][0]), int(t_boxes[i][1])), (int(t_boxes[i][2]), int(t_boxes[i][3])), (255, 0, 0), -1)
            img2 = cv2.addWeighted(img2, 1, blk2, t_values[i]*100, 0)
            cv2.rectangle(img2, (int(t_boxes[i][0]+i*2), int(t_boxes[i][1]+i*2)), (int(t_boxes[i][2]-(3-i)*2), int(t_boxes[i][3]-(3-i)*2)), colors[i], 2)
            cv2.putText(img2, t_texts[i], (int(t_boxes[i][0]+10), int(t_boxes[i][1]+(i+1)*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
        
        #将图像格式转换成matplotlib可显示的格式
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
        # 分四列显示，显示标题word，显示纵轴标签
        axs[2*(idx//4), idx%4].imshow(img1)
        # axs[2*(idx//4), idx%4].axis('off')
        axs[2*(idx//4), idx%4].set_title(word)
        axs[2*(idx//4)+1, idx%4].imshow(img2)
        # axs[2*(idx//4)+1, idx%4].axis('off')
        
        #去掉坐标轴，但显示纵轴标签
        axs[2*(idx//4), idx%4].set_xticks([])
        axs[2*(idx//4), idx%4].set_yticks([])
        axs[2*(idx//4)+1, idx%4].set_xticks([])
        axs[2*(idx//4)+1, idx%4].set_yticks([])
        
        if idx%4 == 0:
            axs[2*(idx//4), idx%4].set_ylabel('Objects')
            axs[2*(idx//4)+1, idx%4].set_ylabel('Contexts')
        
        #去掉边界框
        axs[2*(idx//4), idx%4].spines['top'].set_visible(False)
        axs[2*(idx//4), idx%4].spines['right'].set_visible(False)
        axs[2*(idx//4), idx%4].spines['bottom'].set_visible(False)
        axs[2*(idx//4), idx%4].spines['left'].set_visible(False)
        axs[2*(idx//4)+1, idx%4].spines['top'].set_visible(False)
        axs[2*(idx//4)+1, idx%4].spines['right'].set_visible(False)
        axs[2*(idx//4)+1, idx%4].spines['bottom'].set_visible(False)
        axs[2*(idx//4)+1, idx%4].spines['left'].set_visible(False)

    #将剩余的子图去掉
    for i in range(len(data), math.ceil(len(data)/4)*4):
        fig.delaxes(axs[2*(i//4), i%4])
        fig.delaxes(axs[2*(i//4)+1, i%4])

    #保存图片
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

img_path = 'data/181474.jpg'
dir_path = 'data/result'
save_path = os.path.join(dir_path, 'result.jpg')

img = cv2.imread(img_path)
# data = [data[i] for i in [1,2,4,7]]
draw_bounding_boxes(img)