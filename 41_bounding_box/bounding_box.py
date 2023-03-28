import torch 
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread('../data/img/catdog.jpg')
#  d2l.plt.imshow(img)
#  d2l.plt.show()

def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 +  y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=1)

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=1)
 

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [ 60.0, 45.0, 378.0, 516.0 ], [ 400.0, 112.0, 655.0, 493.0 ]
boxes = torch.tensor([dog_bbox, cat_bbox])
print(boxes)
print(box_corner_to_center(boxes))
print(box_center_to_corner(box_corner_to_center(boxes)))

def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height = bbox[3] - bbox[1], 
                                fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))

d2l.plt.show()
