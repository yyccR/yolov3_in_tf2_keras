
```python
def encoder(boxes,labels):
    '''
    boxes (tensor) [[x1,y1,x2,y2],[]], 其中xy均已归一化
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 7
    target = torch.zeros((grid_num,grid_num,30))
    # 计算宽高
    wh = boxes[:,2:]-boxes[:,:2]
    # 计算中心点xy
    cxcy = (boxes[:,2:]+boxes[:,:2])/2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        # 将归一化坐标中心点映射到7x7grid量纲下, 取整得到对应0-7下的坐标
        ij = (cxcy_sample * grid_num).ceil()-1
        # 对应目标概率为1, 对应类别概率为1
        target[int(ij[1]),int(ij[0]),4] = 1
        target[int(ij[1]),int(ij[0]),9] = 1
        target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
        xy = ij / grid_num
        # 计算xy偏移量
        delta_xy = (cxcy_sample -xy) * grid_num
        target[int(ij[1]),int(ij[0]),2:4] = wh[i]
        target[int(ij[1]),int(ij[0]),:2] = delta_xy
        target[int(ij[1]),int(ij[0]),7:9] = wh[i]
        target[int(ij[1]),int(ij[0]),5:7] = delta_xy
    return target
```