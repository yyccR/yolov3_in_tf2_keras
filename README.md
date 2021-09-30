
```python
def encoder(boxes,labels):
    '''
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 7
    target = torch.zeros((grid_num,grid_num,30))
    wh = boxes[:,2:]-boxes[:,:2]
    cxcy = (boxes[:,2:]+boxes[:,:2])/2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample * grid_num).ceil()-1 #
        target[int(ij[1]),int(ij[0]),4] = 1
        target[int(ij[1]),int(ij[0]),9] = 1
        target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
        xy = ij / grid_num
        delta_xy = (cxcy_sample -xy) * grid_num
        target[int(ij[1]),int(ij[0]),2:4] = wh[i]
        target[int(ij[1]),int(ij[0]),:2] = delta_xy
        target[int(ij[1]),int(ij[0]),7:9] = wh[i]
        target[int(ij[1]),int(ij[0]),5:7] = delta_xy
    return target
```