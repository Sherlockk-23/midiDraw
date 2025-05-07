import numpy as np
import matplotlib.pyplot as plt

def vis_full_bitmap_data(path = 'quickDraw/full_numpy_bitmap_cat.npy', id = 0):
    # numpy_bitmap 的data, 每一个label单独一个文件，每张图 28*28 = 784， dtpye 是uint8, 0~255
    data = np.load(path)
    print(data.shape)
    # (123202, 784)
    print(data[0])
    
    pic = data[id].reshape(28, 28)
    plt.imshow(pic, cmap='gray')
    plt.title("QuickDraw Visualization")
    plt.show()

def vis_quick_rnn_data(path = 'quickDraw/sketchrnn_cat.npz', id = 0):
    # sketchrnn 的 data, 每一个label单独一个文件，每张图是一个长度 96 的时间序列，(dx, dy, pen)
    # 表示当前点与上一个点的相对坐标，以及笔画状态，0表示笔画中，1表示笔抬起

    data = np.load(path, encoding='latin1', allow_pickle=True)

    train, test, validate = data['train'], data['test'], data['valid']

    print('Train:', train.shape)
    print('Test:', test.shape)
    print('Validate:', validate.shape)

    # Train: (70000,)
    # Test: (2500,)
    # Validate: (2500,)
    # (96, 3)
    '''
    like: 
    [[ -34   -2    0]
    [ -26   28    0]
    [  -6   12    0]
    [  -2   26    0]
    [   2   14    0]
    [   8   12    0]
    ...
    [  12   10    1]]
    '''


    # show the first sketch
    sketch = train[id]
    # 初始化坐标
    x, y = 0, 0
    x_coords = [x]
    y_coords = [y]

    # 遍历数据，计算绝对坐标
    for dx, dy, pen in sketch:
        x += dx
        y += dy
        if pen == 0:  # 笔画中
            x_coords.append(x)
            y_coords.append(y)
        else:  # 笔抬起，记录当前点，但不连接
            x_coords.append(x)
            y_coords.append(y)
            x_coords.append(None)  # 添加 None 来断开线条
            y_coords.append(None)

    # 绘制图像
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-')
    plt.title("QuickDraw Visualization")
    plt.gca().set_aspect('equal', adjustable='box')  # 保持比例
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # vis_quick_rnn_data()
    vis_full_bitmap_data(id=15)