import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import pyparsing


def read_exr(path, channel=3):
    """
    path: exr文件的路径
    channel: 需要保存的通道数
    """
    # print("read exr from path: " + path)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # 把读取到的BGR转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = image[:, :, :channel]
    return image


def write_exr(path, image):
    """
    保存exr格式图片
    :param image: 图片数据
    :param path: 想要保存的图片路径
    """
    H, W, C = image.shape
    if C == 1:
        image = np.tile(image, (1, 1, 3))
    else:
        while C < 3:
            image = np.concatenate([image, np.zeros((H, W, 1), dtype=image.dtype)], axis=2)
            C = C + 1
    if C == 3:
        image = np.concatenate([image, np.ones((H, W, 1), dtype=image.dtype)], axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, image)


def read_matrix(path):
    """
    从以下格式中解析出投影矩阵
    ClipToView: [1 0 0 0] [0 0.5625 0 0] [0 0 0 0.1] [-0 -0 1 -0]  
    ViewMatrix: [-0.181067 0.170778 0.968529 0] [0.983471 0.031442 0.178317 0] [0 0.984808 -0.173648 0] [-3353.06 -1604.63 -4124.29 1] 
    ProjectionMatrix: [1 0 0 0] [0 1.77778 0 0] [0 0 0 1] [0 0 10 0] 
    FOV: 90
    NearClipDistance: 10
    FarClipDistance: 0
    """
    # 读取TXT文件
    with open(path, 'r') as f:
        text = f.read()

    number = pyparsing.Combine(
        pyparsing.Optional(pyparsing.Char("+-")) +
        pyparsing.Word(pyparsing.nums) +
        pyparsing.Optional("." + pyparsing.Word(pyparsing.nums)) +
        pyparsing.Optional(pyparsing.CaselessLiteral("E") + pyparsing.Optional(pyparsing.Char("+-")) + pyparsing.Word(pyparsing.nums))
    )
    number.setParseAction(lambda tokens: [float(tokens[i]) for i in range(len(tokens))])

    # 找到文本中所有的数字
    parsed_results = number.searchString(text)
    numbers = []
    for sublist in parsed_results:
        for number in sublist:
            numbers.append(number)
    
    # 解析各个矩阵
    view_matrix = np.array(numbers[16:32]).reshape((4, 4))
    projection_matrix = np.array(numbers[32:48]).reshape((4, 4))

    matrix = view_matrix @ projection_matrix
    return matrix