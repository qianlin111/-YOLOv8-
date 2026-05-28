# 使用训练好的模型进行目标检测推理
from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径

model=YOLO(r'D:\参照物、训练图片\新建文件夹\图片\2\weights\digit2.pt')
# 修改为自己的图像或者文件夹的路径
source = r'D:\参照物、训练图片\新建文件夹\图片\2\images\img_1.png' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)