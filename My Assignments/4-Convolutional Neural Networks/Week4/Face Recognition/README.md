# Face Recognition for the Happy House（开心表情面部识别）
- 代码中的许多想法出自FaceNet
- 人脸验证（Face Verification）：当前对象是否是需要找的人，是一个1:1的问题。
- 人脸识别（Face Recognition）

- 运行GPU版本时，遇到BatchNormalization(axis=1)(x)函数报错。将[keras降级](https://github.com/keras-team/keras/issues/10382)到2.1.6即可解决问题：

'''
pip uninstall keras
pip install -I keras==2.1.6
'''