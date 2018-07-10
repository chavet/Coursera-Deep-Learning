# Keras tutorial - the Happy House（开心表情面部检测）
- 通过Keras实现面部表情检测，判断表情是否开心。
- Keras是一个高级神经网络API编程框架，基于Python编写，能够在包括TensorFlow和CNTK在内的几个较低级别的框架上运行。
- 通过 graphviz 和 pydot 绘制模型图。利用Anaconda安装graphviz包，需要添加“D:\Anaconda3\pkgs\graphviz-2.38.0-4\Library\bin\graphviz”到环境变量path中，不然无法找到“dot.exe”。
- 模型的优化器用"Adam"，损失函数用"binary_crossentropy"，迭代到50次左右趋于平稳，测试集精度为97.3%。

