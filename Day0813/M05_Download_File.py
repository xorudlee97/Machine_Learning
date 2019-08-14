from keras.applications import VGG19, Xception,ResNet50, InceptionV3, MobileNet

conv_base = VGG19()
conv_base.summary()

conv_base = Xception()
conv_base.summary()

conv_base = ResNet50()
conv_base.summary()

conv_base = InceptionV3()
conv_base.summary()

conv_base = MobileNet()
conv_base.summary()