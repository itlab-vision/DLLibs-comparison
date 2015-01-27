#-------------------------------------------------
#
# Project created by QtCreator 2014-11-19T06:52:41
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = caffe_predict
CONFIG   += console
CONFIG   -= app_bundle
CONFIG += Debug
TEMPLATE = app


SOURCES += main.cpp
INCLUDEPATH += "/usr/local/cuda-6.5/include"
LIBS += -L/home/evgeniy/caffe/build/lib -lcaffe
LIBS += -L/home/evgeniy/caffe/build/src/caffe/proto -lproto
LIBS += -L/usr/lib -lprotobuf
LIBS += -L/usr/local/lib -lglog
INCLUDEPATH += "/usr/local/include"
LIBS += -L/usr/lib -lopencv_highgui -lopencv_imgproc -lopencv_core

