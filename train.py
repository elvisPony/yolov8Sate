from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('./yolov8x-seg.pt')
    results = model.train(data="./datasets/newtest200/data.yaml", epochs=400,
                          batch=20)
    results = model.val(data='./datasets/newtest200/data.yaml')
