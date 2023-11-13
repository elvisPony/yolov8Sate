# 使用yolo v8 預測圖片，並且從航拍圖辨識出的物件獲得矩陣
from ultralytics import YOLO
import pickle
import config
import os
from enum import Enum, auto
import torch
import math
import shutil


class UAV_Info(Enum):
    FILE_NAME = 0
    LATITUDE = auto()
    LONGITUDE = auto()
    FLY_HEIGHT = auto()
    ROLL = auto()
    PITCH = auto()
    YAW = auto()


NAME_TO_CLASS = {
    'TinHouse': 1,
    'Sunboard': 3,
    'WarnHouse': 4,
    'FishPond': 5,
    'School': 7
}


def cal_distance(centerPos, objPos):
    x_diff = centerPos[0] - objPos[0]
    y_diff = centerPos[1] - objPos[1]
    return math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))


def cal_theta(centerPos, objPos):
    sqrt = math.sqrt
    pow = math.pow
    acos = math.acos
    pi = math.pi
    obj_vector = [objPos[0]-centerPos[0], objPos[1] - centerPos[1]]
    # unit_vector = [0, 1]
    vector_prod = -obj_vector[1]
    length_prod = sqrt(pow(obj_vector[0], 2) + pow(obj_vector[1], 2)
                       ) * 1
    cos = vector_prod * 1.0 / (length_prod * 1.0)

    theta = acos(cos)  # 弧度
    if objPos[0] > centerPos[0]:
        theta = math.degrees(theta)
        theta = 360 - theta
        theta = math.radians(theta)

    return theta  # 弧度


def covert_class_yolo_to_custom(yoloClass, yoloClassDict):
    yoloClassName = yoloClassDict[yoloClass]
    return NAME_TO_CLASS[yoloClassName]


def sort_with_dis(x):
    return x[2]


def choice_center_obj(cls_list, it_result):
    def point_distance(p1, p2):
        x_diff = p1[0] - p2[0]
        y_diff = p1[1] - p2[1]
        x_diff = math.pow(x_diff, 2)
        y_diff = math.pow(y_diff, 2)
        return math.sqrt(x_diff + y_diff)

    image_center = [7952//2, 5304//2]
    minIndex = None
    minDistance = 99999999999999

    for obj_index in range(len(cls_list)):
        if cls_list[obj_index] == 1:
            continue
        obj_center_position = it_result.boxes.xywh[obj_index][:2].to(
            torch.int32).tolist()
        two_point_distance = point_distance(obj_center_position, image_center)
        if minDistance > two_point_distance:
            minDistance = two_point_distance
            minIndex = obj_index

    return minIndex


def main():
    # init
    if os.path.isdir(config.SAVE_PATH):
        shutil.rmtree(config.SAVE_PATH)
    os.mkdir(config.SAVE_PATH)

    # step 1 : load model
    model = YOLO('./best.pt')

    # step 2: predict object
    results = model.predict(source=config.TESTING_PATH, show=False, save=True, show_labels=True,
                            show_conf=False, conf=0.5, save_txt=False, save_crop=False, line_width=10)

    # step 3: calculator matrix

    for it_result in results:
        cls_list = it_result.boxes.cls.to(torch.int32).tolist()
        # cls_filter_list = list(
        #     filter(lambda it: it != 1, cls_list))  # 1 == river
        image_Name = it_result.path.split('\\')[-1]
        if len(cls_list) < 4:
            if os.path.isfile(f'{config.TESTING_PATH}/{image_Name}'):
                os.remove(f'{config.TESTING_PATH}/{image_Name}')
            continue

        matrix = []
        point = []
        yoloClass_to_name = it_result.names
        choiceObj_Index = choice_center_obj(cls_list, it_result)

        choiceObj_center_position = it_result.boxes.xywh[choiceObj_Index][:2].to(
            torch.int32).tolist()
        obj_type = covert_class_yolo_to_custom(
            cls_list[choiceObj_Index], yoloClass_to_name)
        matrix.append([obj_type, 0, 0])
        point.append([choiceObj_center_position])
        for obj_index in range(len(cls_list)):
            if choiceObj_Index == obj_index:
                continue
            obj_type = covert_class_yolo_to_custom(
                cls_list[obj_index], yoloClass_to_name)
            obj_center_position = it_result.boxes.xywh[obj_index][:2].to(
                torch.int32).tolist()
            dis = cal_distance(choiceObj_center_position, obj_center_position)
            theta = cal_theta(choiceObj_center_position, obj_center_position)
            matrix.append([obj_type, theta, dis])
            point.append([choiceObj_center_position, obj_center_position])

        sortMatrix = sorted(matrix, key=sort_with_dis)

        with open(f'{config.SAVE_PATH}/{image_Name}.pickle', 'wb') as f:
            pickle.dump(f,  sortMatrix)


if __name__ == "__main__":
    main()
