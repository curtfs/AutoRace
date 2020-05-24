import csv
import cv2
import json
import os

list_path = "train.csv"
frame_out_path = "data/cones.txt"
dataset_path = "YOLO_Dataset/train/images/"
labels_out_path = "YOLO_Dataset/train/labels/"

with open(list_path) as r_csv_file:
    csv_reader = csv.reader(r_csv_file)
    frame_txt_file = open(frame_out_path,'w+')
    
    for i, row in enumerate(csv_reader): 
        name = (row[0].split('.'))[0]
        processed = os.path.isfile(labels_out_path+name+".txt")
        if i < 2:
            continue
        if processed:
            frame_txt_file.write(dataset_path+row[0])
            continue

        #Perview the image
        if row[0] != "":
            im = cv2.imread(dataset_path+row[0])
        else:
            break
        frame_txt_file.write(dataset_path+row[0]+"\r\n")
        j=5
        exit = False
        image_width = im.shape[1]
        image_height = im.shape[0]
        img_file = open(labels_out_path+name+".txt",'a+')
        for img_box_str in row[5:]:
            if not img_box_str == "":
                print("found boxes")
                img_boxes = []
                pred_class = None
                co = json.loads(img_box_str)
                img = im.copy()
                cv2.rectangle(img, (co[0], co[1]), (co[0]+co[3], co[1]+co[2]), (255,0,0), 2)
                img = cv2.resize(img, (1024, 768))
                #Perview the cone
                try:
                    cv2.imshow('Frame',img)
                    valid = False
                    while(not valid):
                        pressed = cv2.waitKey(0)
                        if pressed == ord('a'):
                            pred_class = 0
                            valid = True
                        elif pressed == ord('d'):
                            pred_class = 1
                            valid =True
                        elif pressed == ord('s'):
                            pred_class = 2
                            valid =True
                        elif pressed == ord('e'):
                            pred_class = -1
                            valid = True
                        elif pressed == ord('q'):
                            cv2.destroyAllWindows()
                            exit = True
                            break
                    if pred_class == -1:
                        continue
                    if exit:
                        break
                    cv2.destroyAllWindows()
                    img_boxes.append(pred_class)
                    img_boxes.append(json.loads(img_box_str)[0])
                    img_boxes.append(json.loads(img_box_str)[1])
                    img_boxes.append(json.loads(img_box_str)[2])
                    img_boxes.append(json.loads(img_box_str)[3])
                    strrow = str(img_boxes[0]) + " " + str((img_boxes[1]+(img_boxes[4]/2))/image_width) + " " + str((img_boxes[2]+(img_boxes[3]/2))/image_height) + " " + str(img_boxes[4]/image_width) + " " + str(img_boxes[3]/image_height)
                    img_file.write(strrow+"\r\n")
                    j +=1
                except:
                    continue
        img_file.close()
        if exit:
            break
    frame_txt_file.close()

                    

    