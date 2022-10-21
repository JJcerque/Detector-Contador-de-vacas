import torch
import cv2
import numpy  as np

# Model
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'custom','yolov5s')  # yolov5n - yolov5x6 official model
#                                            'custom', 'path/to/best.pt')  # custom model

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'C:/Users/NITRO 5/Desktop/Vid/Modelo/VacasAir.pt' )

# Images
im = cv2.VideoCapture("C:/Users/NITRO 5/Desktop/Vid/Arreando_el_ganado_drone.mp4")   # or file, Path, URL, PIL, OpenCV, numpy, list

while True:
    #lectura
    ret, frame = im.read()
    # Inference
    results = model(frame)

    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    # results.xyxy[0]  # im predictions (tensor)

    # results.pandas().xyxy[0]  # im predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

    # Conteo = results.pandas().xyxy[0].value_counts('name')
    Conteo = results.pandas().xyxy[0].value_counts('name')# class counts (pandas)
   
    # print((Conteo))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(frame ,"Cantidad",Conteo,(10,50),font,0.75,
#     (255,0,0),2,cv2.LINE_AA)
    
#     cv2.imshow('Vacas_p1', np.squeeze(results.render()))
#     t = cv2.waitKey(1) & 0xFF
#     if t == ord("s"):
#         break

    
# im.release()
# cv2.destroyAllWindows