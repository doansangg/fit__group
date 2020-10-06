*create enviroment python3
....command lines:
*install requiments:pip3 install -r requiments.txt
*download model:
https://drive.google.com/file/d/1YHqPbGOiXlmgHLhc5a9lJrxRS1GIheKJ/view
___________________
test result:
- results save test1.txt
trong test bao gồm tên ảnh và độ chính xác của ảnh thật. tùy mình chọn ngưỡng . bài toán sẽ có 2 class
command lines test image:
python3 test1.py --pretrained_model path_to_folder_models --image_size sizeimage --batch_size batch_size --image_folder path_to_folder_image --save_file path_to_file_save --model model --select select

example :python3 test1.py --pretrained_model /home/doan/Pictures/CVPR19-Face-Anti-spoofing-master/models/model_A_color_32/checkpoint --image_size 32 --batch_size 2 --image_folder /home/doan/Pictures/test_face_spoot --save_file /home/doan/Pictures/CVPR19-Face-Anti-spoofing-master --model model_A --select color

Note:
-chúng ta có 2 model : baseline,model_A
với mỗi model_A ta có các ảnh dạng color(màu),depth(sâu),ir
khi chọn model có size 32 thì ta nên để image có size 32...
trong select có color,ir,depth
**************
kết quả cuối cùng sẽ được lưu vào file text1.txt gồm có tên ảnh và độ chính xác như mình nói trên....

