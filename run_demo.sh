# python demo.py --image_path demo_images/tanager.jpg --text_prompt back,beak,belly,breast,crown,forehead,eyes,legs,wings,nape,tail,throat -o outputs/ --device cuda --get_topk
# python demo.py --image_path table_items.png --text_prompt="white salt bottle,red pepper bottle,pot,ladle" -o outputs/ --device cuda
# python demo.py --image_path items_right_camera.jpg --text_prompt="ketchup,mustard,pot,ladle" -o outputs/ --device cuda --get_topk
python realtime_OD.py --video_path table_left.MOV --text_prompt="ketchup,blue tartar bottle,black pot,ladle" -o video_outputs_left/ --device cuda --get_topk --view=right
