import argparse
import cv2
import numpy as np
import os
import json
import pickle
from PIL import ImageFont, ImageDraw, Image

blank = ' '*30
def parse_path(path):
	folder, split, category, file = path.replace('\n', '').split(',')
	return folder, split, category, file


def write_text(text, image, is_best):
	cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	pil_im = Image.fromarray(cv2_im_rgb)

	draw = ImageDraw.Draw(pil_im)

	# Choose a font
	font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
	if is_best:
		# Draw the text
		draw.text((0, 0), text, font=font, fill=(255,0,0,255))
	else:
		draw.text((0, 0), text, font=font, fill=(255,255,255,255))

	# Save the image
	cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
	return cv2_im_processed

orig_path ='/data/kclee/tw/AAAI/step1/images/dummy_dataset/'
def read_all_viscam(args, targets, img_path):
	_, split, category, file = parse_path(img_path)
	vis_list = []
	cam_list = []
	orig_image = cv2.imread(os.path.join(orig_path,split,category,file))
	for targ in targets:
		vis = cv2.imread(os.path.join(args.cams_path, targ, 
			split, category, 'vis', file))
		vis_list.append(vis)
		cam = pickle.load(open(os.path.join(args.cams_path, targ, 
			split, category, 'cam', file.replace('jpg','pkl')),'rb'))
		cam_list.append(cam)
	orig_image = cv2.resize(orig_image, (vis_list[0].shape[1], vis_list[0].shape[0]))
	return category, file, orig_image, vis_list, cam_list

def determine_best(args):
	target_models = sorted(os.listdir(args.cams_path))
	my_job_img_list = sorted(open(f'../{args.worker_name}_job.csv').readlines())
	idx = 0
	if os.path.isfile(args.worker_name+'_results.json'):
		best_json = json.load(open(args.worker_name+'_results.json','rb'))
		print("Previous progress loaded.")
	else:
		best_json = {}
		for il in my_job_img_list:
			_, split, category, file = parse_path(il)
			best_json[file] = {'best_model':-1}
	print("n : next frame, p : prev frame, q: quit")
	while True:
		category, file, orig, visuals, cams = read_all_viscam(args, target_models, my_job_img_list[idx])
		if best_json[file]['best_model'] != -1:
			print(f"\rCurrent Image : {file}, Best Model index : {best_json[file]['best_model']}{blank}", 
				sep=' ', end='', flush=True)
		else:
			print(f"\rCurrent Image : {file}, Best Model index : *[Type 1 ~ {len(cams)}]*{blank}", 
				sep=' ', end='', flush=True)
		show = write_text(f'Label : {category}', orig, False)
		for i in range(len(visuals)):
			show = np.concatenate((show,write_text(target_models[i], 
				visuals[i], best_json[file]['best_model'] == target_models[i])),axis=1)
		cv2.imshow('',show)
		key = cv2.waitKey(0)

		if key == 110:#'n' - next image
			idx += 1
			if idx >= len(my_job_img_list):
				print('\nEnd of Files!')
				idx -= 1
		elif key == 112:#'p' - previous image
			idx -= 1
			if idx < 0:
				print('\nFirst File!')
				idx +=1
		elif key in range(49, 49+len(cams)):#assign best model index to an image
			best_json[file]['best_model'] = target_models[key-49]
		elif key == 113:# 'q' - quit program
			break
		else:
			pass
	print('Annotation Finish')
	json.dump(best_json, open(args.worker_name+'_results.json','w'))
	print('Saved Progress in '+args.worker_name+'_results.json')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_name', type=str, required=True,
                        help='annotator name to determine work file list')
    parser.add_argument('--cams_path', type=str, default='../step1/gradcams',
                        help='path where model-wise gradcams are extracted')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = get_args()
	determine_best(args)