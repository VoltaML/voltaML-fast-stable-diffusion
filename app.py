
from flask import Flask, flash, request, redirect, url_for
# from werkzeug.utils import secure_filename
from flask import request

import json
import os
import pandas as pd
import math
import logging
import threading
import random
from pytorch_model import inference, load_model
from PIL import Image
import uuid
from glob import glob
from volta_accelerate import *
from flask import Flask, send_from_directory


UPLOAD_FOLDER = 'data/INCOMING_JSON/'
ALLOWED_EXTENSIONS = {"jpg"}
# input_path = 'data/INPUT/'
# json_path = 'data/OUTPUT/'


app = Flask(__name__, static_folder='static')
# app.config['JSON_FOLDER'] = UPLOAD_FOLDER
# app.secret_key = '7503214cfgzdf'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():  # put application's code here
    return send_from_directory('static', 'index.html')

@app.route('/voltaml/job', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
       
        jsonFile = request.get_json()
        print(jsonFile)
        
        if jsonFile:    ### check if file is json format or not
            # f = request.files['jsonFile']
            body = jsonFile
            job_id = random.randint(10000000, 99999999)
                        
            prompt = body["prompt"]
            img_height = body["img_height"]
            img_width = body["img_width"]
            num_inference_steps = body["num_inference_steps"]
            guidance_scale = body["guidance_scale"]
            seed = body['seed']
            num_images_per_prompt = body["num_images_per_prompt"]
            model = body["model"]
            backend = body["backend"]
            # Create directory to save images if it does not exist
            saving_path = 'static/output/'+str(job_id)
            if not os.path.exists(saving_path):
                   os.makedirs(saving_path)
                                
            if "pt" in backend.lower():
                thread = threading.Thread(target=infer_pt, kwargs={
                                    'saving_path':saving_path, 
                                    'model':model,
                                    'prompt':prompt,
                                    'img_height':img_height, 
                                    'img_width':img_width, 
                                    'num_inference_steps':num_inference_steps, 
                                    'guidance_scale':guidance_scale, 
                                    'num_images_per_prompt':num_images_per_prompt, 
                                    'seed':seed
                                })
                thread.start()
            elif "trt" in backend.lower():
                thread = threading.Thread(target=infer_trt, kwargs={
                                    'saving_path':saving_path, 
                                    'model':model,
                                    'prompt':prompt,
                                    'img_height':img_height, 
                                    'img_width':img_width, 
                                    'num_inference_steps':num_inference_steps, 
                                    'guidance_scale':guidance_scale, 
                                    'num_images_per_prompt':num_images_per_prompt, 
                                    'seed':seed
                                })
                thread.start()
                # status = infer_trt(saving_path=saving_path, 
                #                     model=model,
                #                     prompt=prompt,
                #                     img_height=img_height, 
                #                     img_width=img_width, 
                #                     num_inference_steps=num_inference_steps, 
                #                     guidance_scale=guidance_scale, 
                #                     num_images_per_prompt=num_images_per_prompt, 
                #                     seed=seed
                #         )
                
            return {
                        "status": "SUCCESS",
                        "jobId": job_id,
                        "details": "Task submitted successfully"
                    }

        else:
            return  {
                        "status": "FAILED",
                        "details": "File does not exist."
                    }

    else:
            return {
                    "status": "STARTED",
                    "progress": "10%"
                    }
        


@app.route('/voltaml/accelerate_onnx')
def accelerate_onnx():
    args = get_args()
    args.onnx_trt='onnx'
    convert_to_onnx(args)
    sys.stdout.flush()

    return "Success"

@app.route('/voltaml/accelerate_trt')
def accelerate_trt():
    args = get_args()
    args.onnx_trt='trt'
    ## Command to convert onnx to tensorrt
    convert_to_trt(args)
    return "Success"

@app.route('/voltaml/results')
def get_result():
    
    jobId = request.args.get("jobId")
    out_path = 'static/output/' + str(jobId)
    print(out_path)
    print('Images ', glob(out_path+'/*'))
    if os.path.exists(out_path):
        if len(glob(out_path+'/*.png'))>0:
            imgs = os.listdir(out_path)
            imgs = [out_path + "/" + file for file in imgs]
            return {
                    "status":"SUCCESS",
                    "progress": "100%",
                    "out_path": imgs
                    }
        else:
            return {
                "status": "STARTED",
                "progress": "10%"
                }
    else:
        return {
                "status": "STARTED",   
                "progress": "10%"
                }


if __name__== "__main__":
    app.run(host="0.0.0.0",port=5002, debug=True)