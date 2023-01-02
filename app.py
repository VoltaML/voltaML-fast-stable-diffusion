import gc
import json
import os
import random
from glob import glob

from flask import Flask, request, send_from_directory

from volta_accelerate import infer_pt, infer_trt

UPLOAD_FOLDER = "data/INCOMING_JSON/"
ALLOWED_EXTENSIONS = {"jpg"}


app = Flask(__name__, static_folder="static")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello_world():  # put application's code here
    return send_from_directory("static", "index.html")


@app.route("/acc")
def acc():  # put application's code here
    return send_from_directory("static", "acc.html")


@app.route("/voltaml/job", methods=["GET", "POST"])
def upload_file():
    jsonFile = request.get_json()
    print(jsonFile)
    if request.method == "POST":

        jsonFile = request.get_json()
        if jsonFile:  ### check if file is json format or not
            # f = request.files['jsonFile']
            body = jsonFile
            print("Request Params : ", body)
            job_id = random.randint(10000000, 99999999)

            prompt = body["prompt"]
            negative_prompt = body["prompt-ve"]
            img_height = body["img_height"]
            img_width = body["img_width"]
            num_inference_steps = body["num_inference_steps"]
            guidance_scale = body["guidance_scale"]
            seed = body["seed"]
            num_images_per_prompt = body["num_images_per_prompt"]
            model = body["model"]
            backend = body["backend"]

            # Create directory to save images if it does not exist
            saving_path = "static/output/" + str(job_id)
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)

            if "pt" in backend.lower():
                pipeline_time = infer_pt(
                    saving_path=saving_path,
                    model_path=model,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    img_height=img_height,
                    img_width=img_width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    seed=seed,
                )
            elif "trt" in backend.lower():
                pipeline_time = infer_trt(
                    saving_path=saving_path,
                    model=model,
                    prompt=prompt,
                    neg_prompt=negative_prompt,
                    img_height=img_height,
                    img_width=img_width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    seed=seed,
                )
            else:
                pipeline_time = None
            gc.collect()
            with open(saving_path + "/%s.json" % job_id, "w") as f:
                json.dump(
                    {"pipeline_time": pipeline_time if pipeline_time else "Unknown"}, f
                )

            return {
                "status": "SUCCESS",
                "jobId": job_id,
                "details": "Task submitted successfully",
            }

        else:
            return {"status": "FAILED", "details": "File does not exist."}

    else:
        return {"status": "STARTED", "progress": "10%"}


@app.route("/voltaml/accelerate_onnx")
def accelerate_onnx():
    args = get_args()
    args.onnx_trt = "onnx"
    convert_to_onnx(args)
    sys.stdout.flush()

    return "Success"


@app.route("/voltaml/accelerate_trt")
def accelerate_trt():
    args = get_args()
    args.onnx_trt = "trt"
    ## Command to convert onnx to tensorrt
    convert_to_trt(args)
    return "Success"


@app.route("/voltaml/scan_dir")
def scan_directory():
    pt_model_path = "/root/.cache/huggingface/diffusers"
    pt_models = []
    if os.path.exists(pt_model_path):
        models = os.listdir(pt_model_path)
        for i in models:
            if "model" in i.split("--")[0]:
                pt_models.append(os.path.join(i.split("--")[1], i.split("--")[2]))

    trt_model_path = "engine"
    trt_models = []
    if os.path.exists(trt_model_path):
        models = os.listdir(trt_model_path)
        for i in models:
            tmp2 = os.listdir(os.path.join(trt_model_path, i))
            for j in tmp2:
                tmp3 = os.listdir(os.path.join(trt_model_path, i, j))
                if len(tmp3) >= 3:
                    trt_models.append(os.path.join(i, j))
    print(pt_models, trt_models)

    return json.dumps({"pt_models": pt_models, "trt_models": trt_models})


@app.route("/voltaml/results")
def get_result():
    jobId = request.args.get("jobId")
    out_path = "static/output/" + str(jobId)
    print(out_path)
    print("Images ", glob(out_path + "/*"))
    if os.path.exists(out_path):
        if len(glob(out_path + "/*.png")) > 0:
            # imgs = os.listdir(out_path)
            # imgs = [out_path + "/" + file for file in imgs]
            imgs = glob(out_path + "/*.png")
            f = open(out_path + "/%s.json" % jobId)
            temp = json.load(f)
            pipeline_time = temp["pipeline_time"]
            return {
                "status": "SUCCESS",
                "progress": "100%",
                "out_path": imgs,
                "inference_time": pipeline_time + " s",
            }
        else:
            return {"status": "STARTED", "progress": "10%"}
    else:
        return {"status": "STARTED", "progress": "10%"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
