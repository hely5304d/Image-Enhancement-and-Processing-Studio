#!/usr/bin/env python3
"""
Image Enhancement & Processing Studio - Web Application
Full-Stack Interactive Computer Vision Studio built with Flask, NumPy, OpenCV, and Pillow.
"""

import os
import io
import uuid
import zipfile
import base64
import numpy as np
import cv2
from PIL import Image
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    url_for,
    send_from_directory,
)

from core import (
    ImagePipeline,
    enhancement,
    restoration,
    filters,
    geometry,
    metrics,
    presets,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLES_FOLDER = os.path.join(BASE_DIR, "samples")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "image-enhancement-studio-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# In-memory session registry: session_id -> { "pipeline": ImagePipeline, "filename": str }
SESSIONS = {}


def get_or_create_session(session_id: str = None) -> tuple[str, ImagePipeline]:
    if not session_id or session_id not in SESSIONS:
        new_id = str(uuid.uuid4())
        SESSIONS[new_id] = {
            "pipeline": ImagePipeline(),
            "filename": "image.png",
        }
        return new_id, SESSIONS[new_id]["pipeline"]
    return session_id, SESSIONS[session_id]["pipeline"]


def numpy_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Converts OpenCV BGR image to base64 data URI."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode image to base64.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


@app.route("/")
def index():
    available_presets = [
        {"id": k, **v} for k, v in presets.PRESETS_REGISTRY.items()
    ]
    return render_template("index.html", presets=available_presets)


@app.route("/batch")
def batch():
    available_presets = [
        {"id": k, **v} for k, v in presets.PRESETS_REGISTRY.items()
    ]
    return render_template("batch.html", presets=available_presets)


@app.route("/docs")
def documentation():
    return render_template("docs.html")


@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read file into OpenCV BGR array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback using Pillow
            file.seek(0)
            pil_img = Image.open(file.stream).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        session_id = str(uuid.uuid4())
        pipeline = ImagePipeline(img)
        SESSIONS[session_id] = {
            "pipeline": pipeline,
            "filename": file.filename,
        }

        h, w = img.shape[:2]
        orig_b64 = numpy_to_base64(pipeline.original, ext=".png")
        curr_b64 = numpy_to_base64(pipeline.current, ext=".png")
        hist_data = metrics.compute_histogram_data(pipeline.current)
        eval_stats = pipeline.evaluate()

        return jsonify({
            "session_id": session_id,
            "filename": file.filename,
            "width": w,
            "height": h,
            "original_image": orig_b64,
            "current_image": curr_b64,
            "histograms": hist_data,
            "metrics": eval_stats,
            "recipe": pipeline.get_recipe(),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process uploaded image: {str(e)}"}), 500


@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    sample_name = request.json.get("sample", "low_light.png")
    sample_path = os.path.join(SAMPLES_FOLDER, sample_name)
    if not os.path.isfile(sample_path):
        return jsonify({"error": f"Sample {sample_name} not found"}), 404

    try:
        pipeline = ImagePipeline.from_file(sample_path)
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "pipeline": pipeline,
            "filename": sample_name,
        }

        h, w = pipeline.current.shape[:2]
        orig_b64 = numpy_to_base64(pipeline.original, ext=".png")
        curr_b64 = numpy_to_base64(pipeline.current, ext=".png")
        hist_data = metrics.compute_histogram_data(pipeline.current)
        eval_stats = pipeline.evaluate()

        return jsonify({
            "session_id": session_id,
            "filename": sample_name,
            "width": w,
            "height": h,
            "original_image": orig_b64,
            "current_image": curr_b64,
            "histograms": hist_data,
            "metrics": eval_stats,
            "recipe": pipeline.get_recipe(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/apply-preset", methods=["POST"])
def apply_preset():
    data = request.json or {}
    session_id = data.get("session_id")
    preset_name = data.get("preset")

    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid or expired session"}), 400

    if preset_name not in presets.PRESETS_REGISTRY:
        return jsonify({"error": f"Unknown preset: {preset_name}"}), 400

    pipeline: ImagePipeline = SESSIONS[session_id]["pipeline"]
    preset_info = presets.PRESETS_REGISTRY[preset_name]
    pipeline._apply_op(f"preset_{preset_name}", preset_info["function"])

    curr_b64 = numpy_to_base64(pipeline.current, ext=".png")
    hist_data = metrics.compute_histogram_data(pipeline.current)
    eval_stats = pipeline.evaluate()

    return jsonify({
        "current_image": curr_b64,
        "histograms": hist_data,
        "metrics": eval_stats,
        "recipe": pipeline.get_recipe(),
    })


@app.route("/api/apply-action", methods=["POST"])
def apply_action():
    data = request.json or {}
    session_id = data.get("session_id")
    action = data.get("action")
    params = data.get("params", {})

    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid or expired session"}), 400

    pipeline: ImagePipeline = SESSIONS[session_id]["pipeline"]

    try:
        if action == "clahe":
            clip = float(params.get("clip_limit", 2.5))
            grid = int(params.get("grid_size", 8))
            pipeline.clahe(clip_limit=clip, tile_grid_size=(grid, grid))

        elif action == "gamma":
            gamma_val = float(params.get("gamma", 1.2))
            pipeline.gamma(gamma=gamma_val)

        elif action == "retinex":
            pipeline.retinex()

        elif action == "white_balance":
            method = params.get("method", "gray_world")
            pipeline.white_balance(method=method)

        elif action == "auto_contrast":
            clip = float(params.get("clip_percent", 1.0))
            pipeline.auto_contrast(clip_percent=clip)

        elif action == "brightness_contrast":
            b = float(params.get("brightness", 0.0))
            c = float(params.get("contrast", 0.0))
            pipeline.brightness_contrast(brightness=b, contrast=c)

        elif action == "vibrance":
            sat = float(params.get("saturation", 1.0))
            vib = float(params.get("vibrance", 0.0))
            pipeline.vibrance(saturation=sat, vibrance=vib)

        elif action == "pseudo_hdr":
            pipeline.pseudo_hdr()

        elif action == "denoise_nlm":
            strength = float(params.get("strength", 10.0))
            pipeline.denoise_nlm(h_lum=strength, h_col=strength)

        elif action == "denoise_bilateral":
            d = int(params.get("diameter", 9))
            pipeline.denoise_bilateral(d=d)

        elif action == "dehaze":
            pipeline.dehaze()

        elif action == "document_clean":
            pipeline.document_clean()

        elif action == "unsharp_mask":
            strength = float(params.get("strength", 1.5))
            sigma = float(params.get("sigma", 1.5))
            pipeline.unsharp_mask(sigma=sigma, strength=strength)

        elif action == "cartoonize":
            pipeline.cartoonize()

        elif action == "pencil_sketch":
            color = bool(params.get("color", False))
            pipeline.pencil_sketch(color=color)

        elif action == "vignette":
            v_strength = float(params.get("strength", 0.6))
            pipeline.vignette(strength=v_strength)

        elif action == "sepia":
            pipeline.sepia()

        elif action == "rotate":
            angle = float(params.get("angle", 90.0))
            pipeline.rotate(angle=angle)

        elif action == "flip":
            direction = params.get("direction", "horizontal")
            pipeline.flip(direction=direction)

        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

        curr_b64 = numpy_to_base64(pipeline.current, ext=".png")
        hist_data = metrics.compute_histogram_data(pipeline.current)
        eval_stats = pipeline.evaluate()

        return jsonify({
            "current_image": curr_b64,
            "histograms": hist_data,
            "metrics": eval_stats,
            "recipe": pipeline.get_recipe(),
        })
    except Exception as e:
        return jsonify({"error": f"Operation error: {str(e)}"}), 500


@app.route("/api/undo", methods=["POST"])
def undo_action():
    session_id = request.json.get("session_id")
    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    pipeline: ImagePipeline = SESSIONS[session_id]["pipeline"]
    pipeline.undo()

    curr_b64 = numpy_to_base64(pipeline.current, ext=".png")
    hist_data = metrics.compute_histogram_data(pipeline.current)
    eval_stats = pipeline.evaluate()

    return jsonify({
        "current_image": curr_b64,
        "histograms": hist_data,
        "metrics": eval_stats,
        "recipe": pipeline.get_recipe(),
    })


@app.route("/api/reset", methods=["POST"])
def reset_action():
    session_id = request.json.get("session_id")
    if not session_id or session_id not in SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    pipeline: ImagePipeline = SESSIONS[session_id]["pipeline"]
    pipeline.reset()

    curr_b64 = numpy_to_base64(pipeline.current, ext=".png")
    hist_data = metrics.compute_histogram_data(pipeline.current)
    eval_stats = pipeline.evaluate()

    return jsonify({
        "current_image": curr_b64,
        "histograms": hist_data,
        "metrics": eval_stats,
        "recipe": pipeline.get_recipe(),
    })


@app.route("/api/download", methods=["GET"])
def download_image():
    session_id = request.args.get("session_id")
    fmt = request.args.get("format", "png").lower()
    quality = int(request.args.get("quality", 95))

    if not session_id or session_id not in SESSIONS:
        return "Invalid session", 400

    pipeline: ImagePipeline = SESSIONS[session_id]["pipeline"]
    current_img = pipeline.current

    if fmt in ["jpg", "jpeg"]:
        ext = ".jpg"
        mimetype = "image/jpeg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "webp":
        ext = ".webp"
        mimetype = "image/webp"
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        ext = ".png"
        mimetype = "image/png"
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, buffer = cv2.imencode(ext, current_img, encode_params)
    if not success:
        return "Encoding error", 500

    byte_io = io.BytesIO(buffer)
    base_name = os.path.splitext(SESSIONS[session_id]["filename"])[0]
    out_filename = f"enhanced_{base_name}{ext}"

    return send_file(
        byte_io,
        mimetype=mimetype,
        as_attachment=True,
        download_name=out_filename,
    )


@app.route("/api/batch-process", methods=["POST"])
def batch_process():
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist("images")
    preset_name = request.form.get("preset", "landscape_vivid")

    if preset_name not in presets.PRESETS_REGISTRY:
        return jsonify({"error": f"Unknown preset: {preset_name}"}), 400

    preset_func = presets.PRESETS_REGISTRY[preset_name]["function"]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in files:
            if not file.filename:
                continue
            try:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                pipe = ImagePipeline(img)
                pipe._apply_op(f"preset_{preset_name}", preset_func)

                success, enc_buf = cv2.imencode(".png", pipe.current)
                if success:
                    clean_name = f"enhanced_{file.filename}"
                    zip_file.writestr(clean_name, enc_buf.tobytes())
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"enhanced_batch_{preset_name}.zip",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n===========================================================")
    print(f" * Image Enhancement Studio running at: http://127.0.0.1:{port}")
    print(f"===========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
