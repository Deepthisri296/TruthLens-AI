import sys
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.predict import predict_image

TESTING_IMAGES_FOLDER = BASE_DIR / "testing_images"
RESULTS_FOLDER = BASE_DIR / "evaluation" / "results"

TESTING_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "app" / "templates"),
    static_folder=str(BASE_DIR / "app" / "static")
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/testing-images/<path:filename>")
def testing_image(filename):
    return send_from_directory(
        TESTING_IMAGES_FOLDER,
        filename
    )


@app.route("/results/<path:filename>")
def result_image(filename):
    return send_from_directory(
        RESULTS_FOLDER,
        filename
    )


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify({
            "error": "No image was selected."
        }), 400

    image = request.files["image"]

    if not image.filename:
        return jsonify({
            "error": "Please select an image."
        }), 400

    if not allowed_file(image.filename):
        return jsonify({
            "error": "Only PNG, JPG, and JPEG files are allowed."
        }), 400

    original_name = secure_filename(image.filename)

    extension = Path(original_name).suffix.lower()

    filename = (
        f"{Path(original_name).stem}_"
        f"{uuid4().hex[:8]}"
        f"{extension}"
    )

    filepath = TESTING_IMAGES_FOLDER / filename

    image.save(filepath)

    try:

        result = predict_image(str(filepath))

        if not isinstance(result, dict):
            return jsonify({
                "error": "Model returned an invalid result."
            }), 500

        heatmap_path = result.get("heatmap_path")

        if heatmap_path:

            heatmap_filename = Path(
                heatmap_path
            ).name

            result["heatmap_url"] = (
                f"/results/{heatmap_filename}"
            )

        result["original_image_url"] = (
            f"/testing-images/{filename}"
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "result": result
        })

    except Exception as error:

        app.logger.exception(
            "Model analysis failed"
        )

        return jsonify({
            "error": (
                "Image was uploaded, "
                f"but analysis failed: {error}"
            )
        }), 500


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "File size must be less than 10 MB."
    }), 413


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )