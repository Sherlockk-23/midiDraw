from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

@app.route("/generate-midi", methods=["POST"])
def generate():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes))
    image.save("input.png")

    #generate output.mid from input.png

    return send_file("output.mid", mimetype="audio/midi")

if __name__ == "__main__":
    app.run(port=5000)
