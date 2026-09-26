import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pcb_yolo11n"
    / "weights"
    / "best.pt"
)


# ============================================================
# YOLO SETTINGS
# ============================================================

INFERENCE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = YOLO(str(MODEL_PATH))

    return model


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔍 PCB Defect Detection System")

st.write(
    "Detect Printed Circuit Board (PCB) defects "
    "using a trained YOLO11n Deep Learning model."
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error("❌ YOLO model not found.")

    st.code(str(MODEL_PATH))

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error("❌ Failed to load YOLO model.")

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** YOLO11n")

    st.write("**Inference Size:** 640")

    st.write("**Confidence:** 0.25")

    st.write("**IoU:** 0.45")

    st.markdown("---")

    st.write(
        "The trained YOLO11n model detects "
        "defects in PCB images."
    )


# ============================================================
# IMAGE INPUT OPTIONS
# ============================================================

st.subheader("📷 PCB Image Input")

st.write(
    "Choose one of the following options "
    "to provide a PCB image."
)


# ------------------------------------------------------------
# OPTION 1 — IMAGE UPLOAD
# ------------------------------------------------------------

st.markdown("### 📁 Option 1: Upload PCB Image")

uploaded_file = st.file_uploader(
    "Choose a PCB image",
    type=["jpg", "jpeg", "png"]
)


# ------------------------------------------------------------
# OPTION 2 — WEB CAMERA
# ------------------------------------------------------------

st.markdown("### 📷 Option 2: Use Web Camera")

camera_image = st.camera_input(
    "Capture PCB using Web Camera"
)


# ============================================================
# SELECT INPUT IMAGE
# ============================================================

image = None

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

elif camera_image is not None:

    image = Image.open(
        camera_image
    ).convert("RGB")


# ============================================================
# IMAGE PROCESSING
# ============================================================

if image is not None:

    # --------------------------------------------------------
    # ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    width, height = image.size

    st.info(
        f"Original Image Size: "
        f"{width} × {height} pixels"
    )


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader("📷 Original PCB Image")

    st.image(
        image,
        caption=f"Original PCB Image ({width} × {height})",
        use_container_width=True
    )


    # --------------------------------------------------------
    # DETECTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Detect PCB Defects",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🔄 YOLO11n is detecting PCB defects..."
        ):

            try:

                # ------------------------------------------------
                # YOLO PREDICTION
                # ------------------------------------------------

                results = model.predict(
                    source=image,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )


                # ------------------------------------------------
                # FIRST RESULT
                # ------------------------------------------------

                result = results[0]


                # ------------------------------------------------
                # DRAW DETECTIONS
                # ------------------------------------------------

                annotated_image = result.plot()

                # Convert BGR → RGB
                annotated_image = (
                    annotated_image[:, :, ::-1]
                )


                # ------------------------------------------------
                # DISPLAY RESULT
                # ------------------------------------------------

                st.subheader("🎯 Detection Result")

                st.image(
                    annotated_image,
                    caption="YOLO11n Detection Result",
                    use_container_width=True
                )


                # ------------------------------------------------
                # DETECTION DETAILS
                # ------------------------------------------------

                boxes = result.boxes


                if boxes is not None and len(boxes) > 0:

                    st.success(
                        f"✅ {len(boxes)} "
                        f"defect(s) detected."
                    )


                    st.subheader(
                        "📊 Detection Details"
                    )


                    # --------------------------------------------
                    # DISPLAY EACH DETECTION
                    # --------------------------------------------

                    for index, box in enumerate(
                        boxes,
                        start=1
                    ):

                        # Class ID
                        class_id = int(
                            box.cls[0].item()
                        )


                        # Confidence
                        confidence = float(
                            box.conf[0].item()
                        )


                        # Class name
                        class_name = (
                            model.names[class_id]
                        )


                        # Bounding box
                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                            .astype(int)
                        )


                        # ----------------------------------------
                        # DISPLAY DETECTION
                        # ----------------------------------------

                        st.write(
                            f"### 🔴 Defect {index}"
                        )

                        st.write(
                            f"**Defect Type:** "
                            f"{class_name}"
                        )

                        st.write(
                            f"**Confidence:** "
                            f"{confidence * 100:.2f}%"
                        )

                        st.write(
                            f"**Bounding Box:** "
                            f"({x1}, {y1}) → "
                            f"({x2}, {y2})"
                        )

                        st.divider()


                else:

                    st.success(
                        "✅ No PCB defects detected."
                    )


            except Exception as error:

                st.error(
                    "❌ Error during YOLO prediction."
                )

                st.exception(error)


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.markdown("---")

st.subheader("ℹ️ How It Works")

st.write(
    """
    1. Upload a PCB image or capture an image using the webcam.
    2. The original image size is displayed.
    3. YOLO11n preprocesses the image.
    4. The model performs defect detection at 640 pixels.
    5. Detected defects are displayed with bounding boxes.
    6. The defect class and confidence score are displayed.
    7. Bounding-box coordinates are displayed.
    """
)


# ============================================================
# MANUFACTURING APPLICATION
# ============================================================

st.markdown("---")

st.subheader("🏭 Manufacturing Application")

st.write(
    """
    This system can be extended for automated PCB quality
    inspection in manufacturing environments, where a camera
    continuously monitors PCBs moving through an inspection area.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "PCB Defect Detection | "
    "YOLO11n + Ultralytics + Streamlit"
)