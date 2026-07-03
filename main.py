import streamlit as st
from ultralytics import YOLO
import cv2
import time
from PIL import Image
import numpy as np
import re
import threading
import tempfile
import os
import torch
import supervision as sv
from tracker import SimpleTracker
import ssl
# Bỏ qua xác thực chứng chỉ SSL trên macOS để tải file ONNX tự động từ GitHub
ssl._create_default_https_context = ssl._create_unverified_context
from fast_plate_ocr import LicensePlateRecognizer as FastPlateRecognizer

# ----------------------------
# Helper: load models once
# ----------------------------
@st.cache_resource
def load_models(yolo_path="files_model/license_plate_detector_yolov8.pt"):
    # Xác định thiết bị tối ưu nhất
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    yolo = YOLO(yolo_path)
    
    # Khởi tạo mô hình fast-plate-ocr
    # cct-s-v2-global-model tự động tìm execution provider tốt nhất (CUDA/CoreML/CPU)
    ocr_model = FastPlateRecognizer('cct-s-v2-global-model')
        
    return yolo, ocr_model, None, device

# ----------------------------
# Recognizer class (same logic as yours)
# ----------------------------
class LicensePlateRecognizer:
    def __init__(self, yolo, ocr_model, ocr_tokenizer, device="cpu"):
        self.yolo = yolo
        self.ocr_model = ocr_model
        self.ocr_tokenizer = ocr_tokenizer
        self.device = device

    def detect_plates(self, image):
        # image: BGR numpy array
        # Chạy YOLO trên GPU/MPS nếu khả dụng
        yolo_device = self.device if self.device in ["cuda", "mps"] else "cpu"
        results = self.yolo.predict(image, device=yolo_device, conf=0.6, verbose=False)[0] # ngưỡng xác định biển số 
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # ensure coordinates within image
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            plate_img = image[y1:y2, x1:x2]
            plates.append((plate_img, (x1, y1, x2, y2)))
        return plates

    def extract_text(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return ""
        try:
            # fast-plate-ocr chạy trực tiếp trên numpy BGR
            predictions = self.ocr_model.run(plate_img)
            return predictions[0].plate if predictions else ""
        except Exception as e:
            return ""

    def preprocess_plate_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip().upper()
        # Chuyển Đ thành D để đồng bộ xử lý ký tự không dấu
        text = text.replace('Đ', 'D')
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        if len(text) < 3:
            return text
            
        # Bảng chuyển chữ cái sang chữ số
        letter_to_digit = {
            'A': '4', 'B': '8', 'D': '0', 'G': '6', 'I': '1', 
            'O': '0', 'Q': '0', 'S': '5', 'T': '7', 'Z': '2'
        }
        
        # 1. Ký tự 1 và 2 (index 0, 1) luôn là chữ số mã tỉnh
        char_0 = letter_to_digit.get(text[0], text[0])
        char_1 = letter_to_digit.get(text[1], text[1])
        text = char_0 + char_1 + text[2:]
        
        # 2. Ký tự 3 (index 2) luôn là chữ cái series
        digit_to_letter = {
            '0': 'D', '1': 'L', '2': 'Z', '3': 'E', '4': 'A', 
            '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
        }
        char_2 = text[2]
        if char_2.isdigit():
            char_2 = digit_to_letter.get(char_2, 'L')  # Mặc định số 1 -> L
        elif char_2 == 'I':
            char_2 = 'L'  # Lỗi nghiêng chữ L -> I rất phổ biến
        elif char_2 == 'O':
            char_2 = 'D'
        text = text[:2] + char_2 + text[3:]
        
        # 3. Ký tự 4 (index 3) có thể là số hoặc chữ -> Không cần xét điều kiện, giữ nguyên
        # 4. Ký tự 5 trở đi (index 4 trở đi) luôn là chữ số
        if len(text) >= 5:
            fixed_rest = ""
            for char in text[4:]:
                if char.isalpha():
                    fixed_rest += letter_to_digit.get(char, '1')
                else:
                    fixed_rest += char
            text = text[:4] + fixed_rest
            
        return text


def compute_iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def vote_text(ocr_history):
    if not ocr_history:
        return "", None
    
    # 1. Đếm tần suất xuất hiện của từng chuỗi biển số
    counts = {}
    for text, area, img in ocr_history:
        if text.strip() != "":
            counts[text] = counts.get(text, 0) + 1
            
    if not counts:
        # Nếu chưa nhận diện được chữ nào hợp lệ, lấy mẫu đầu tiên làm đại diện
        return ocr_history[0][0], ocr_history[0][2]
        
    # 2. Tìm tần suất xuất hiện lớn nhất
    max_count = max(counts.values())
    
    # 3. Lọc ra các chuỗi đạt số phiếu bầu cao nhất
    candidates = [text for text, cnt in counts.items() if cnt == max_count]
    
    if len(candidates) == 1:
        # Thắng tuyệt đối
        winner = candidates[0]
    else:
        # Hòa phiếu -> Ưu tiên chọn biển số có diện tích ảnh cắt to nhất (rõ nhất)
        best_area = -1
        winner = candidates[0]
        for text, area, img in ocr_history:
            if text in candidates and area > best_area:
                best_area = area
                winner = text
                
    # Tìm ảnh đại diện rõ nhất (area lớn nhất) của chuỗi thắng cuộc
    winner_img = None
    best_area_for_winner = -1
    for text, area, img in ocr_history:
        if text == winner and area > best_area_for_winner:
            best_area_for_winner = area
            winner_img = img
            
    return winner, winner_img





# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="LPR - Real-time", page_icon="🚘", layout="wide")
st.title("🚘 License Plate Recognition - Image & Real-time Stream")

# Load models once
with st.spinner("Loading models (YOLO + OCR)... this can take a while"):
    yolo_model, ocr_model, ocr_tokenizer, device = load_models()
    recognizer = LicensePlateRecognizer(yolo_model, ocr_model, ocr_tokenizer, device)

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ("Image Upload", "Video Upload", "Webcam (local)", "RTSP / IP Camera"))

# common controls
display_fps = st.sidebar.checkbox("Show FPS", value=True)
show_boxes = st.sidebar.checkbox("Show bounding boxes & text", value=True)
max_boxes = st.sidebar.slider("Max plates to display per frame", 1, 10, 5)
process_every_n_frame = st.sidebar.slider("Process every N-th frame (video)", 1, 30, 7)

# ----------------------------
# IMAGE UPLOAD
# ----------------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        plates = recognizer.detect_plates(image)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original image", width="stretch")
        with col2:
            if not plates:
                st.warning("No plates detected.")
            else:
                start = time.time()

                for i, (plate_img, (x1, y1, x2, y2)) in enumerate(plates[:max_boxes]):
                    text = recognizer.extract_text(plate_img)
                    text_clean = recognizer.preprocess_plate_text(text)

                    # Hiển thị ảnh
                    st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

                    # Hiển thị caption to, màu đỏ
                    st.markdown(
                        f"<h3 style='color:red; text-align:left;'>Plate #{i+1}: {text_clean}</h3>",
                        unsafe_allow_html=True
                    )
                
                elapsed = time.time() - start
                st.write('\n⏱️ Thời gian xử lý: {:02d}:{:02d}:{:02d}'.format(
                    int(elapsed // 3600),
                    int((elapsed % 3600) // 60),
                    int(elapsed % 60)
                ))

# ----------------------------
# VIDEO UPLOAD
# ----------------------------
elif mode == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Khởi tạo các khung hiển thị trực tiếp
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Đang xử lý và phát video trực quan...")
        video_placeholder = st.empty()
        gallery_placeholder = st.empty()

        frame_count = 0
        start_time = time.time()
        last_time = time.time()
        fps = 0.0

        # Bộ nhớ để tránh trùng
        detected_plates = []       # lưu ảnh + text
        seen_texts = set()         # chỉ lưu text duy nhất
        update_gallery = False
        
        # Khởi tạo SimpleTracker và bộ nhớ ocr_results cho video này
        tracker = SimpleTracker(iou_threshold=0.2, max_missed=15)
        ocr_results = {}           # {tracker_id: {'text': str, 'history': [(text, area, img)], 'last_ocr_frame': int}}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            is_processed = False
            if frame_count % process_every_n_frame == 0:
                is_processed = True
                
                # 1. Chạy YOLOv8 tìm tất cả các biển số trong frame
                yolo_device = recognizer.device if recognizer.device in ["cuda", "mps"] else "cpu"
                results = recognizer.yolo.predict(frame, device=yolo_device, conf=0.6, verbose=False)[0]
                
                # 2. Chuyển đổi sang supervision và cập nhật tracker
                detections = sv.Detections.from_ultralytics(results)
                tracked_detections = tracker.update(detections, frame_idx=frame_count)
                
                # 3. Duyệt qua các đối tượng được tracker gán ID ổn định
                if tracked_detections.tracker_id is not None:
                    for bbox, conf, class_id, tracker_id in zip(
                        tracked_detections.xyxy,
                        tracked_detections.confidence,
                        tracked_detections.class_id,
                        tracked_detections.tracker_id
                    ):
                        x1, y1, x2, y2 = map(int, bbox)
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        tid = int(tracker_id)
                        det_area = (x2 - x1) * (y2 - y1)
                        plate_img = frame[y1:y2, x1:x2]
                        
                        if tid not in ocr_results:
                            # Xe mới -> Chạy OCR lần đầu
                            text = recognizer.extract_text(plate_img)
                            text_clean = recognizer.preprocess_plate_text(text)
                            
                            # Chỉ lưu nếu chiều dài ký tự hợp lệ (7-10 ký tự)
                            if 7 <= len(text_clean) <= 10:
                                ocr_results[tid] = {
                                    'text': text_clean,
                                    'history': [(text_clean, det_area, plate_img.copy())],
                                    'last_ocr_frame': frame_count
                                }
                                if text_clean not in seen_texts:
                                    seen_texts.add(text_clean)
                                    detected_plates.append((plate_img.copy(), text_clean))
                                    update_gallery = True
                            else:
                                ocr_results[tid] = {
                                    'text': 'Detecting...',
                                    'history': [],
                                    'last_ocr_frame': frame_count
                                }
                        else:
                            # Xe cũ -> Thu thập thêm mẫu để bỏ phiếu số đông (tối đa 5 mẫu, cách nhau ít nhất 3 lần quét)
                            history = ocr_results[tid]['history']
                            if len(history) < 5 and (frame_count - ocr_results[tid]['last_ocr_frame'] >= 3 * process_every_n_frame):
                                text = recognizer.extract_text(plate_img)
                                text_clean = recognizer.preprocess_plate_text(text)
                                
                                # Chỉ chấp nhận mẫu có chiều dài hợp lệ (7-10 ký tự)
                                if 7 <= len(text_clean) <= 10:
                                    history.append((text_clean, det_area, plate_img.copy()))
                                    ocr_results[tid]['last_ocr_frame'] = frame_count
                                    
                                    # Bầu chọn biển số thắng cuộc dựa trên số phiếu
                                    old_winner = ocr_results[tid]['text']
                                    new_winner, winner_img = vote_text(history)
                                    ocr_results[tid]['text'] = new_winner
                                    
                                    if new_winner != old_winner:
                                        # Xóa text cũ thắng cuộc trước đó khỏi danh sách
                                        if old_winner in seen_texts:
                                            seen_texts.remove(old_winner)
                                            detected_plates = [item for item in detected_plates if item[1] != old_winner]
                                        
                                        # Thêm text mới thắng cuộc
                                        if new_winner.strip() != "" and new_winner not in seen_texts:
                                            seen_texts.add(new_winner)
                                            detected_plates.append((winner_img.copy(), new_winner))
                                            update_gallery = True
                                    elif new_winner.strip() != "":
                                        # Nếu biển số giữ nguyên nhưng có ảnh tốt hơn, cập nhật ảnh rõ hơn
                                        for idx, (p_img, t_clean) in enumerate(detected_plates):
                                            if t_clean == new_winner:
                                                detected_plates[idx] = (winner_img.copy(), new_winner)
                                                update_gallery = True
                                                break

            # Vẽ khung và thông tin lên frame hiện tại từ các active tracks
            if show_boxes:
                for t in tracker.tracks:
                    if t.missed == 0:  # Chỉ vẽ khi xe đang xuất hiện (missed == 0)
                        x1, y1, x2, y2 = map(int, t.bbox_xyxy())
                        tid = int(t.id)
                        text_clean = ocr_results.get(tid, {}).get('text', 'Detecting...')
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text_clean, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if display_fps:
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
                last_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Cập nhật frame lên giao diện Streamlit
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

            # Cập nhật lưới ảnh Gallery thời gian thực
            if update_gallery and detected_plates:
                with gallery_placeholder.container():
                    st.markdown("### 🚘 Biển số nhận diện được")
                    cols_per_row = 4
                    rows = (len(detected_plates) + cols_per_row - 1) // cols_per_row
                    idx = 0
                    for r in range(rows):
                        cols = st.columns(cols_per_row)
                        for c in range(cols_per_row):
                            if idx < len(detected_plates):
                                p_img, t_clean = detected_plates[idx]
                                with cols[c]:
                                    st.image(
                                        cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB),
                                        caption=f"**{t_clean}**",
                                        width="stretch",
                                    )
                                idx += 1
                update_gallery = False

            # Delay nhẹ ở frame xen kẽ để tránh phát video quá nhanh
            if not is_processed:
                time.sleep(0.01)

        cap.release()

        elapsed = time.time() - start_time
        status_placeholder.success(
            '\n⏱️ Thời gian xử lý: {:02d}:{:02d}:{:02d}'.format(
                int(elapsed // 3600),
                int((elapsed % 3600) // 60),
                int(elapsed % 60),
            )
        )

        print("\nDone!")


# ----------------------------
# Webcam (local), RTSP / IP Camera
# ----------------------------                   
elif mode in ("Webcam (local)", "RTSP / IP Camera"):
    if mode == "Webcam (local)":
        src = st.sidebar.text_input("Webcam index", "0")
    else:
        src = st.sidebar.text_input("RTSP/HTTP URL", "rtsp://admin:admin@192.168.10.114:554/unicaststream/1")

    # Quản lý trạng thái luồng phát trong session state
    if "run_stream" not in st.session_state:
        st.session_state.run_stream = False

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Stream")
    with col2:
        stop_button = st.button("Stop Stream")

    # Vùng hiển thị video và thông báo
    video_slot = st.empty()
    info_slot = st.empty()
    gallery_placeholder = st.empty()

    if start_button:
        st.session_state.run_stream = True
        st.rerun()

    if stop_button:
        st.session_state.run_stream = False
        st.rerun()

    if st.session_state.run_stream:
        source = int(src) if mode == "Webcam (local)" and str(src).isdigit() else src
        
        # Thiết lập cấu hình tối ưu cho luồng RTSP qua TCP
        if isinstance(source, str) and source.startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if cap is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap is None or not cap.isOpened():
            info_slot.error(f"❌ Không thể kết nối tới nguồn video: {source} (Hãy kiểm tra lại URL, thiết bị hoặc tài khoản login)")
            st.session_state.run_stream = False
        else:
            info_slot.info("🔌 Đang hiển thị luồng trực tiếp từ camera...")
            
            tracker = SimpleTracker(iou_threshold=0.2, max_missed=15)
            ocr_results = {}
            detected_plates = []       # lưu ảnh + text
            seen_texts = set()         # chỉ lưu text duy nhất
            update_gallery = False
            
            last_time = time.time()
            fps = 0.0
            frame_idx = 0
            
            try:
                while st.session_state.run_stream:
                    ret, frame = cap.read()
                    if not ret:
                        info_slot.warning("⚠️ Mất kết nối luồng video từ camera.")
                        break

                    frame_idx += 1
                    
                    # 1. Chạy YOLOv8 tìm tất cả các biển số trong frame
                    yolo_device = recognizer.device if recognizer.device in ["cuda", "mps"] else "cpu"
                    results = recognizer.yolo.predict(frame, device=yolo_device, conf=0.6, verbose=False)[0]
                    
                    # 2. Chuyển đổi sang supervision và cập nhật tracker
                    detections = sv.Detections.from_ultralytics(results)
                    tracked_detections = tracker.update(detections, frame_idx=frame_idx)
                    
                    # 3. Duyệt qua các đối tượng được tracker gán ID ổn định
                    if tracked_detections.tracker_id is not None:
                        for bbox, conf, class_id, tracker_id in zip(
                            tracked_detections.xyxy,
                            tracked_detections.confidence,
                            tracked_detections.class_id,
                            tracked_detections.tracker_id
                        ):
                            x1, y1, x2, y2 = map(int, bbox)
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            tid = int(tracker_id)
                            det_area = (x2 - x1) * (y2 - y1)
                            plate_img = frame[y1:y2, x1:x2]
                            
                            if tid not in ocr_results:
                                # Xe mới -> Chạy OCR lần đầu
                                text = recognizer.extract_text(plate_img)
                                text_clean = recognizer.preprocess_plate_text(text)
                                
                                # Chỉ lưu nếu chiều dài ký tự hợp lệ (7-10 ký tự)
                                if 7 <= len(text_clean) <= 10:
                                    ocr_results[tid] = {
                                        'text': text_clean,
                                        'history': [(text_clean, det_area, plate_img.copy())],
                                        'last_ocr_frame': frame_idx
                                    }
                                    if text_clean not in seen_texts:
                                        seen_texts.add(text_clean)
                                        detected_plates.append((plate_img.copy(), text_clean))
                                        update_gallery = True
                                else:
                                    ocr_results[tid] = {
                                        'text': 'Detecting...',
                                        'history': [],
                                        'last_ocr_frame': frame_idx
                                    }
                            else:
                                # Xe cũ -> Thu thập thêm mẫu để bỏ phiếu (tối đa 5 mẫu, cách nhau ít nhất 6 frame ~ 0.2 giây)
                                history = ocr_results[tid]['history']
                                if len(history) < 5 and (frame_idx - ocr_results[tid]['last_ocr_frame'] >= 6):
                                    text = recognizer.extract_text(plate_img)
                                    text_clean = recognizer.preprocess_plate_text(text)
                                    
                                    # Chỉ chấp nhận mẫu có chiều dài hợp lệ (7-10 ký tự)
                                    if 7 <= len(text_clean) <= 10:
                                        history.append((text_clean, det_area, plate_img.copy()))
                                        ocr_results[tid]['last_ocr_frame'] = frame_idx
                                        
                                        # Chạy bỏ phiếu
                                        old_winner = ocr_results[tid]['text']
                                        new_winner, winner_img = vote_text(history)
                                        ocr_results[tid]['text'] = new_winner

                                        if new_winner != old_winner:
                                            # Xóa text cũ thắng cuộc trước đó khỏi danh sách
                                            if old_winner in seen_texts:
                                                seen_texts.remove(old_winner)
                                                detected_plates = [item for item in detected_plates if item[1] != old_winner]
                                            
                                            # Thêm text mới thắng cuộc
                                            if new_winner.strip() != "" and new_winner not in seen_texts:
                                                seen_texts.add(new_winner)
                                                detected_plates.append((winner_img.copy(), new_winner))
                                                update_gallery = True
                                        elif new_winner.strip() != "":
                                            # Nếu biển số giữ nguyên nhưng có ảnh tốt hơn, cập nhật ảnh rõ hơn
                                            for idx, (p_img, t_clean) in enumerate(detected_plates):
                                                if t_clean == new_winner:
                                                    detected_plates[idx] = (winner_img.copy(), new_winner)
                                                    update_gallery = True
                                                    break

                    # Vẽ khung và thông tin lên frame hiện tại từ các active tracks
                    if show_boxes:
                        for t in tracker.tracks:
                            if t.missed == 0:  # Chỉ vẽ khi xe đang xuất hiện (missed == 0)
                                x1, y1, x2, y2 = map(int, t.bbox_xyxy())
                                tid = int(t.id)
                                text_clean = ocr_results.get(tid, {}).get('text', 'Detecting...')
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, text_clean, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if display_fps:
                        now = time.time()
                        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
                        last_time = now
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    # hiển thị frame
                    video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

                    # Cập nhật danh sách ảnh cắt biển số (gallery) ở dưới thời gian thực
                    if update_gallery and detected_plates:
                        with gallery_placeholder.container():
                            st.markdown("### 🚘 Biển số nhận diện được")
                            cols_per_row = 4
                            rows = (len(detected_plates) + cols_per_row - 1) // cols_per_row
                            idx = 0
                            for r in range(rows):
                                cols = st.columns(cols_per_row)
                                for c in range(cols_per_row):
                                    if idx < len(detected_plates):
                                        p_img, t_clean = detected_plates[idx]
                                        with cols[c]:
                                            st.image(
                                                cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB),
                                                caption=f"**{t_clean}**",
                                                width="stretch",
                                            )
                                        idx += 1
                        update_gallery = False

                    # delay nhẹ
                    time.sleep(0.01)
            except Exception as e:
                info_slot.error(f"Stream error: {e}")
            finally:
                cap.release()
                st.session_state.run_stream = False

# Footer / tips
st.markdown("---")
st.write("**Mẹo:**")
st.write("- Với luồng RTSP, hãy sử dụng URL RTSP của camera (thường có dạng rtsp://user:pass@ip:554/...).")
st.write("- Sử dụng GPU (PyTorch + CUDA) để OCR và YOLO chạy nhanh hơn. Nếu không có GPU, hãy giảm tốc độ khung hình.")
st.write("- Nếu luồng RTSP bị lỗi, thử tăng thời gian chờ mạng hoặc kiểm tra thông tin đăng nhập.")
