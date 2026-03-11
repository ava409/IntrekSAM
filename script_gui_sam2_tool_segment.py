
#TODO: Some optimisation to speed up inference? (Since it is maxing out at about 11fps)
#TODO: Load as frames asynchronesly (faster startup and also fixes the long video problem!)
#TODO: Move inference to another thread, so that it is not blocked by GUI / waits for gui to updat
#TODO: Fix sam2 lack of post-proccessing!!
#TODO: Maybe number key shortcuts for class selection

import os
import sys
import cv2
import json
import torch
import numpy as np
from PIL import Image, ImageColor
from natsort import natsorted
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from PyQt5 import QtCore, QtWidgets, QtGui

from pycocotools.mask import decode as decode_masks, encode as encode_masks
from sam2.demo.backend.server.inference.predictor import (
    InferenceAPI,
    AddMaskRequest,
    AddPointsRequest,
    ClearPointsInFrameRequest,
    Mask as SamMask,
    PropagateInVideoRequest,
    StartSessionRequest,
    CloseSessionRequest
)

LABEL_INFO_FILE = "/gris/gris-f/homestud/ssivakum/SurgSimBridge/ann/Cataract-1K/ann_tool_classes.json"
INPUT_VIDEO_DIR = "/local/scratch/sharvien/SurgSimBridge/Cataract-1K/videos"
OUTPUT_ANN_DIR = "/local/scratch/adhingra/Cataract_1K/mask_tools"

# LABEL_INFO_FILE = "/gris/gris-f/homestud/ssivakum/SurgSimBridge/ann/Cataracts-50/ann_tool_classes.json"
# INPUT_VIDEO_DIR = "/local/scratch/sharvien/SurgSimBridge/Cataracts-50/videos"
# OUTPUT_ANN_DIR = "/local/scratch/sharvien/SurgSimBridge/Cataracts-50/masks_tool"

SKIP_LABEL = ["1", "2", "3", "9", "10"]
VRAM_FRAME_LIMIT = 1700
RAM_FRAME_LIMIT = 2100

@dataclass
class Prompt:
    """
    Stores point prompts for segmentation. 
    Tuple[frame_idx, x, y, label]
    """
    points: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    def clear(self, frame_idx: int):
        self.points = [p for p in self.points if p[0] != frame_idx]


@dataclass
class Mask:
    """
    Stores predicted masks per frame.
    Dict[frame_idx, mask]
    """
    masks: Dict[np.ndarray, int] = field(default_factory=dict)

    def update(self, mask: np.ndarray, frame_idx: int):
        self.masks[frame_idx] = mask

    def get(self, frame_idx: int) -> np.ndarray | None:
        return self.masks.get(frame_idx)
    
    def clear(self, frame_idx: int):
        if frame_idx in self.masks:
            del self.masks[frame_idx]


@dataclass
class AnnObject:
    id: int
    name: str
    color: Tuple[int, int, int, int]
    prompts: Prompt = field(default_factory=Prompt)
    masks: Mask = field(default_factory=Mask)


class SAM2_Video_Annotator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Video Annotator")
        self.setGeometry(50, 50, 800, 450)
        self.video_path = None
        self.output_ann_path = OUTPUT_ANN_DIR
        self.label_info_path = LABEL_INFO_FILE
        self.skip_label = SKIP_LABEL
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_size = (512, 288)
        self.play_fps = 60
        self.total_frames = 0
        self.current_frame_idx = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.sam_api = InferenceAPI()
        self.init_obj()
        self.init_ui()
        

    def init_obj(self):
        self.objects: List[AnnObject] = []
        self.active_object_id: Optional[int] = None
        self.mask_palette = []

        # Read class info from path and create AnnObject instances for them
        with open(self.label_info_path, "r") as f:
            class_info = json.load(f)
        
        # Make sure there is a background class with id 0
        # Background class with fully false mask to avoid no prompt error and uninterrupted propagation
        assert class_info["0"]["name"].lower() == 'background', "Background class does not match expected definition."               
        for cls in class_info:
            color = ImageColor.getrgb(class_info[cls]["color"])
            self.mask_palette.append(color)
            if cls in self.skip_label:
                continue
            obj = AnnObject(id=cls, name=class_info[cls]["name"], color=tuple(color))
            self.objects.append(obj)
            

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        slider_layout = QtWidgets.QHBoxLayout()
        display_layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QVBoxLayout()

        # Video and annotated mask display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(self.frame_size[0], self.frame_size[1])
        self.video_label.setScaledContents(True)
        self.video_label.installEventFilter(self)
        self.current_qimage = None

        self.vid_title_status = QtWidgets.QLabel("")
        display_layout.addWidget(self.vid_title_status)
        display_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignCenter)

        self.move_backward_btn = QtWidgets.QPushButton("<<")
        self.move_backward_btn.setFixedSize(40, 30)
        self.move_backward_btn.clicked.connect(lambda: self.goto_frame(self.current_frame_idx - 1))
        self.move_backward_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        slider_layout.addWidget(self.move_backward_btn)
        self.move_forward_btn = QtWidgets.QPushButton(">>")
        self.move_forward_btn.setFixedSize(40, 30)
        self.move_forward_btn.clicked.connect(lambda: self.goto_frame(self.current_frame_idx + 1))
        self.move_forward_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        slider_layout.addWidget(self.move_forward_btn)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        slider_layout.addWidget(self.slider)
        display_layout.addLayout(slider_layout)

        self.frame_status = QtWidgets.QLabel("")
        display_layout.addWidget(self.frame_status)
        self.status = QtWidgets.QLabel("")
        display_layout.addWidget(self.status)

        # Right top panel with buttons for controls
        btn_load = QtWidgets.QPushButton("Load Video")
        btn_load.clicked.connect(self.select_video)
        btn_load.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(btn_load)
        btn_auto = QtWidgets.QPushButton("Load Auto")
        btn_auto.clicked.connect(self.select_auto)
        btn_auto.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(btn_auto)
        self.btn_load_next = QtWidgets.QPushButton("Load Next")
        self.btn_load_next.clicked.connect(self.next_video)
        self.btn_load_next.setEnabled(False)
        self.btn_load_next.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(self.btn_load_next)
        button_layout.addStretch()

        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(self.btn_play)
        btn_undo = QtWidgets.QPushButton("Undo Points")
        btn_undo.clicked.connect(self.undo_points)
        btn_undo.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(btn_undo)
        btn_clear = QtWidgets.QPushButton("Clear Annotations")
        btn_clear.clicked.connect(self.clear_annotations)
        btn_clear.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(btn_clear)
        btn_export = QtWidgets.QPushButton("Export Annotations")
        btn_export.clicked.connect(self.export_annotations)
        btn_export.setFocusPolicy(QtCore.Qt.NoFocus)
        button_layout.addWidget(btn_export)
        button_layout.addStretch()

        # Right bottom panel with buttons for class selection
        self.btn_class = QtWidgets.QButtonGroup(self)
        self.btn_class.setExclusive(True)

        for obj in self.objects:
            r, g, b = obj.color
            if obj.name.lower() == "background":
                continue
            btn = QtWidgets.QPushButton(obj.id + ". " + obj.name)
            btn.setCheckable(True)
            btn.setStyleSheet(f"""text-align: left; padding-left: 5px; color: rgb({r}, {g}, {b})""")
            btn.clicked.connect(lambda checked, obj_id=obj.id: self.on_obj_selected(obj_id))
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            self.btn_class.addButton(btn)
            button_layout.addWidget(btn)

        # instr_text = "\nInstructions: \n- Left-click: Add positive point \n- Right-click: Add negative point \n- Middle-click: Play/Pause"
        # lbl_instructions = QtWidgets.QLabel(instr_text)
        # button_layout.addWidget(lbl_instructions)
        # button_layout.addStretch()
        layout.addLayout(display_layout)
        layout.addLayout(button_layout)


    def select_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4)")
        if not path:
            QtWidgets.QMessageBox.critical(self, "Error", "Not a valid format")
            return
        self.video_path = path
        self.vid_title_status.setText(f"<b>Video Name:<b> {os.path.basename(self.video_path)}")
        self.btn_load_next.setEnabled(True)
        self.load_video()


    def select_auto(self):
        input_list = [i.removesuffix(".mp4") for i in os.listdir(INPUT_VIDEO_DIR)]
        ann_list = os.listdir(OUTPUT_ANN_DIR)
        to_ann_list = natsorted(list(set(input_list) - set(ann_list)))   
        self.video_path = os.path.join(INPUT_VIDEO_DIR, to_ann_list[0]+".mp4")
        self.vid_title_status.setText(f"<b>Video Name:<b> {os.path.basename(self.video_path)}")
        self.btn_load_next.setEnabled(True)
        self.load_video()


    def next_video(self):
        # Load the next video in the same directory after loading the current video
        parent_path = os.path.dirname(self.video_path)
        video_path = os.path.basename(self.video_path)
        video_files = [f for f in os.listdir(parent_path) if f.lower().endswith(('.mp4'))]
        video_files = natsorted(video_files)
    
        video_idx = video_files.index(video_path)
        next_video_idx = video_idx + 1
        if next_video_idx >= len(video_files):
            QtWidgets.QMessageBox.information(self, "Info", "No more videos in the directory")
            return
        next_video_path = os.path.join(parent_path, video_files[next_video_idx])
        self.video_path = next_video_path
        self.vid_title_status.setText(f"<b>Video Name:<b> {os.path.basename(self.video_path)}")
        self.load_video()


    def load_video(self):
        # Close previous sesssion if it exists 
        if hasattr(self, "start_response"):
            self.close_req = CloseSessionRequest(type="close_session", session_id=self.session_id)
            self.sam_api.close_session(self.close_req)
            torch.cuda.empty_cache()

        # Loads the video_cap, shows number of frames and shows first frame 
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Could not open video")
            return
        self.cap = cap
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.slider.setMaximum(max(0, self.total_frames - 1))
        self.current_frame_idx = 0

        offload_video_flag = self.total_frames > VRAM_FRAME_LIMIT
        sam2_video_path = self.video_path if self.total_frames < RAM_FRAME_LIMIT else self.video_path.replace("/videos/", "/video_frames_jpg/").replace(".mp4", "")

        self.start_req = StartSessionRequest(type="start_session", path=sam2_video_path)
        self.start_response = self.sam_api.start_session(self.start_req, offload_video_flag=offload_video_flag)
        self.session_id = self.start_response.session_id

        #TODO: Will repeat init_obj twice for initial/first loading. But needed for init_ui.
        self.init_obj()
        self.goto_frame(0)
        self.frame_status.setText(f"Loaded {os.path.basename(self.video_path)} — {self.total_frames} frames @ {self.cap.get(cv2.CAP_PROP_FPS):.2f} fps")


    def toggle_play(self):
        # Handle play/pause button and start or stop self.timer
        if not self.cap:
            return
        if self.playing:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.playing = False
        else:
            interval = int(max(1, round(1000.0 / self.play_fps)))
            self.timer.start(interval)
            self.btn_play.setText("Pause")
            self.playing = True
            self.start_propagation()


    def next_frame(self):
        # Advances to next frame when self.timer by calling goto_frame
        if not self.cap:
            return
        next_idx = self.current_frame_idx + 1
        if next_idx >= self.total_frames:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("Play")
            self.stop_propagation()
            return


    def goto_frame(self, idx: int, propagation: bool = False):
        # Handles showing the frame, including any overlays on the frame
        idx = max(0, min(idx, max(0, self.total_frames - 1)))
        
        # Optimisation using only self.cap.read() for propagation since only seeking forward by 1
        if not propagation:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, self.frame_size, interpolation=cv2.INTER_AREA)
        self.current_frame_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        overlay = self.render_overlays(frame_rgb)
        self.video_label_set_frame(overlay)
        self.frame_status.setText(f"Frame {self.current_frame_idx+1}/{self.total_frames}")


    def on_slider_changed(self, val: int):
        self.goto_frame(val)


    def on_obj_selected(self, id: int):
        self.video_label.active_object_id = id
        self.video_label.update()


    def get_object_by_id(self, id: int) -> Optional[AnnObject]:
        for obj in self.objects:
            if obj.id == id:
                return obj


    def is_prompt_applied(self) -> bool:
        session = self.sam_api._InferenceAPI__get_session(self.session_id)
        return len(session["state"]["obj_ids"]) > 0


    def start_propagation(self):
        start_idx = self.current_frame_idx
        prop_req = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=self.session_id,
            start_frame_index=start_idx
        )
        # Background class with fully false mask to avoid no prompt error and uninterrupted propagation
        obj = self.get_object_by_id("0")
        obj.masks.update(np.zeros(self.frame_size, dtype=bool), self.current_frame_idx)
        self.apply_mask_to_active(obj)

        propagation = False
        try:
            for frame_response in self.sam_api.propagate_in_video(prop_req):
                if not self.playing:
                    self.stop_propagation()
                    break
                frame_idx = frame_response.frame_index
                #TODO: Change the name val
                for val in frame_response.results:
                    obj_id = val.object_id
                    mask_rle = val.mask
                    # mask = decode_masks({"counts": mask_rle.counts, "size": mask_rle.size})
                    mask = mask_rle.array
                    # update AnnObject's mask
                    obj = self.get_object_by_id(obj_id)
                    obj.masks.update(mask.astype(bool), frame_idx)

                self.goto_frame(frame_idx, propagation=propagation)
                propagation = True
                QtCore.QCoreApplication.processEvents()

        except RuntimeError as e:
            self.status.setText(f"Stopped due to error: {str(e)[:80]}...")
            self.stop_propagation()
            self.toggle_play()


    def stop_propagation(self):
        session = self.sam_api._InferenceAPI__get_session(self.session_id)
        self.sam_api.predictor.reset_state(session["state"])
        torch.cuda.empty_cache()


    def apply_point_to_active(self, obj: AnnObject):
        frame_idx, x, y, label = obj.prompts.points[-1]

        add_req = AddPointsRequest(
            type="add_points",
            session_id=self.session_id,
            frame_index=frame_idx,
            object_id=obj.id,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([label], dtype=np.int32),
            clear_old_points=False
        )
        response = self.sam_api.add_points(add_req, normalize_coords=True)
        return response
    

    def apply_mask_to_active(self, obj: AnnObject):
        mask = obj.masks.get(self.current_frame_idx)
        if mask is None:
            return
        mask = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
        mask = SamMask(counts=mask["counts"], size=mask["size"])
        
        add_req = AddMaskRequest(
            type="add_mask",
            session_id=self.session_id,
            frame_index=self.current_frame_idx,
            object_id=obj.id,
            mask=mask
        )
        response = self.sam_api.add_mask(add_req)
        return response


    def undo_points(self):
        # Undo points in a frame for the chosen class before starting propagation
        # But the object is still in SAM2, and might pop up in the future
        active_object_id = getattr(self.video_label, "active_object_id", None)
        if active_object_id is None:
            self.status.setText("Error: No class selected")
            return True
        obj = self.get_object_by_id(active_object_id)
        
        undo_req = ClearPointsInFrameRequest(
            type="clear_points_in_frame",
            session_id=self.session_id,
            frame_index=self.current_frame_idx,
            object_id=active_object_id
        )
        response = self.sam_api.clear_points_in_frame(undo_req)
        obj.prompts.clear(self.current_frame_idx)
        obj.masks.clear(self.current_frame_idx)
        self.goto_frame(self.current_frame_idx)
        self.status.setText(f"Points in this frame are cleared.")
        return response
    

    def clear_annotations(self):
        # Clear all saved masks in the future frames from self.objects
        # Does not remove the object from SAM2 state
        for idx in range(self.current_frame_idx, self.total_frames):
            for obj in self.objects:
                obj.prompts.clear(idx)
                obj.masks.clear(idx)
        self.goto_frame(self.current_frame_idx)
        self.status.setText(f"Masks in future frames are cleared.")


    def export_annotations(self):
        # The visible overlay masks in AnnObject and combined exported
        mask_dir = os.path.join(self.output_ann_path, os.path.basename(self.video_path).split(".")[0])
        os.makedirs(mask_dir, exist_ok=True)
        
        for idx in range(self.total_frames):
            mask_all = np.full((self.frame_size[1], self.frame_size[0]), 0, dtype=np.uint8)
            mask_path = os.path.join(mask_dir, f"{idx:010d}.png")

            for obj in self.objects:
                if obj.id == "0":
                    continue
                mask_obj = obj.masks.get(idx)
                if mask_obj is not None:
                    mask_all[mask_obj] = int(obj.id)

            mask_all = Image.fromarray(mask_all)
            mask_all.putpalette(np.uint8(self.mask_palette))
            mask_all.save(mask_path)
        self.status.setText(f"Masks are exported to: {mask_dir}")

                
    def render_overlays(self, frame_rgb: np.ndarray) -> np.ndarray:
        canvas = frame_rgb.copy()
        if canvas.dtype != np.uint8:
            canvas = (canvas * 255).astype(np.uint8) if canvas.max() <= 1.0 else canvas.astype(np.uint8)
        h, w, _ = canvas.shape
        canvas_f = canvas.astype(np.float32)

        for obj in self.objects:
            # Skip the overlay for background class
            if obj.id == "0":
                continue
            
            color = tuple(int(c) for c in obj.color)
            mask = obj.masks.get(self.current_frame_idx)
            if mask is not None:
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                alpha = 0.4
                canvas_f = np.where(mask[:, :, None], (1 - alpha) * canvas_f + alpha * np.array(color, dtype=np.float32), canvas_f)
                contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas_f, contours, -1, color, 2)

            for p in obj.prompts.points:
                frame_idx, x, y, label = p
                if self.current_frame_idx == frame_idx:
                    cv2.circle(canvas_f, (int(x), int(y)), 3, color, -1 if label == 1 else 2, lineType=cv2.LINE_AA)

        canvas_out = np.clip(canvas_f, 0, 255).astype(np.uint8)
        return canvas_out


    def eventFilter(self, obj, event):
        # Handle mouse and resize events for video_label
        # Middle-click to play/pause
        if obj == self.video_label:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.MiddleButton:
                    self.btn_play.click()
                    return True 
                return self.video_label_mousePressEvent(event)
        return super().eventFilter(obj, event)
    

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Left and right arrow keys for frame navigation
        # Number keys for class selection
        if not self.playing:
            if event.key() == QtCore.Qt.Key_Left:
                self.move_backward_btn.click()
                event.accept()
                return
            if event.key() == QtCore.Qt.Key_Right:
                self.move_forward_btn.click()
                event.accept()
                return
            if QtCore.Qt.Key_1 <= event.key() <= QtCore.Qt.Key_8:
                num = event.key() - QtCore.Qt.Key_1
                for i, btn in enumerate(self.btn_class.buttons()):
                    if i != num:
                        continue
                    btn.click()
                    event.accept()
                    return
        super().keyPressEvent(event)


    def video_label_set_frame(self, rgb_frame: np.ndarray):
        if rgb_frame is None:
            return
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.current_qimage = qimg.copy()
        pix = QtGui.QPixmap.fromImage(self.current_qimage)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    def video_label_mousePressEvent(self, event: QtGui.QMouseEvent):
        # Cannot add points while playing/propagating masks
        if self.playing:
            return True

        active_object_id = getattr(self.video_label, "active_object_id", None)
        if active_object_id is None:
            self.status.setText("Error: No class selected")
            return True
        obj = self.get_object_by_id(active_object_id)
        pos = event.pos()
        x, y = pos.x(), pos.y()
        
        if event.button() == QtCore.Qt.LeftButton:
            obj.prompts.points.append((self.current_frame_idx, x, y, 1))
            self.status.setText(f"Added positive point to {obj.name} in frame {self.current_frame_idx+1}: ({x},{y})")

        elif event.button() == QtCore.Qt.RightButton:
            obj.prompts.points.append((self.current_frame_idx, x, y, 0))
            self.status.setText(f"Added negative point to {obj.name} in frame {self.current_frame_idx+1}: ({x},{y})")

        response = self.apply_point_to_active(obj)
        response_mask = {d.object_id: d for d in response.results}.get(obj.id)
        # obj.masks.update(decode_masks({"counts": response_mask.mask.counts, "size": response_mask.mask.size}).astype(bool), self.current_frame_idx)
        obj.masks.update((response_mask.mask.array).astype(bool), self.current_frame_idx)
        self.goto_frame(self.current_frame_idx)
        return True


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = SAM2_Video_Annotator()
    win.show()
    sys.exit(app.exec_())