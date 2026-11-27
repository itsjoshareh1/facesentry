# PyInstaller recipe for FaceSentry.app. Build: pyinstaller FaceSentry.spec
from PyInstaller.utils.hooks import collect_data_files

a = Analysis(
    ["app_main.py"],
    pathex=["."],
    datas=collect_data_files("cv2") + [("models/yolov8n-face.onnx", "models"), ("models/face_recognition_sface_2021dec.onnx", "models")],
    hiddenimports=["enroll", "watch", "facesentry"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FaceSentry", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="FaceSentry")
app = BUNDLE(
    coll,
    name="FaceSentry.app",
    bundle_identifier="com.itsjoshareh1.facesentry",
    info_plist={
        "CFBundleName": "FaceSentry",
        "CFBundleShortVersionString": "1.3.0",
        "NSHighResolutionCapable": True,
        "NSCameraUsageDescription": "FaceSentry uses the camera to recognize your face and spot unknown people at your laptop. Nothing leaves your Mac.",
    },
)
