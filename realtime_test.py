# =========================
# Webcam inference for Age + Gender
# =========================
import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ---- Model definition must MATCH training ----
class AgeGenderResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        in_feats = 512
        self.gender_head = nn.Linear(in_feats, 1)
        self.age_head = nn.Linear(in_feats, 1)

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        g = self.gender_head(feats)  # logits
        a = self.age_head(feats)     # raw age
        return g, a

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeGenderResNet18().to(device)
model.load_state_dict(torch.load("best_age_gender.pth", map_location=device))
model.eval()

# Same normalization as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0) # It's usually 0 but change it if (ouf of index error) occurs .
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    h, w = frame.shape[:2]
    cv2.flip(frame, 1, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_t = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            g_logit, a_pred = model(img_t)
            gender_prob = round(torch.sigmoid(g_logit).item(), 2)
            gender_label = "Female" if gender_prob > 0.5 else "Male"
            age_val = float(a_pred.item())
            age_val = max(0.0, min(116.0, age_val))  # clamp to plausible range

        # Draw bbox & labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = f"{gender_label} ({(gender_prob*100) if gender_prob > 0.5 else (100 - gender_prob*100)}%), Age: {age_val:.0f}"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age & Gender", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
