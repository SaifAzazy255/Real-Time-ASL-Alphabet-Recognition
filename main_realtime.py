import cv2
import mediapipe as mp
import pandas as pd
import pickle
import warnings

# إخفاء التحذيرات
warnings.filterwarnings("ignore")

# 1. تحميل الموديل (الذي يحتوي على الـ Scaler والـ Classifier معاً)
print("Loading the model pipeline...")
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)
print("Model loaded successfully! Starting camera...")

# 2. تهيئة MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 3. تجهيز أسماء الأعمدة لتطابق بيانات التدريب تماماً
columns = []
for i in range(21):
    columns.extend([f'x{i}', f'y{i}'])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # عكس الصورة لتكون مرآة
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # استخراج معالم اليد
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # استخراج الـ 42 إحداثي بنفس الطريقة المتبعة في التجميع
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([int(lm.x * w), int(lm.y * h)])
            
            # تحويل البيانات إلى DataFrame بنفس أسماء الأعمدة لمنع أي Data Mismatch
            X_live = pd.DataFrame([row], columns=columns)
            
            # التنبؤ بالحرف
            prediction = pipeline.predict(X_live)[0]
            
            # رسم مربع أسود شفاف كخلفية للنص (لتحسين الرؤية)
            cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), cv2.FILLED)
            
            # كتابة النتيجة على الشاشة
            cv2.putText(frame, f"Sign: {prediction}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Sign Language Recognition - Real Time", frame)
    
    # الخروج عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()