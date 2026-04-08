# RF-DETR Endoscopy (실시간 대장 내 용종 검출 시스템)

건양대학교 창의적 종합설계 프로젝트 (Team OmniDow) - **RF-DETR 기반 실시간 의료 객체 탐지 데스크탑 애플리케이션**입니다.

## 📌 프로젝트 소개 (Project Overview)
본 프로젝트는 내시경(Endoscopy) 영상 및 웹캠 스트림에서 대장 내 용종(Polyp)을 실시간으로 추론 및 검출하는 **PyQt5 기반 소프트웨어**입니다. Roboflow Inference SDK(또는 Local Inference)를 연동하여 고성능 비전 모델(RF-DETR 등)을 실시간으로 서빙하며, 실제 임상/의료 환경에 맞춘 '스마트 자동 클립 추출' 및 'AI 예비 소견서 자동 생성' 기능을 탑재했습니다.

## ✨ 핵심 기능 (Key Features)

1. **실시간 용종 탐지 (Real-Time Object Detection)**
   - 웹캠(실시간 스트리밍) 및 로컬 동영상 파일(`.mp4`, `.avi` 등)을 입력 소스로 지원.
   - 비동기 처리(QThread)를 통해 GUI 멈춤 없이 쾌적한 실시간 Bounding Box 렌더링.

2. **스마트 하이라이트 클립 자동 추출 (Automated Clip Saving)**
   - 용종이 설정된 시간(기본 4초) 이상 연속으로 탐지될 경우 이벤트로 간주.
   - 감지 기점 전/후의 영상(기본 5초)을 덧붙인 **이벤트 하이라이트 비디오 클립**을 자동으로 추출하여 저장합니다.

3. **AI 임상 소견서 자동 생성 (GPT-driven Clinical Report)**
   - 스마트 클립이 추출될 때마다 OpenAI API를 백그라운드로 호출합니다.
   - 환자 정보, 발견 시점, 병변의 신뢰도(Confidence) 데이터를 종합하여 작성된 **AI 예비 요약 보고서(.txt)**를 클립과 함께 자동 병합 저장합니다.

4. **환자 정보 관리 (Patient Info Management)**
   - 환자의 이름, 나이, 성별, 생년월일 데이터를 입력받아 시스템에 연작.
   - 실시간 화면 및 저장되는 영상/클립에 **데이터 오버레이(Data Overlay)**를 씌우며, 추후 `Excel` 혹은 `CSV` 포맷으로 정보 관리 로깅.

5. **수동 녹화 및 스냅샷 (Manual Recording & Snapshot)**
   - `녹화 시작/종료` 버튼으로 수동 전체 영상 캡처 기능.
   - 키보드 `S`키를 눌러 중요 순간을 고화질 캡처. 모든 데이터는 지정된 환자별 고유 폴더에 체계적으로 기록됩니다.

## 🛠️ 기술 스택 (Tech Stack)
- **Application Logic**: Python 3.x
- **GUI Framework**: PyQt5
- **Computer Vision**: OpenCV (`cv2`)
- **Model Inference**: Roboflow Inference SDK (`inference`)
- **Generative AI**: OpenAI API (for Auto Reporting)
- **Data Export**: `csv`, `openpyxl`

## ⚙️ 실행 방법 및 환경 변수 (Installation & Configs)

### 1. 의존성 설치
```bash
pip install PyQt5 opencv-python inference-sdk openpyxl openai
```

### 2. 주요 환경 변수 (Environment Variables)
애플리케이션은 다양한 환경 변수를 통해 동작을 튜닝할 수 있습니다:
- `ROBOFLOW_API_KEY` : 추론에 사용할 Roboflow API 키 (필수)
- `OPENAI_API_KEY` : AI 보고서 생성용 API 키 (활성화 시 필수)
- `RFDETR_CONFIDENCE` : 탐지 임계값 (기본값: `0.3`)
- `RFDETR_CLIP_TRIGGER` : 스마트 클립 저장을 트리거할 최소 연속 인식 시간 단위(초) (기본값: `4`)
- `RFDETR_ENABLE_REPORTS` : AI 보고서 생성 기능 활성화 여부 (`1` 설정 시 활성화)

### 3. 애플리케이션 실행
```bash
python RFDETR_v3.py
```

## 👨‍💻 기여자 (Contributors)
- **안진경 (Team Leader & Vision Pipeline Engineer)**
- 본 프로젝트는 건양대학교 제17회 캡스톤디자인 경진대회 등에서 수상한 이력이 포함된 학술/엔지니어링 결과물입니다.
