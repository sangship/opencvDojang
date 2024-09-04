#######################################################################################################################
# 개발 환경 노트북 사양:
# 프로세서: Intel Core i5-1340P (13세대)는 
# **12개 코어(4개 성능 코어 + 8개 효율 코어)**로 구성되어 있습니다. 
# 이 CPU는 멀티코어 작업에 최적화되어 있어 멀티 프로세스에서 큰 이점을 발휘할 수 있습니다.

# 메모리: 16GB RAM은 멀티 프로세스와 멀티 스레드 둘 다 충분히 지원할 수 있는 용량입니다. 
# 다만, 프로세스 간의 메모리 복사나 데이터 전송이 필요한 멀티 프로세스 환경에서는 
# 일부 추가 메모리 사용이 있을 수 있습니다.

# 운영 체제: 64비트 OS는 멀티코어, 멀티프로세싱 환경을 지원하며, 
# 스레드 및 프로세스 간 메모리 공간을 잘 관리합니다.
#######################################################################################################################

# 개발 환경에서 구현시에 멀티 프로세스v멀티 스레드 비교

# 멀티 프로세스 방식의 장점
    # GIL 문제 회피:
    # Python의 GIL(Global Interpreter Lock)로 인해 멀티 스레딩이 제한적이지만, 
    # 멀티 프로세싱은 이를 회피할 수 있습니다. 
    # 특히, CPU 코어가 많은 환경에서는 멀티 프로세스를 사용하면 성능이 대폭 향상됩니다.

    # CPU 성능 최대 활용:
    # 13세대 Intel Core i5-1340P는 12개의 코어(4P+8E)로 구성되어 있어 
    # 멀티코어 성능을 활용하기에 좋습니다. 
    # 멀티 프로세스 환경에서는 각 프로세스가 독립적으로 실행되므로, 
    # 비디오 캡처와 저장 작업을 병렬로 실행할 때 더 많은 성능을 발휘할 수 있습니다.
    
    # 작업 간 간섭 최소화:
    # 프로세스 간 메모리 공유가 없기 때문에 작업 간 간섭이 적고 안정성이 높습니다. 
    # 비디오 녹화와 파일 저장을 서로 다른 프로세스로 분리하면, 
    # 저장 작업이 캡처 작업에 영향을 주지 않습니다.
    
# 멀티 스레드 방식의 장점:
    # 적은 메모리 사용량:
    # 멀티 스레드는 프로세스 간 메모리 공유를 통해 메모리 사용량을 줄일 수 있습니다. 
    # 따라서, 메모리 사용량을 최소화하고 싶다면 멀티 스레드를 고려할 수 있습니다.
    
    # I/O 바운드 작업에서 효율적:
    # 파일 저장과 같은 I/O 바운드 작업을 동시에 처리하는 데 유리합니다. 
    # 다만, Python의 GIL 문제로 인해 CPU 바운드 작업에서는 효율이 떨어질 수 있습니다.
    
# 결론 및 추천 방식
# 멀티 프로세스 방식이 더 적합합니다.

# 이유:
# 멀티코어 환경에서의 성능 극대화: 
#   멀티코어 CPU의 장점을 최대한 활용하여, 
#   비디오 캡처와 저장 작업을 병렬로 처리하여 성능을 극대화할 수 있습니다.

# GIL 문제 회피: 
#   Python의 GIL 문제를 피하여 CPU 바운드 작업에서도 성능을 제대로 활용할 수 있습니다.

# 프로세스 안정성: 
#   프로세스 간 독립성 덕분에 하나의 작업이 실패해도 다른 작업에 영향을 주지 않습니다.


##############################################################################################################
# +-------------------------------------------+
# |            Main Process (main)            |
# |-------------------------------------------|
# |                                           |
# |  1. Create Pipe for Inter-process         |
# |     Communication (parent_conn,           |
# |     child_conn)                           |
# |                                           |
# |  2. Start Capture Process                 |
# |     (video_capture_process)               |
# |                                           |
# |  3. Start Writer Process                  |
# |     (video_writer_process)                |
# |                                           |
# |  4. Wait for Processes to Complete        |
# |     (join capture_process, writer_process)|
# +-------------------------------------------+
#              |                   |
#              | (Pipe: Frame Data | "END")
#              |                   |
#              v                   v
# +-----------------------+    +--------------------------+
# |  Video Capture Process|    |   Video Writer Process   |
# | (video_capture_process)|   | (video_writer_process)   |
# |-----------------------|    |--------------------------|
# |                       |    |                          |
# | 1. Initialize webcam  |    | 1. Create/Select Folder  |
# |    (cv2.VideoCapture) |    |    (Every 2 minutes)     |
# |                       |    |                          |
# | 2. Loop to capture    |    | 2. Create new video file |
# |    frames every 5 sec |    |    (Every 5 seconds)     |
# |                       |    |                          |
# | 3. Display frames on  |    | 3. Receive frame data    |
# |    screen (cv2.imshow)|    |    via Pipe              |
# |                       |    |                          |
# | 4. Send frames to the |    | 4. Write frames to video |
# |    writer process via |    |    file (cv2.VideoWriter)|
# |    Pipe               |    |                          |
# |                       |    | 5. Repeat until "END"    |
# | 5. Check for ESC key  |    |    signal is received    |
# |    input to exit loop |    |                          |
# |                       |    | 6. Release VideoWriter   |
# | 6. Send "END" signal  |    |                          |
# |    to writer process  |    |                          |
# |    via Pipe           |    | 7. Manage Storage Size   |
# |                       |    |    (Delete oldest folder)|
# | 7. Release webcam and |    |                          |
# |    close display      |    |                          |
# +-----------------------+    +--------------------------+


##############################################################################################################
# 다이어그램 주요 구성 요소
# Main Process (main):
# 파이프를 생성하여 두 프로세스 간의 데이터 통신을 관리합니다.
# video_capture_process와 video_writer_process 두 개의 프로세스를 시작하고, 두 프로세스가 종료될 때까지 대기합니다.

# Video Capture Process (video_capture_process):
# 웹캠을 초기화하여 프레임을 캡처하고, Pipe를 통해 비디오 프레임 데이터를 video_writer_process로 전송합니다.
# 60초마다 새로운 비디오 캡처 루프가 시작되며, 프레임을 전송합니다.
# ESC 키 입력을 감지하여 프로그램 종료 신호 "END"를 Pipe를 통해 전송합니다.

# Video Writer Process (video_writer_process):
# 60분마다 새 폴더를 생성하고, 각 폴더에 60초마다 새로운 비디오 파일을 저장합니다.
# Pipe를 통해 프레임 데이터를 수신하여 비디오 파일로 저장합니다.
# 수신된 프레임이 "END" 신호일 경우, 비디오 파일을 저장하고 프로세스를 종료합니다.
# 저장 공간을 관리하기 위해 폴더의 크기를 확인하고, 최대 크기(500MB)를 초과하면 가장 오래된 폴더를 삭제합니다.


##############################################################################################################
# 코드의 핵심 흐름

# 프로세스 생성 및 통신:
# 메인 프로세스는 두 개의 멀티프로세스를 생성하고, 파이프를 통해 프레임 데이터를 전송합니다.

# 동영상 캡처와 저장:
# video_capture_process가 60초마다 웹캠에서 프레임을 캡처하여 video_writer_process로 전송합니다.
# video_writer_process는 60분마다 새로운 폴더를 생성하고, 
# 캡처된 비디오 프레임을 60초 동안 저장한 후 새로운 비디오 파일을 생성합니다.

# 프로그램 종료 처리:
# 사용자가 ESC 키를 누르면 캡처 프로세스가 "END" 신호를 전송하여 프로그램이 종료됩니다.
##############################################################################################################

import cv2
import os
import shutil
import time
from datetime import datetime, timedelta
from multiprocessing import Process, Pipe

# 블랙박스 설정
MAX_FOLDER_SIZE = 500 * 1024 * 1024  # 500MB 최대 폴더 크기
RECORD_DURATION = 60  # 60 seconds (비디오 녹화 길이)

def get_current_time_str():
    """현재 시간을 'YYYYMMDD_HHMMSS' 형식으로 반환"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_current_folder_name():
    """현재 시간을 기준으로 60분 단위 폴더 이름 'YYYYMMDD_HH' 생성"""
    now = datetime.now()
    rounded_time = now - timedelta(minutes=now.minute % 60, seconds=now.second, microseconds=now.microsecond)
    return rounded_time.strftime('%Y%m%d_%H')

def create_folder_if_not_exists(folder_path):
    """폴더가 존재하지 않으면 생성"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_folder_size(folder_path):
    """폴더의 총 크기를 계산하여 반환"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def remove_oldest_folder(base_path):
    """가장 오래된 폴더를 삭제하여 스토리지 용량 확보"""
    folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    if folders:
        shutil.rmtree(os.path.join(base_path, folders[0]))

def check_and_manage_storage(base_path):
    """전체 폴더 크기를 확인하고, 최대 크기를 초과하면 오래된 폴더 삭제"""
    total_size = 0
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            total_size += get_folder_size(folder_path)
    if total_size > MAX_FOLDER_SIZE:
        remove_oldest_folder(base_path)

def video_capture_process(pipe, fps):
    """비디오 프레임을 캡처하고 다른 프로세스로 전송하는 프로세스"""
    cap = cv2.VideoCapture(0)  # 기본 웹캠(장치 0)에서 비디오 캡처 시작
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        pipe.send("END")  # 종료 신호 전송
        pipe.close()
        return

    # 웹캠 프레임 크기 및 FPS 설정
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Frame Size: {frameSize}")  # 디버깅용으로 프레임 크기 출력

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # fps가 0일 경우 기본값 설정
        fps = 30  # 명시적 프레임 속도 설정 (초당 30프레임)
    
    while True:
        # 녹화 프레임 수를 설정하여 정확히 60초 녹화
        total_frames_to_record = RECORD_DURATION * fps  # 60초 녹화를 위해 필요한 프레임 수
        frames_recorded = 0

        # 60초간 녹화
        while frames_recorded < total_frames_to_record:
            retval, frame = cap.read()  # 프레임 캡처
            if not retval:
                print("Error: Failed to capture frame.")
                break
            
            pipe.send(frame)  # 캡처한 프레임을 파이프로 전송
            frames_recorded += 1  # 녹화된 프레임 수 증가
            
            # 화면에 프레임을 표시
            cv2.imshow('frame', frame)
            
            # ESC 키 입력을 감지하여 종료
            if cv2.waitKey(1) & 0xFF == 27:
                pipe.send("END")  # 종료 신호 전송
                cap.release()  # 웹캠 해제
                cv2.destroyAllWindows()  # 모든 창 닫기
                pipe.close()  # 파이프 닫기
                return

def video_writer_process(pipe, base_path, fps):
    """비디오 파일로 프레임을 저장하는 프로세스"""
    codec = 'XVID'  # 사용할 코덱
    frameSize = (640, 480)  # 기본 프레임 크기 설정 (수신 후 조정 가능)

    while True:
        # 60분마다 폴더를 생성하여 새로운 폴더로 변경
        new_folder_name = get_current_folder_name()  # 현재 시간 기준으로 새로운 폴더 이름 생성
        new_folder_path = os.path.join(base_path, new_folder_name)
        create_folder_if_not_exists(new_folder_path)  # 새 폴더가 없으면 생성

        # 비디오 파일명 생성
        file_name = get_current_time_str() + '.avi'
        file_path = os.path.join(new_folder_path, file_name)
        
        # 비디오 쓰기 객체 생성
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(file_path, fourcc, fps, frameSize)

        if not out.isOpened():
            print(f"Error: Could not open VideoWriter for {file_path}")
            return

        frame_count = 0  # 프레임 수 초기화
        while frame_count < RECORD_DURATION * fps:
            frame = pipe.recv()  # 파이프를 통해 프레임 수신
            if isinstance(frame, str) and frame == "END":  # 녹화 종료 신호 수신 시 종료
                out.release()  # 비디오 파일 닫기
                print(f"Video saved to {file_path}")
                check_and_manage_storage(base_path)  # 스토리지 용량 관리
                return  # 프로세스 종료
            if frame is not None:
                out.write(frame)  # 비디오 파일에 프레임 저장
                frame_count += 1  # 프레임 수 증가
        
        out.release()  # 비디오 파일 닫기
        print(f"Video saved to {file_path}")
        check_and_manage_storage(base_path)  # 스토리지 용량 관리

if __name__ == "__main__":
    base_path = "./recordings"  # 녹화 파일을 저장할 기본 경로
    fps = 30  # 기본 프레임 속도 설정
    
    # 파이프 생성
    parent_conn, child_conn = Pipe()

    # 프로세스 생성
    writer_process = Process(target=video_writer_process, args=(parent_conn, base_path, fps))
    capture_process = Process(target=video_capture_process, args=(child_conn, fps))

    # 프로세스 시작
    writer_process.start()
    capture_process.start()

    # 프로세스가 종료될 때까지 대기
    capture_process.join()
    writer_process.join()
