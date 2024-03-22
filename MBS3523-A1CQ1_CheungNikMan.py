import cv2
import threading
import serial

last_sensor_data = "Waiting for data..."

port = 'COM11'
baud_rate = 115200
ser = serial.Serial(port, baud_rate)


def read_sensor():
    global last_sensor_data
    while True:
        if ser.inWaiting() > 0:
            last_sensor_data = ser.readline().decode().strip()


def stream_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        sensor_data = last_sensor_data

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{sensor_data}", (10, 30), font, 0.67, (0, 0, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sensor_thread = threading.Thread(target=read_sensor, daemon=True)
    sensor_thread.start()

    webcam_thread = threading.Thread(target=stream_webcam, daemon=True)
    webcam_thread.start()
    webcam_thread.join()